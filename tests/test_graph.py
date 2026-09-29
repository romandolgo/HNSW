"""Проверки общего ядра: метрика, обход слоя и отбор соседей."""

import numpy as np
import pytest

from graph import search_layer, select_heuristic, select_simple
from metrics import euclidean


@pytest.fixture
def complete_layer() -> tuple[np.ndarray, dict[int, list[int]]]:
    """Двадцать случайных точек, связанных все со всеми.

    На полном графе жадный обход не может застрять, поэтому любое
    расхождение с полным перебором означает ошибку в самом алгоритме, а не
    в структуре графа.
    """
    points = np.random.default_rng(0).uniform(size=(20, 2))
    layer = {i: [j for j in range(20) if j != i] for i in range(20)}
    return points, layer


def test_euclidean_matches_numpy() -> None:
    """euclidean совпадает с np.linalg.norm на случайных парах точек."""
    rng = np.random.default_rng(0)
    a, b = rng.uniform(size=(50, 3)), rng.uniform(size=(50, 3))
    for x, y in zip(a, b, strict=True):
        assert euclidean(x, y) == pytest.approx(np.linalg.norm(x - y))


def test_euclidean_broadcasts_over_a_batch() -> None:
    """При массиве точек вторым аргументом возвращается массив расстояний."""
    rng = np.random.default_rng(1)
    q, batch = rng.uniform(size=3), rng.uniform(size=(7, 3))
    got = euclidean(batch, q)
    assert got.shape == (7,)
    assert got == pytest.approx(np.linalg.norm(batch - q, axis=-1))


def test_search_layer_is_exact_with_large_ef(complete_layer) -> None:
    """При ef >= n обход полного слоя совпадает с полным перебором."""
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    found = [i for _, i in search_layer(q, [0], 20, layer, points, euclidean)]
    exact = np.argsort(np.linalg.norm(points - q, axis=1)).tolist()
    assert found == exact


def test_search_layer_respects_ef(complete_layer) -> None:
    """Результат не длиннее ef и отсортирован по возрастанию расстояния."""
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    for ef in (1, 3, 7, 20):
        found = search_layer(q, [0], ef, layer, points, euclidean)
        assert len(found) == ef
        distances = [d for d, _ in found]
        assert distances == sorted(distances)


def test_search_layer_computes_each_distance_once(complete_layer) -> None:
    """Число вызовов метрики равно числу посещённых вершин.

    Расстояние до вершины вычисляется в момент первой встречи и дальше
    едет в кортеже вместе с индексом. Если счётчик покажет больше, значит
    где-то потерялась проверка visited или расстояние пересчитывается.
    """
    points, layer = complete_layer
    calls = 0

    def counting(a, b):
        nonlocal calls
        calls += 1
        return euclidean(a, b)

    search_layer(np.array([0.5, 0.5]), [0], 20, layer, points, counting)
    assert calls == len(points)


def test_search_layer_accepts_several_entry_points(complete_layer) -> None:
    """Точек входа может быть несколько — строка 17 Alg. 1 передаёт весь W."""
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    from_one = search_layer(q, [0], 5, layer, points, euclidean)
    from_many = search_layer(q, [0, 7, 13], 5, layer, points, euclidean)
    assert from_one == from_many


def test_search_layer_does_not_mutate_the_layer(complete_layer) -> None:
    """Поиск ничего не меняет в графе."""
    points, layer = complete_layer
    before = {v: list(ns) for v, ns in layer.items()}
    search_layer(np.array([0.5, 0.5]), [0], 10, layer, points, euclidean)
    assert layer == before


def test_trace_records_the_walk_without_changing_the_result(complete_layer) -> None:
    """trace не влияет на ответ и пишет вершины в порядке разворачивания."""
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    trace: list[int] = []
    with_trace = search_layer(q, [0], 5, layer, points, euclidean, trace=trace)
    without = search_layer(q, [0], 5, layer, points, euclidean)

    assert with_trace == without
    assert trace
    assert len(trace) == len(set(trace)), "вершина развёрнута дважды"


# --- отбор соседей -------------------------------------------------------
#
# Геометрия для эвристики: база в начале координат, r — ближайший кандидат,
# shadowed лежит за ним по той же прямой, diverse — сбоку.
#
#   diverse (0, 1.2)
#        |
#     база ---- r (1, 0) ---- shadowed (1.5, 0)

HEURISTIC_POINTS = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0], [0.0, 1.2]])
BASE, NEAR, SHADOWED, DIVERSE = 0, 1, 2, 3


@pytest.fixture
def heuristic_candidates() -> list[tuple[float, int]]:
    """Кандидаты с расстояниями до базовой вершины, по возрастанию."""
    base = HEURISTIC_POINTS[BASE]
    return sorted(
        (float(euclidean(HEURISTIC_POINTS[i], base)), i)
        for i in (NEAR, SHADOWED, DIVERSE)
    )


def test_select_simple_returns_m_nearest(heuristic_candidates) -> None:
    """Alg. 3 берёт ровно m ближайших, в порядке возрастания расстояния."""
    got = select_simple(
        HEURISTIC_POINTS[BASE], heuristic_candidates, 2, {}, HEURISTIC_POINTS, euclidean
    )
    assert got == [NEAR, DIVERSE]


def test_select_simple_returns_everything_when_m_exceeds_candidates(
    heuristic_candidates,
) -> None:
    """При m больше числа кандидатов возвращается всё, без ошибки."""
    got = select_simple(
        HEURISTIC_POINTS[BASE], heuristic_candidates, 99, {}, HEURISTIC_POINTS, euclidean
    )
    assert sorted(got) == [NEAR, SHADOWED, DIVERSE]


def test_select_heuristic_drops_the_shadowed_candidate(heuristic_candidates) -> None:
    """Alg. 4 отбрасывает кандидата, закрытого уже выбранным соседом.

    shadowed ближе к near (0.5), чем к базе (1.5), поэтому прямое ребро к
    нему избыточно: жадный поиск дойдёт через near. diverse наоборот ближе
    к базе (1.2), чем к near (1.56), и открывает новое направление.
    """
    got = select_heuristic(
        HEURISTIC_POINTS[BASE], heuristic_candidates, 3, {}, HEURISTIC_POINTS, euclidean
    )
    assert got == [NEAR, DIVERSE]
    assert SHADOWED not in got


def test_keep_pruned_fills_the_result_back_to_m(heuristic_candidates) -> None:
    """keep_pruned добирает отвергнутых, начиная с ближайших."""
    got = select_heuristic(
        HEURISTIC_POINTS[BASE],
        heuristic_candidates,
        3,
        {},
        HEURISTIC_POINTS,
        euclidean,
        keep_pruned=True,
    )
    assert got == [NEAR, DIVERSE, SHADOWED]


def test_selectors_are_interchangeable(heuristic_candidates) -> None:
    """Обе стратегии принимают одни и те же аргументы по именам.

    На это опирается bench: селектор переключается одним параметром.
    """
    kwargs = dict(
        q=HEURISTIC_POINTS[BASE],
        candidates=heuristic_candidates,
        m=2,
        layer={},
        points=HEURISTIC_POINTS,
        metric=euclidean,
    )
    for selector in (select_simple, select_heuristic):
        assert len(selector(**kwargs)) <= 2
