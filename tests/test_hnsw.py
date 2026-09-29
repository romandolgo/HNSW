"""Проверки многоуровневого HNSW."""

import numpy as np
import pytest

import brute
import data
from hnsw import HNSW

N = 300
K = 5
M = 8


@pytest.fixture(scope="module")
def points() -> np.ndarray:
    return data.uniform(N, seed=0)


@pytest.fixture(scope="module")
def index(points) -> HNSW:
    """m_l завышен против умолчания, чтобы слоёв было несколько.

    При рекомендованном 1/ln(8) на трёхстах точках верхних слоёв почти не
    возникает, и спуск остался бы непроверенным.
    """
    graph = HNSW(points, m=M, ef_construction=40, m_l=1.0, seed=5)
    graph.build()
    return graph


def test_hierarchy_is_actually_built(index) -> None:
    """Иначе все остальные тесты проверяли бы одноуровневый граф."""
    assert index.max_level >= 2


def test_layer_count_matches_max_level(index) -> None:
    assert len(index.layers) == index.max_level + 1


def test_layer_zero_holds_every_point(index, points) -> None:
    assert len(index.layers[0]) == len(points)


def test_layer_membership_is_nested(index) -> None:
    """Вершина слоя l присутствует во всех слоях ниже."""
    for lc in range(index.max_level, 0, -1):
        assert set(index.layers[lc]) <= set(index.layers[lc - 1])


def test_layers_shrink_going_up(index) -> None:
    sizes = [len(layer) for layer in index.layers]
    assert sizes == sorted(sizes, reverse=True)


def test_entry_point_is_in_the_top_layer(index) -> None:
    """Иначе спуск упадёт на первом же обращении к списку соседей."""
    assert index.entry_point in index.layers[index.max_level]


def test_degree_within_bounds(index) -> None:
    """Нулевой слой ограничен m_max0, верхние — m_max."""
    for lc, layer in enumerate(index.layers):
        cap = index.m_max0 if lc == 0 else index.m_max
        assert max((len(ns) for ns in layer.values()), default=0) <= cap


def test_no_self_loops_and_no_duplicates(index) -> None:
    for layer in index.layers:
        for vertex, neighbours in layer.items():
            assert vertex not in neighbours
            assert len(neighbours) == len(set(neighbours))


def test_neighbours_belong_to_the_same_layer(index) -> None:
    """Ребро не может вести на вершину, которой в этом слое нет."""
    for layer in index.layers:
        for neighbours in layer.values():
            assert all(neighbour in layer for neighbour in neighbours)


def test_defaults_follow_the_paper(points) -> None:
    """m_max -> m, m_max0 -> 2m, m_l -> 1/ln(m); раздел 4.1."""
    graph = HNSW(points, m=M, ef_construction=40)
    assert graph.m_max == M
    assert graph.m_max0 == 2 * M
    assert graph.m_l == pytest.approx(1 / np.log(M))


def test_level_distribution_matches_m_l() -> None:
    """Среднее число слоёв у вершины близко к 1 / (1 - exp(-1 / m_l)).

    Формула (1) и следующий за ней абзац раздела 4.1. При рекомендованном
    m_l = 1 / ln(M) выражение сворачивается в M / (M - 1).
    """
    graph = HNSW(np.zeros((2, 2)), m=M, ef_construction=4, seed=11)
    levels = np.fromiter(
        (graph._random_level() for _ in range(50_000)), dtype=np.int64
    )

    assert (levels + 1).mean() == pytest.approx(M / (M - 1), rel=0.02)
    assert (levels == 0).mean() == pytest.approx(1 - 1 / M, rel=0.01)
    assert levels.min() == 0


def test_same_seed_gives_the_same_graph(points) -> None:
    built = []
    for _ in range(2):
        graph = HNSW(points, m=M, ef_construction=40, m_l=1.0, seed=5)
        graph.build()
        built.append(graph.layers)
    assert built[0] == built[1]


def test_recall_is_one_with_large_ef(index, points) -> None:
    """При ef=n широкий поиск на нулевом слое вырождается в полный перебор."""
    queries = data.uniform(10, seed=1)
    truth = brute.knn_batch(points, queries, K)
    for query, exact in zip(queries, truth, strict=True):
        found = [i for _, i in index.search(query, K, ef=N)]
        assert found == exact.tolist()


def test_recall_is_high_at_modest_ef(index, points) -> None:
    """На равномерных двумерных данных граф почти не ошибается."""
    queries = data.uniform(50, seed=2)
    truth = brute.knn_batch(points, queries, K)
    hits = sum(
        len({i for _, i in index.search(query, K, ef=4 * K)} & set(exact.tolist()))
        for query, exact in zip(queries, truth, strict=True)
    )
    assert hits / (len(queries) * K) > 0.95


def test_search_does_not_mutate_the_index(index) -> None:
    before = index.entry_point, [dict(layer) for layer in index.layers]
    for query in data.uniform(20, seed=3):
        index.search(query, K, ef=20)
    assert before == (index.entry_point, [dict(layer) for layer in index.layers])


def test_trace_has_one_entry_per_layer(index) -> None:
    trace: list[list[int]] = []
    found = index.search(np.array([0.5, 0.5]), K, ef=20, trace=trace)

    assert len(trace) == index.max_level + 1
    assert found == index.search(np.array([0.5, 0.5]), K, ef=20)
    assert all(len(hop) >= 1 for hop in trace[:-1]), "спуск должен идти по слоям"


def test_graph_property_exposes_layer_zero(index) -> None:
    """На это свойство опирается bench, чтобы работать с NSW и HNSW одинаково."""
    assert index.graph is index.layers[0]


def test_build_twice_raises(points) -> None:
    graph = HNSW(points[:50], m=M, ef_construction=40, seed=1)
    graph.build()
    with pytest.raises(RuntimeError):
        graph.build()


def test_search_before_build_raises(points) -> None:
    with pytest.raises(RuntimeError):
        HNSW(points, m=M, ef_construction=40).search(points[0], K, ef=20)


def test_ef_below_k_raises(index) -> None:
    with pytest.raises(ValueError):
        index.search(np.array([0.5, 0.5]), K, ef=K - 1)


def test_single_point_graph(points) -> None:
    graph = HNSW(points[:1], m=M, ef_construction=40, seed=1)
    graph.build()
    assert graph.entry_point == 0
    assert graph.search(points[0], 1, ef=1) == [(pytest.approx(0.0), 0)]
