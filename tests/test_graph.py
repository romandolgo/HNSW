import numpy as np
import pytest

from graph import search_layer, select_heuristic, select_simple
from metrics import euclidean


@pytest.fixture
def complete_layer() -> tuple[np.ndarray, dict[int, list[int]]]:
    points = np.random.default_rng(0).uniform(size=(20, 2))
    layer = {i: [j for j in range(20) if j != i] for i in range(20)}
    return points, layer


def test_euclidean_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.uniform(size=(50, 3)), rng.uniform(size=(50, 3))
    for x, y in zip(a, b, strict=True):
        assert euclidean(x, y) == pytest.approx(np.linalg.norm(x - y))


def test_euclidean_broadcasts_over_a_batch() -> None:
    rng = np.random.default_rng(1)
    q, batch = rng.uniform(size=3), rng.uniform(size=(7, 3))
    got = euclidean(batch, q)
    assert got.shape == (7,)
    assert got == pytest.approx(np.linalg.norm(batch - q, axis=-1))


def test_search_layer_is_exact_with_large_ef(complete_layer) -> None:
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    found = [i for _, i in search_layer(q, [0], 20, layer, points, euclidean)]
    exact = np.argsort(np.linalg.norm(points - q, axis=1)).tolist()
    assert found == exact


def test_search_layer_respects_ef(complete_layer) -> None:
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    for ef in (1, 3, 7, 20):
        found = search_layer(q, [0], ef, layer, points, euclidean)
        assert len(found) == ef
        distances = [d for d, _ in found]
        assert distances == sorted(distances)


def test_search_layer_computes_each_distance_once(complete_layer) -> None:
    points, layer = complete_layer
    calls = 0

    def counting(a, b):
        nonlocal calls
        calls += 1
        return euclidean(a, b)

    search_layer(np.array([0.5, 0.5]), [0], 20, layer, points, counting)
    assert calls == len(points)


def test_search_layer_accepts_several_entry_points(complete_layer) -> None:
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    from_one = search_layer(q, [0], 5, layer, points, euclidean)
    from_many = search_layer(q, [0, 7, 13], 5, layer, points, euclidean)
    assert from_one == from_many


def test_search_layer_does_not_mutate_the_layer(complete_layer) -> None:
    points, layer = complete_layer
    before = {v: list(ns) for v, ns in layer.items()}
    search_layer(np.array([0.5, 0.5]), [0], 10, layer, points, euclidean)
    assert layer == before


def test_trace_records_the_walk_without_changing_the_result(complete_layer) -> None:
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    trace: list[int] = []
    with_trace = search_layer(q, [0], 5, layer, points, euclidean, trace=trace)
    without = search_layer(q, [0], 5, layer, points, euclidean)

    assert with_trace == without
    assert trace
    assert len(trace) == len(set(trace)), "a vertex was expanded twice"


HEURISTIC_POINTS = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0], [0.0, 1.2]])
BASE, NEAR, SHADOWED, DIVERSE = 0, 1, 2, 3


@pytest.fixture
def heuristic_candidates() -> list[tuple[float, int]]:
    base = HEURISTIC_POINTS[BASE]
    return sorted(
        (float(euclidean(HEURISTIC_POINTS[i], base)), i)
        for i in (NEAR, SHADOWED, DIVERSE)
    )


def test_select_simple_returns_m_nearest(heuristic_candidates) -> None:
    got = select_simple(
        HEURISTIC_POINTS[BASE], heuristic_candidates, 2, {}, HEURISTIC_POINTS, euclidean
    )
    assert got == [NEAR, DIVERSE]


def test_select_simple_returns_everything_when_m_exceeds_candidates(
    heuristic_candidates,
) -> None:
    got = select_simple(
        HEURISTIC_POINTS[BASE],
        heuristic_candidates,
        99,
        {},
        HEURISTIC_POINTS,
        euclidean,
    )
    assert sorted(got) == [NEAR, SHADOWED, DIVERSE]


def test_select_heuristic_drops_the_shadowed_candidate(heuristic_candidates) -> None:
    got = select_heuristic(
        HEURISTIC_POINTS[BASE], heuristic_candidates, 3, {}, HEURISTIC_POINTS, euclidean
    )
    assert got == [NEAR, DIVERSE]
    assert SHADOWED not in got


def test_keep_pruned_fills_the_result_back_to_m(heuristic_candidates) -> None:
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
