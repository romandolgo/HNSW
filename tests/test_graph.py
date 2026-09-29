"""The shared core: metric, layer walk, neighbour selection."""

import numpy as np
import pytest

from graph import search_layer, select_heuristic, select_simple
from metrics import euclidean


@pytest.fixture
def complete_layer() -> tuple[np.ndarray, dict[int, list[int]]]:
    """Twenty random points, every pair connected.

    A greedy walk cannot get stuck on a complete graph, so any disagreement
    with exhaustive search points at the algorithm rather than the topology.
    """
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
    """At ef >= n the walk over a complete layer matches exhaustive search."""
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    found = [i for _, i in search_layer(q, [0], 20, layer, points, euclidean)]
    exact = np.argsort(np.linalg.norm(points - q, axis=1)).tolist()
    assert found == exact


def test_search_layer_respects_ef(complete_layer) -> None:
    """No longer than ef, and sorted by distance."""
    points, layer = complete_layer
    q = np.array([0.5, 0.5])

    for ef in (1, 3, 7, 20):
        found = search_layer(q, [0], ef, layer, points, euclidean)
        assert len(found) == ef
        distances = [d for d, _ in found]
        assert distances == sorted(distances)


def test_search_layer_computes_each_distance_once(complete_layer) -> None:
    """Metric calls equal visited vertices.

    A vertex's distance is measured when first met and then travels in the
    tuple. A higher count means either a lost visited check or a distance
    being recomputed.
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
    """Line 17 of alg. 1 hands down the whole of W, so several are allowed."""
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
    """Candidates with their distance to the base vertex, nearest first.

    The geometry the heuristic is meant to act on: base at the origin, NEAR
    the closest candidate, SHADOWED behind it on the same line, DIVERSE off
    to the side.

        DIVERSE (0, 1.2)
             |
          BASE ---- NEAR (1, 0) ---- SHADOWED (1.5, 0)
    """
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
    """A candidate an already chosen neighbour covers better is dropped.

    SHADOWED sits nearer to NEAR (0.5) than to the base (1.5), so a direct
    edge to it is redundant — the walk gets there through NEAR. DIVERSE is
    the other way round, nearer the base (1.2) than NEAR (1.56), and opens a
    direction nothing covers yet.
    """
    got = select_heuristic(
        HEURISTIC_POINTS[BASE], heuristic_candidates, 3, {}, HEURISTIC_POINTS, euclidean
    )
    assert got == [NEAR, DIVERSE]
    assert SHADOWED not in got


def test_keep_pruned_fills_the_result_back_to_m(heuristic_candidates) -> None:
    """Rejects are taken back nearest first."""
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
    """Both take the same arguments by name; bench relies on it."""
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
