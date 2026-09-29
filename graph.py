import heapq
from collections.abc import Callable, Iterable

import numpy.typing as npt

from metrics import Metric

type Layer = dict[int, list[int]]
"""Vertex index -> its neighbours. A key present means the vertex is in the layer."""

type Point = tuple[float, int]
"""Distance to some base vertex, paired with the vertex index."""

type Selector = Callable[..., list[int]]
"""Neighbour selection strategy: select_simple or select_heuristic."""


def search_layer(
    q: npt.NDArray,
    ep: Iterable[int],
    ef: int,
    layer: Layer,
    points: npt.NDArray,
    metric: Metric,
    trace: list[int] | None = None,
) -> list[Point]:
    ids: list[int] = list(ep)
    visited: set[int] = set(ids)

    distances: list[float] = [metric(points[e], q) for e in ids]
    candidates: list[Point] = [(d, e) for d, e in zip(distances, ids)]
    nearest_points: list[Point] = [(-d, e) for d, e in zip(distances, ids)]
    heapq.heapify(candidates)
    heapq.heapify(nearest_points)

    while candidates:
        candidate: Point = heapq.heappop(candidates)
        worst_nearest: Point = nearest_points[0]

        if candidate[0] > -worst_nearest[0]:
            break

        if trace is not None:
            trace.append(candidate[1])

        for e in layer[candidate[1]]:
            if e in visited:
                continue
            visited.add(e)
            d_e: float = metric(points[e], q)
            worst_nearest: Point = nearest_points[0]
            if d_e < -worst_nearest[0] or len(nearest_points) < ef:
                heapq.heappush(candidates, (d_e, e))
                heapq.heappush(nearest_points, (-d_e, e))

                if len(nearest_points) > ef:
                    heapq.heappop(nearest_points)

    return sorted([(-dist, idx) for dist, idx in nearest_points], key=lambda x: x[0])


def select_simple(
    q: npt.NDArray,
    candidates: list[Point],
    m: int,
    layer: Layer,
    points: npt.NDArray,
    metric: Metric,
) -> list[int]:
    return [e for distance, e in heapq.nsmallest(m, candidates)]


def select_heuristic(
    q: npt.NDArray,
    candidates: list[Point],
    m: int,
    layer: Layer,
    points: npt.NDArray,
    metric: Metric,
    *,
    extend_candidates: bool = False,
    keep_pruned: bool = False,
) -> list[int]:
    pool: list[Point] = list(candidates)

    if extend_candidates:
        seen: set[int] = {e for _, e in candidates}
        for _, c in candidates:
            for adj in layer[c]:
                if adj not in seen:
                    seen.add(adj)
                    pool.append((metric(q, points[adj]), adj))

    nearest_points: list[Point] = sorted(pool)
    heuristic_selected: list[int] = []
    discarded_candidates: list[int] = []

    for distance, e in nearest_points:
        if len(heuristic_selected) >= m:
            break
        if all(distance < metric(points[e], points[r]) for r in heuristic_selected):
            heuristic_selected.append(e)
        else:
            discarded_candidates.append(e)

    if keep_pruned:
        for e in discarded_candidates:
            if len(heuristic_selected) >= m:
                break
            heuristic_selected.append(e)

    return heuristic_selected
