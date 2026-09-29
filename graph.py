"""Layer walk and neighbour selection, shared by NSW and HNSW.

Algorithm numbers follow Malkov & Yashunin, "Efficient and robust approximate
nearest neighbor search using Hierarchical Navigable Small World graphs".
"""

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
    """Alg. 2. Greedy walk over one layer, keeping the ef nearest found.

    :param q: query vector
    :param ep: entry points; there may be several, since line 17 of alg. 1
               passes the whole of W down to the next layer
    :param ef: size of the dynamic candidate list
    :param layer: the layer to walk
    :param points: points (n, d)
    :param metric: distance metric
    :param trace: if given, receives vertices in the order they are expanded —
                  the walk itself, not the visited set. For plotting only
    :return: at most ef (distance, index) pairs, nearest first
    """

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
    """Alg. 3. The m candidates nearest to the base vertex.

    layer, points and metric go unused; they are in the signature so that
    alg. 3 and alg. 4 stay interchangeable behind one parameter.

    :param q: base vertex
    :param candidates: (distance to q, index) pairs
    :param m: how many neighbours to return
    :return: indices of the chosen vertices
    """
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
    """Alg. 4. Neighbours chosen for diversity of direction.

    A candidate e is kept only when it is nearer to the base vertex than to
    any already chosen r — line 11 compares d(e, q) against d(e, r), not
    against d(r, q). An edge to a candidate that some chosen neighbour covers
    better is redundant: the walk will reach it through that neighbour.

    :param q: base vertex
    :param candidates: (distance to q, index) pairs
    :param m: how many neighbours to return
    :param layer: only needed for extend_candidates
    :param points: points (n, d)
    :param metric: distance metric
    :param extend_candidates: widen the pool with the candidates' own
                              neighbours (lines 3-7); worth it only on
                              heavily clustered data, and only when the base
                              vertex is not yet in the layer — otherwise it
                              pulls itself in and selects itself
    :param keep_pruned: top the result back up to m from the rejects
                        (lines 15-17)
    :return: indices of the chosen vertices
    """
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
