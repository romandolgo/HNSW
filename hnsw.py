"""Multi-layer HNSW: insertion (alg. 1) and search (alg. 5).

Both descend the upper layers greedily with ef=1, then widen on the target
layer. The walking is done by search_layer; this module only manages levels.
"""

import numpy as np
import numpy.typing as npt

from graph import Layer, Selector, search_layer, select_heuristic
from metrics import Metric, euclidean


class HNSW:
    """Hierarchical graph built by inserting points one at a time.

    points is held by reference and assumed immutable for the object's
    lifetime.

    :param points: points (n, d)
    :param m: connections created per insertion
    :param ef_construction: candidate list size during construction
    :param m_max: degree cap above layer zero; None -> m
    :param m_max0: degree cap on layer zero; None -> 2 * m, per section 4.1
    :param m_l: level sampling scale; None -> 1 / ln(m), per section 4.1.
                Each layer then holds about 1/m of the one below
    :param metric: distance metric
    :param selector: select_heuristic (alg. 4) or select_simple (alg. 3)
    :param seed: seed for level sampling
    """

    points: npt.NDArray
    layers: list[Layer]
    entry_point: int | None

    def __init__(
        self,
        points: npt.NDArray,
        *,
        m: int,
        ef_construction: int,
        m_max: int | None = None,
        m_max0: int | None = None,
        m_l: float | None = None,
        metric: Metric = euclidean,
        selector: Selector = select_heuristic,
        seed: int | None = None,
    ) -> None:
        self.points: npt.NDArray = points
        self.m: int = m
        self.ef_construction: int = ef_construction
        self.m_max: int = m if m_max is None else m_max
        self.m_max0: int = 2 * m if m_max0 is None else m_max0
        self.m_l: float = float(1 / np.log(m)) if m_l is None else m_l
        self.metric: Metric = metric
        self.selector: Selector = selector
        self.seed: int | None = seed
        self.layers: list[Layer] = []
        self.entry_point: int | None = None
        self.rng = np.random.default_rng(self.seed)

    @property
    def max_level(self) -> int:
        """Index of the top layer; -1 while the graph is empty."""
        if self.entry_point is None:
            return -1
        return len(self.layers) - 1

    @property
    def graph(self) -> Layer:
        """Layer zero, where the final search happens.

        Lets bench read degrees and reachability the same way for NSW and
        HNSW. Layer zero is the meaningful one to inspect: the upper layers
        only ferry the search to the right neighbourhood.
        """
        return self.layers[0]

    def build(self) -> None:
        """Insert every point, in array order."""
        if self.layers:
            raise RuntimeError(
                "Graph is already built: build() is meant to be called once. "
                "Create a new HNSW to rebuild."
            )
        for i in range(self.points.shape[0]):
            self._insert(i)

    def _random_level(self) -> int:
        """Line 4 of alg. 1: l = floor(-ln(U(0, 1]) * m_l).

        The vertex then lives on layers 0..l, not on layer l alone. The draw
        ignores coordinates entirely — height is a lottery, not a property of
        where the point sits.

        :return: top layer for the new vertex
        """
        return int(
            np.floor(-np.log(1 - self.rng.uniform(low=0.0, high=1.0)) * self.m_l)
        )

    def _connect(self, i: int, neighbours: list[int], lc: int) -> None:
        """Lines 11-16 of alg. 1: link both ways, then shrink what overflows.

        It is the neighbour's list that gets shrunk, never the new vertex's —
        that one holds len(neighbours) <= m to begin with. Shrinking may drop
        the vertex just added, leaving a one-way edge; that is intended.

        :param i: the vertex being inserted
        :param neighbours: neighbours chosen by the selector
        :param lc: layer index
        """
        layer: Layer = self.layers[lc]
        m_max: int = self.m_max0 if lc == 0 else self.m_max

        layer[i] = list(neighbours)
        for n in neighbours:
            layer[n].append(i)

        for n in neighbours:
            connected: list[int] = layer[n]
            if len(connected) <= m_max:
                continue
            pairs: list[tuple[float, int]] = [
                (self.metric(self.points[n], self.points[c]), c) for c in connected
            ]
            layer[n] = self.selector(
                q=self.points[n],
                candidates=pairs,
                m=m_max,
                layer=layer,
                points=self.points,
                metric=self.metric,
            )

    def _insert(self, i: int) -> None:
        """Alg. 1. Draw a level, descend to it, then connect down to zero.

        Lines 5-7 walk from the top down to l+1 with ef=1, keeping one entry
        point. Lines 8-16 then run from min(L, l) to zero with
        ef_construction, connecting on each layer. Line 17 hands the whole of
        W to the next layer, not just the nearest vertex.

        The graph's entry point moves only when the drawn level beats the
        current top, and only at the very end — moving it earlier would start
        the descent from a vertex that is in no layer yet.
        """
        elem_layer: int = self._random_level()
        if self.entry_point is None:
            self.layers = [{i: []} for _ in range(elem_layer + 1)]
            self.entry_point = i
            return

        L: int = self.max_level
        ep: list[int] = [self.entry_point]

        for lc in range(L, elem_layer, -1):
            nearest_points: list[tuple[float, int]] = search_layer(
                q=self.points[i],
                ep=ep,
                ef=1,
                layer=self.layers[lc],
                points=self.points,
                metric=self.metric,
            )
            ep = [nearest_points[0][1]]

        for lc in range(min(L, elem_layer), -1, -1):
            nearest_points: list[tuple[float, int]] = search_layer(
                q=self.points[i],
                ep=ep,
                ef=self.ef_construction,
                layer=self.layers[lc],
                points=self.points,
                metric=self.metric,
            )

            neighbours: list[int] = self.selector(
                q=self.points[i],
                candidates=nearest_points,
                m=self.m,
                layer=self.layers[lc],
                points=self.points,
                metric=self.metric,
            )

            self._connect(i, neighbours, lc)
            ep = [idx for _, idx in nearest_points]

        if elem_layer > L:
            self.layers.extend({i: []} for _ in range(elem_layer - L))
            self.entry_point = i

    def search(
        self,
        q: npt.NDArray,
        k: int,
        *,
        ef: int,
        trace: list[list[int]] | None = None,
    ) -> list[tuple[float, int]]:
        """Alg. 5. Descend with ef=1, then widen on layer zero.

        :param q: query vector (d,)
        :param k: how many neighbours
        :param ef: candidate list size on layer zero; below k the result would
                   be silently truncated, so it is rejected
        :param trace: if given, receives one walk per layer, top down
        :return: k (distance, index) pairs, nearest first
        """
        if self.entry_point is None:
            raise RuntimeError(
                "Graph is not built: you cannot use search on unbuilt graph. "
                "To build the graph first use build() method."
            )

        if ef < k:
            raise ValueError(
                f"ef must be at least k: got ef={ef}, k={k}. "
                "A smaller ef truncates the result to ef neighbours."
            )
        ep: list[int] = [self.entry_point]
        for lc in range(self.max_level, 0, -1):
            hop: list[int] | None = [] if trace is not None else None
            next_start: int = search_layer(
                q=q,
                ep=ep,
                ef=1,
                layer=self.layers[lc],
                points=self.points,
                metric=self.metric,
                trace=hop,
            )[0][1]
            if trace is not None:
                trace.append(hop)
            ep: list[int] = [next_start]

        bottom: list[int] | None = [] if trace is not None else None
        found = search_layer(
            q=q,
            ep=ep,
            ef=ef,
            layer=self.layers[0],
            points=self.points,
            metric=self.metric,
            trace=bottom,
        )
        if trace is not None:
            trace.append(bottom)
        return found[:k]
