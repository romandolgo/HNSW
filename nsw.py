import numpy as np
import numpy.typing as npt

from graph import Layer, Selector, search_layer, select_simple
from metrics import Metric, euclidean


class NSW:
    def __init__(
        self,
        points: npt.NDArray,
        *,
        m: int,
        ef_construction: int,
        m_max: int | None = None,
        metric: Metric = euclidean,
        selector: Selector = select_simple,
    ) -> None:
        if m < 1:
            raise ValueError("m must be >= 1")
        if m_max is not None and m_max < m:
            raise ValueError("m_max must be >= m")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Duplicate points")
        self.points: npt.NDArray = points
        self.m: int = m
        self.ef_construction: int = ef_construction
        self.m_max: int | None = m_max
        self.metric: Metric = metric
        self.selector: Selector = selector

        self.graph: Layer = {}
        self.entry_point: int | None = None

    def build(self) -> None:
        if self.graph:
            raise RuntimeError("Graph already built")
        for i in range(self.points.shape[0]):
            self._insert(i)

    def _insert(self, i: int) -> None:
        if self.entry_point is None:
            self.entry_point = i
            self.graph[i] = []
            return
        nearest_points: list[tuple[float, int]] = search_layer(
            q=self.points[i],
            ep=[self.entry_point],
            ef=self.ef_construction,
            layer=self.graph,
            points=self.points,
            metric=self.metric,
        )
        neighbours: list[int] = self.selector(
            q=self.points[i],
            candidates=nearest_points,
            m=self.m,
            layer=self.graph,
            points=self.points,
            metric=self.metric,
        )
        self.graph[i] = list(neighbours)
        for n in neighbours:
            self.graph[n].append(i)

        if self.m_max is None:
            return

        for n in neighbours:
            connected: list[int] = self.graph[n]
            if len(connected) <= self.m_max:
                continue
            pairs: list[tuple[float, int]] = [
                (self.metric(self.points[n], self.points[c]), c) for c in connected
            ]
            self.graph[n] = self.selector(
                q=self.points[n],
                candidates=pairs,
                m=self.m_max,
                layer=self.graph,
                points=self.points,
                metric=self.metric,
                exclude=n,
            )

    def search(
        self,
        q: npt.NDArray,
        k: int,
        *,
        ef: int,
    ) -> list[tuple[float, int]]:
        if self.entry_point is None:
            raise RuntimeError("Build graph first")

        if ef < k:
            raise ValueError("ef must be >= k")

        return search_layer(
            q=q,
            ep=[self.entry_point],
            ef=ef,
            layer=self.graph,
            points=self.points,
            metric=self.metric,
        )[:k]
