import numpy.typing as npt

from graph import Layer, Selector, search_layer, select_simple
from metrics import Metric, euclidean


class NSW:
    points: npt.NDArray
    graph: Layer
    entry_point: int | None

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
            raise RuntimeError(
                "Graph is already built: build() is meant to be called once. "
                "Create a new NSW to rebuild."
            )
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
            )

    def search(
        self,
        q: npt.NDArray,
        k: int,
        *,
        ef: int,
        trace: list[int] | None = None,
    ) -> list[tuple[float, int]]:
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

        return search_layer(
            q=q,
            ep=[self.entry_point],
            ef=ef,
            layer=self.graph,
            points=self.points,
            metric=self.metric,
            trace=trace,
        )[:k]
