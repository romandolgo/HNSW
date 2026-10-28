import numpy as np
import numpy.typing as npt

from graph import Layer, Selector, search_layer, select_heuristic
from metrics import Metric, euclidean


class HNSW:
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
        if m < 2:
            raise ValueError("m must be >= 2")
        if m_max is not None and m_max < m:
            raise ValueError("m_max must be >= m")
        if m_max0 is not None and m_max0 < m:
            raise ValueError("m_max0 must be >= m")
        if m_l is not None and (not np.isfinite(m_l) or m_l < 0):
            raise ValueError("m_l must be finite and >= 0")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Duplicate points")

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
        if self.entry_point is None:
            return -1
        return len(self.layers) - 1

    @property
    def graph(self) -> Layer:
        return self.layers[0]

    def build(self) -> None:
        if self.layers:
            raise RuntimeError("Graph is already built.")
        for i in range(self.points.shape[0]):
            self._insert(i)

    def _random_level(self) -> int:
        return int(
            np.floor(-np.log(1 - self.rng.uniform(low=0.0, high=1.0)) * self.m_l)
        )

    def _connect(self, i: int, neighbours: list[int], lc: int) -> None:
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
                exclude=n,
            )

    def _insert(self, i: int) -> None:
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
    ) -> list[tuple[float, int]]:
        if self.entry_point is None:
            raise RuntimeError("Build graph first")

        if ef < k:
            raise ValueError("ef must be >= k")
        ep: list[int] = [self.entry_point]
        for lc in range(self.max_level, 0, -1):
            next_start: int = search_layer(
                q=q,
                ep=ep,
                ef=1,
                layer=self.layers[lc],
                points=self.points,
                metric=self.metric,
            )[0][1]
            ep = [next_start]

        return search_layer(
            q=q,
            ep=ep,
            ef=ef,
            layer=self.layers[0],
            points=self.points,
            metric=self.metric,
        )[:k]
