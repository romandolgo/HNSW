"""Многоуровневый HNSW: вставка (Alg. 1) и поиск (Alg. 5).

Оба алгоритма устроены одинаково: спуск по верхним слоям жадным поиском с
ef=1, затем расширенный поиск на целевом слое. Вся работа делается общим
search_layer из graph.py — этот модуль отвечает только за уровни.
"""

import numpy as np
import numpy.typing as npt

from graph import Layer, Selector, search_layer, select_heuristic
from metrics import Metric, euclidean


class HNSW:
    """Иерархический граф с инкрементальным построением.

    :param points: массив точек (n, d)
    :param m: число связей, создаваемых при вставке вершины
    :param ef_construction: размер динамического списка при построении
    :param m_max: граница степени на слоях выше нулевого; None -> m
    :param m_max0: граница степени на нулевом слое; None -> 2 * m
                   (рекомендация раздела 4.1)
    :param m_l: нормировочный множитель при сэмплировании уровня;
                None -> 1 / ln(m) (рекомендация раздела 4.1)
    :param metric: метрика
    :param selector: отбор соседей — select_heuristic (Alg. 4) либо
                     select_simple (Alg. 3)
    :param seed: зерно генератора для сэмплирования уровней
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
        """Номер верхнего непустого слоя; -1 у пустого графа."""
        if self.entry_point is None:
            return -1
        return len(self.layers) - 1

    @property
    def graph(self) -> Layer:
        """Нулевой слой — тот, по которому идёт финальный поиск.

        Нужен, чтобы bench.py снимал степени и достижимость одинаково с
        NSW и HNSW. Содержательно диагностика связности осмысленна именно
        для нулевого слоя: верхние только подвозят к нужной области.
        """
        return self.layers[0]

    def build(self) -> None:
        """Вставить все точки по одной, в порядке их следования в points."""
        if self.layers:
            raise RuntimeError(
                "Graph is already built: build() is meant to be called once. "
                "Create a new HNSW to rebuild."
            )
        for i in range(self.points.shape[0]):
            self._insert(i)

    def _random_level(self) -> int:
        """Сэмплирование уровня новой вершины, строка 4 Alg. 1.

        l = floor(-ln(U(0, 1)) * m_l). Вершина живёт на слоях 0..l
        включительно, а не только на слое l.

        :return: номер верхнего слоя для новой вершины
        """
        return int(
            np.floor(-np.log(1 - self.rng.uniform(low=0.0, high=1.0)) * self.m_l)
        )

    def _connect(self, i: int, neighbours: list[int], lc: int) -> None:
        """Строки 11-16 Alg. 1: двунаправленные связи и усечение.

        Логика та же, что в NSW, и отличается двумя вещами: работает с
        конкретным слоем и берёт разный лимит степени — на нулевом слое
        m_max0, выше m_max.

        Усекаются связи соседа, а не самой вставляемой вершины: у неё их
        ровно len(neighbours) <= m, переполниться неоткуда.

        :param i: вставляемая вершина
        :param neighbours: отобранные селектором соседи
        :param lc: номер слоя
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
        """Alg. 1: INSERT(hnsw, q, M, Mmax, efConstruction, mL).

        Две фазы. Сверху вниз до слоя l+1 — жадный спуск с ef=1, от него
        остаётся одна точка входа (строки 5-7). Дальше от min(L, l) до нуля
        — поиск с ef_construction, отбор соседей и двунаправленные связи
        (строки 8-16). Точка входа графа меняется только при l > L.

        Строка 17: на следующий слой передаётся весь список W, а не одна
        ближайшая вершина.
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
        """Alg. 5: K-NN-SEARCH(hnsw, q, K, ef).

        :param q: вектор запроса (d,)
        :param k: число соседей
        :param ef: размер динамического списка на нулевом слое, ef >= k;
                   на верхних слоях всегда ef=1
        :param trace: список для записи траекторий по слоям, сверху вниз —
                      по одному вложенному списку на слой, для визуализации
        :return: k пар (расстояние, индекс), по возрастанию расстояния
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
