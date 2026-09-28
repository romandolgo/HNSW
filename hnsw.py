"""Многоуровневый HNSW: вставка (Alg. 1) и поиск (Alg. 5).

Оба алгоритма устроены одинаково: спуск по верхним слоям жадным поиском с
ef=1, затем расширенный поиск на целевом слое. Вся работа делается общим
search_layer из graph.py — этот модуль отвечает только за уровни.
"""

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
        raise NotImplementedError

    @property
    def max_level(self) -> int:
        """Номер верхнего непустого слоя; -1 у пустого графа."""
        raise NotImplementedError

    def build(self) -> None:
        """Вставить все точки по одной, в порядке их следования в points."""
        raise NotImplementedError

    def _random_level(self) -> int:
        """Сэмплирование уровня новой вершины, строка 4 Alg. 1.

        l = floor(-ln(U(0, 1)) * m_l). Вершина живёт на слоях 0..l
        включительно, а не только на слое l.

        :return: номер верхнего слоя для новой вершины
        """
        raise NotImplementedError

    def _insert(self, i: int) -> None:
        """Alg. 1: INSERT(hnsw, q, M, Mmax, efConstruction, mL).

        Две фазы. Сверху вниз до слоя l+1 — жадный спуск с ef=1, от него
        остаётся одна точка входа (строки 5-7). Дальше от min(L, l) до нуля
        — поиск с ef_construction, отбор соседей и двунаправленные связи
        (строки 8-16). Точка входа графа меняется только при l > L.

        Строка 17: на следующий слой передаётся весь список W, а не одна
        ближайшая вершина.
        """
        raise NotImplementedError

    def search(
        self,
        q: npt.NDArray,
        k: int,
        *,
        ef: int,
    ) -> list[tuple[float, int]]:
        """Alg. 5: K-NN-SEARCH(hnsw, q, K, ef).

        :param q: вектор запроса (d,)
        :param k: число соседей
        :param ef: размер динамического списка на нулевом слое, ef >= k;
                   на верхних слоях всегда ef=1
        :return: k пар (расстояние, индекс), по возрастанию расстояния
        """
        raise NotImplementedError
