"""Одноуровневый NSW-граф.

По разделу 4.1 статьи это HNSW при mL = 0: ровно один слой. Если вдобавок
не ограничивать степень вершины (m_max=None), получается классический NSW
с полилогарифмической сложностью поиска; при m_max = m — направленный
k-NN-граф со степенной сложностью.
"""

import numpy.typing as npt

from graph import Layer, Selector, search_layer, select_simple
from metrics import Metric, euclidean


class NSW:
    """Граф ближайших соседей с инкрементальным построением.

    :param points: массив точек (n, d)
    :param m: число связей, создаваемых при вставке вершины
    :param ef_construction: размер динамического списка при построении
    :param m_max: верхняя граница степени вершины; None — не усекать
    :param metric: метрика
    :param selector: отбор соседей — select_simple (Alg. 3) либо
                     select_heuristic (Alg. 4)
    """

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
        raise NotImplementedError

    def build(self) -> None:
        """Вставить все точки по одной, в порядке их следования в points."""
        raise NotImplementedError

    def _insert(self, i: int) -> None:
        """Вставка одной вершины.

        Поиск с ef_construction от точки входа, отбор m соседей, создание
        двунаправленных связей, при необходимости — усечение связей каждого
        соседа до m_max (строки 12-16 Alg. 1). Усекаются связи соседа, не
        самой вставляемой вершины: у неё их и так не больше m.

        Точка входа фиксирована — это первая вставленная вершина. Ранние
        версии NSW вместо этого делали несколько поисков от случайных
        вершин; здесь качество поиска регулируется параметром ef.
        """
        raise NotImplementedError

    def search(
        self,
        q: npt.NDArray,
        k: int,
        *,
        ef: int,
    ) -> list[tuple[float, int]]:
        """Поиск k приближённо ближайших соседей.

        :param q: вектор запроса (d,)
        :param k: число соседей
        :param ef: размер динамического списка, ef >= k
        :return: k пар (расстояние, индекс), по возрастанию расстояния
        """
        raise NotImplementedError
