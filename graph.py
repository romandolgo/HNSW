"""Общее ядро графа: обход слоя (Alg. 2) и отбор соседей (Alg. 3, Alg. 4).

Номера алгоритмов — по статье Malkov & Yashunin, "Efficient and robust
approximate nearest neighbor search using Hierarchical Navigable Small
World graphs".

Этот модуль не знает ни про уровни, ни про порядок вставки: и NSW, и HNSW
используют один и тот же search_layer, различаясь только тем, откуда
берутся точки входа.
"""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt

from metrics import Metric

type Layer = dict[int, list[int]]
"""Слой графа: индекс вершины -> список индексов её соседей.

Словарь, а не список списков: на верхних слоях лежит лишь малая часть
точек, и наличие ключа означает членство вершины в слое.
"""

type Candidates = Sequence[tuple[float, int]]
"""Кандидаты как пары (расстояние до запроса, индекс вершины)."""

type Selector = Callable[..., list[int]]
"""Стратегия отбора соседей: select_simple либо select_heuristic."""


def search_layer(
    q: npt.NDArray,
    ep: Iterable[int],
    ef: int,
    layer: Layer,
    points: npt.NDArray,
    metric: Metric,
) -> list[tuple[float, int]]:
    """Alg. 2: SEARCH-LAYER(q, ep, ef, lc).

    Жадный обход одного слоя с динамическим списком найденных ближайших.
    Множество посещённых вершин — локальное для одного вызова.

    :param q: вектор запроса
    :param ep: точки входа; их может быть несколько — строка 17 Alg. 1
               передаёт на нижний слой весь список W, а не одну вершину
    :param ef: размер динамического списка найденных ближайших
    :param layer: слой, по которому идёт обход
    :param points: массив точек (n, d)
    :param metric: метрика
    :return: не более ef пар (расстояние, индекс), по возрастанию расстояния
    """
    ep_arr: npt.NDArray = np.asarray(ep)
    visited: npt.NDArray = np.ndarray.copy(ep_arr)
    candidates: npt.NDArray = np.ndarray.copy(ep_arr)
    nearest_points = np.ndarray.copy(ep_arr)

    while candidates:
        candidate: npt.NDArray = np.argmin(metric())


def select_simple(
    q: npt.NDArray,
    cand: Candidates,
    m: int,
    layer: Layer,
    points: npt.NDArray,
    metric: Metric,
) -> list[int]:
    """Alg. 3: SELECT-NEIGHBORS-SIMPLE(q, C, M).

    Просто m ближайших к q кандидатов. Параметры layer, points и metric не
    используются — они в сигнатуре, чтобы Alg. 3 и Alg. 4 были
    взаимозаменяемы и bench мог переключать их одним параметром.

    :param q: вектор базовой вершины
    :param cand: кандидаты (расстояние до q, индекс)
    :param m: сколько соседей вернуть
    :return: индексы отобранных вершин
    """
    raise NotImplementedError


def select_heuristic(
    q: npt.NDArray,
    cand: Candidates,
    m: int,
    layer: Layer,
    points: npt.NDArray,
    metric: Metric,
    *,
    extend_candidates: bool = False,
    keep_pruned: bool = False,
) -> list[int]:
    """Alg. 4: SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, ...).

    Отбирает соседей «в разные стороны»: кандидат e принимается, только
    если он ближе к q, чем к любой из уже отобранных вершин r. В строке 11
    сравниваются d(e, q) и d(e, r) — не d(e, q) и d(r, q).

    :param q: вектор базовой вершины
    :param cand: кандидаты (расстояние до q, индекс)
    :param m: сколько соседей вернуть
    :param layer: слой — нужен при extend_candidates
    :param points: массив точек (n, d)
    :param metric: метрика
    :param extend_candidates: расширить множество кандидатов их соседями
                              (строки 3-7); полезно лишь на сильно
                              кластеризованных данных
    :param keep_pruned: добить результат отброшенными кандидатами до m
                        штук (строки 15-17)
    :return: индексы отобранных вершин
    """
    raise NotImplementedError
