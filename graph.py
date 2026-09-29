"""Общее ядро графа: обход слоя (Alg. 2) и отбор соседей (Alg. 3, Alg. 4).

Номера алгоритмов — по статье Malkov & Yashunin, "Efficient and robust
approximate nearest neighbor search using Hierarchical Navigable Small
World graphs".

Этот модуль не знает ни про уровни, ни про порядок вставки: и NSW, и HNSW
используют один и тот же search_layer, различаясь только тем, откуда
берутся точки входа.
"""

import heapq
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt

from metrics import Metric

type Layer = dict[int, list[int]]
"""Слой графа: индекс вершины -> список индексов её соседей.

Словарь, а не список списков: на верхних слоях лежит лишь малая часть
точек, и наличие ключа означает членство вершины в слое.
"""

type Point = tuple[float, int]

type Selector = Callable[..., list[int]]
"""Стратегия отбора соседей: select_simple либо select_heuristic."""


def search_layer(
    q: npt.NDArray,
    ep: Iterable[int],
    ef: int,
    layer: Layer,
    points: npt.NDArray,
    metric: Metric,
) -> list[Point]:
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
    """Alg. 3: SELECT-NEIGHBORS-SIMPLE(q, C, M).

    Просто m ближайших к q кандидатов. Параметры layer, points и metric не
    используются — они в сигнатуре, чтобы Alg. 3 и Alg. 4 были
    взаимозаменяемы и bench мог переключать их одним параметром.

    :param q: вектор базовой вершины
    :param cand: кандидаты (расстояние до q, индекс)
    :param m: сколько соседей вернуть
    :return: индексы отобранных вершин
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
