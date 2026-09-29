"""Генераторы точек для экспериментов.

Размерность вынесена параметром с умолчанием 2: основные прогоны идут на
плоскости, но влияние размерности на recall — отдельный эксперимент, и
ради него незачем заводить вторые версии функций.
"""

import numpy as np
import numpy.typing as npt


def uniform(n: int, seed: int, d: int = 2) -> npt.NDArray:
    """Равномерные точки в единичном кубе.

    :param n: число точек
    :param seed: зерно генератора
    :param d: размерность
    :return: массив (n, d)
    """
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(low=0.0, high=1.0, size=(n, d))


def clusters(
    n: int, n_clusters: int, sigma: float, seed: int, d: int = 2
) -> npt.NDArray:
    """Гауссовы сгустки вокруг случайно разбросанных центров.

    Точки перемешаны по построению: принадлежность кластеру разыгрывается
    независимо для каждой точки, поэтому порядок в массиве не группирует их
    по сгусткам. Для инкрементального построения графа это важно — иначе
    первые вставки пришлись бы на один кластер, и точка входа навсегда
    осталась бы внутри него.

    :param n: число точек
    :param n_clusters: число сгустков
    :param sigma: разброс внутри сгустка; центры лежат в единичном кубе,
                  так что sigma порядка 0.01 даёт хорошо разделённые
                  кластеры, 0.1 — почти перемешанные
    :param seed: зерно генератора
    :param d: размерность
    :return: массив (n, d)
    """
    rng = np.random.default_rng(seed=seed)
    centers: npt.NDArray = rng.uniform(low=0.0, high=1.0, size=(n_clusters, d))
    labels: npt.NDArray = rng.integers(0, n_clusters, size=n)
    return centers[labels] + rng.normal(loc=0.0, scale=sigma, size=(n, d))
