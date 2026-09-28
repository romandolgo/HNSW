"""Точный поиск полным перебором — эталон для оценки recall."""

import numpy as np
import numpy.typing as npt


def knn(
    points: npt.NDArray,
    q: npt.NDArray,
    k: int,
) -> npt.NDArray[np.intp]:
    """Точный k-NN полным перебором.

    Метрика не параметризуется намеренно: эталон всегда евклидов, и
    выгодна векторизованная реализация на numpy, вызовы которой незачем
    считать счётчиком расстояний.

    :param points: массив точек (n, d)
    :param q: вектор запроса (d,)
    :param k: число соседей
    :return: индексы k ближайших точек, по возрастанию расстояния
    """
    distances_sq: npt.NDArray[np.floating] = np.sum((points - q) ** 2, axis=-1)
    nearest_idx: npt.NDArray[np.intp] = np.argpartition(distances_sq, kth=k - 1)[:k]

    return nearest_idx[np.argsort(distances_sq[nearest_idx])]


def knn_batch(
    points: npt.NDArray,
    queries: npt.NDArray,
    k: int,
) -> npt.NDArray[np.intp]:
    """Точный k-NN для набора запросов — построение ground truth.

    :param points: массив точек (n, d)
    :param queries: массив запросов (n_queries, d)
    :param k: число соседей
    :return: массив индексов (n_queries, k), в каждой строке по возрастанию
    """
    return np.array([knn(points, q, k) for q in queries])
