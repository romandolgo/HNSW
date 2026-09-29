"""Exhaustive search, the oracle recall is measured against."""

import numpy as np
import numpy.typing as npt


def knn(
    points: npt.NDArray,
    q: npt.NDArray,
    k: int,
) -> npt.NDArray[np.intp]:
    """Exact k nearest neighbours.

    The metric is fixed on purpose: ground truth is always euclidean, and a
    vectorised implementation has no business going through the distance
    counter.

    :param points: points (n, d)
    :param q: query vector (d,)
    :param k: how many neighbours
    :return: indices of the k nearest, nearest first
    """
    distances_sq: npt.NDArray[np.floating] = np.sum((points - q) ** 2, axis=-1)
    nearest_idx: npt.NDArray[np.intp] = np.argpartition(distances_sq, kth=k - 1)[:k]

    return nearest_idx[np.argsort(distances_sq[nearest_idx])]


def knn_batch(
    points: npt.NDArray,
    queries: npt.NDArray,
    k: int,
) -> npt.NDArray[np.intp]:
    """Exact k nearest neighbours for a batch of queries.

    :param points: points (n, d)
    :param queries: query vectors (n_queries, d)
    :param k: how many neighbours
    :return: indices (n_queries, k), each row nearest first
    """
    return np.array([knn(points, q, k) for q in queries])
