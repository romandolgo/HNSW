import numpy as np
import numpy.typing as npt


def knn(
    points: npt.NDArray,
    q: npt.NDArray,
    k: int,
) -> npt.NDArray[np.intp]:
    distances_sq: npt.NDArray[np.floating] = np.sum((points - q) ** 2, axis=-1)
    nearest_idx: npt.NDArray[np.intp] = np.argpartition(distances_sq, kth=k - 1)[:k]

    return nearest_idx[np.argsort(distances_sq[nearest_idx])]


def knn_batch(
    points: npt.NDArray,
    queries: npt.NDArray,
    k: int,
) -> npt.NDArray[np.intp]:
    return np.array([knn(points, q, k) for q in queries])
