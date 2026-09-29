"""Метрика расстояния."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type Metric = Callable[[npt.NDArray, npt.NDArray], float]


def euclidean(a: npt.NDArray, b: npt.NDArray) -> float:
    """Евклидово расстояние между двумя точками.

    :param a: вектор (d,)
    :param b: вектор (d,)
    :return: расстояние
    """
    return np.linalg.norm(a - b, ord=2, axis=-1)


def manhattan(a: npt.NDArray, b: npt.NDArray) -> float:
    return np.linalg.norm(a - b, ord=1, axis=-1)
