"""Distance metrics."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type Metric = Callable[[npt.NDArray, npt.NDArray], float]


def euclidean(a: npt.NDArray, b: npt.NDArray) -> float:
    """L2 distance.

    :param a: vector (d,)
    :param b: vector (d,)
    :return: distance
    """
    return np.linalg.norm(a - b, ord=2, axis=-1)


def manhattan(a: npt.NDArray, b: npt.NDArray) -> float:
    """L1 distance.

    :param a: vector (d,)
    :param b: vector (d,)
    :return: distance
    """
    return np.linalg.norm(a - b, ord=1, axis=-1)
