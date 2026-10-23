from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type Metric = Callable[[npt.NDArray, npt.NDArray], float]


def euclidean(a: npt.NDArray, b: npt.NDArray) -> float:
    return float(np.linalg.norm(a - b, ord=2, axis=-1))
