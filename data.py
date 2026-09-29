"""Point generators for the experiments."""

import numpy as np
import numpy.typing as npt


def uniform(n: int, seed: int, d: int = 2) -> npt.NDArray:
    """Points drawn uniformly from the unit cube.

    :param n: how many points
    :param seed: generator seed
    :param d: dimensionality
    :return: points (n, d)
    """
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(low=0.0, high=1.0, size=(n, d))


def clusters(
    n: int, n_clusters: int, sigma: float, seed: int, d: int = 2
) -> npt.NDArray:
    """Gaussian blobs around randomly placed centres.

    Points come out shuffled: cluster membership is drawn per point, so array
    order does not group them. Insertion order matters — were the first
    insertions all from one blob, the entry point would stay inside it.

    :param n: how many points
    :param n_clusters: how many blobs
    :param sigma: spread within a blob; centres live in the unit cube, so
                  0.01 gives well separated blobs and 0.1 nearly merged ones
    :param seed: generator seed
    :param d: dimensionality
    :return: points (n, d)
    """
    rng = np.random.default_rng(seed=seed)
    centers: npt.NDArray = rng.uniform(low=0.0, high=1.0, size=(n_clusters, d))
    labels: npt.NDArray = rng.integers(0, n_clusters, size=n)
    return centers[labels] + rng.normal(loc=0.0, scale=sigma, size=(n, d))
