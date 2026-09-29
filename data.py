import numpy as np
import numpy.typing as npt


def uniform(n: int, seed: int, d: int = 2) -> npt.NDArray:
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(low=0.0, high=1.0, size=(n, d))


def clusters(
    n: int, n_clusters: int, sigma: float, seed: int, d: int = 2
) -> npt.NDArray:
    rng = np.random.default_rng(seed=seed)
    centers: npt.NDArray = rng.uniform(low=0.0, high=1.0, size=(n_clusters, d))
    labels: npt.NDArray = rng.integers(0, n_clusters, size=n)
    return centers[labels] + rng.normal(loc=0.0, scale=sigma, size=(n, d))
