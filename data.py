import numpy as np
import numpy.typing as npt

def uniform(n: int, seed: int) -> npt.NDArray:
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(low=0.0, high=0.0, size=(n, 2))

def clusters(n: int, n_clusters: int, sigma: float, seed: int) -> npt.NDArray:
    