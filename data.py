from pathlib import Path
from urllib.request import urlopen

import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def pendigits() -> tuple[npt.NDArray, npt.NDArray]:
    folder = Path("datasets/pendigits")
    folder.mkdir(parents=True, exist_ok=True)
    samples = []
    for split, size in {"tra": 502098, "tes": 234366}.items():
        path = folder / f"pendigits.{split}"
        if not path.exists():
            url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/{path.name}"
            temporary = path.with_suffix(".tmp")
            with urlopen(url, timeout=30) as response, temporary.open("wb") as output:
                total = int(response.headers.get("Content-Length", size))
                with tqdm(
                    total=total, desc=path.name, unit="B", unit_scale=True,
                    unit_divisor=1024, dynamic_ncols=True,
                ) as progress:
                    while chunk := response.read(65536):
                        output.write(chunk)
                        progress.update(len(chunk))
            temporary.replace(path)
        samples.append(np.loadtxt(path, delimiter=",", dtype=float)[:, :-1] / 100)
    return samples[0], samples[1]
