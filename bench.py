from collections.abc import Iterable

import numpy.typing as npt

import bruteforce
from hnsw import HNSW
from metrics import Metric, euclidean
from nsw import NSW


class CountingMetric:
    def __init__(self, metric: Metric = euclidean) -> None:
        self.metric = metric
        self.count = 0

    def __call__(self, a: npt.NDArray, b: npt.NDArray) -> float:
        self.count += 1
        return self.metric(a, b)


def recall_at_k(found: list[list[int]], truth: npt.NDArray) -> float:
    hits = sum(len(set(f) & set(t)) for f, t in zip(found, truth, strict=True))
    return hits / truth.size


def sweep(
    points: npt.NDArray,
    queries: npt.NDArray,
    *,
    k: int,
    configs: Iterable[dict],
    efs: Iterable[int],
    index_cls: type[NSW] | type[HNSW] = NSW,
) -> list[dict]:
    truth = bruteforce.knn_batch(points, queries, k)
    efs = [ef for ef in efs if ef >= k]
    results = []

    for config in configs:
        counter = CountingMetric()
        index = index_cls(points, metric=counter, **config)
        index.build()

        for ef in efs:
            counter.count = 0
            found = [[i for _, i in index.search(q, k, ef=ef)] for q in queries]
            results.append({
                "m": config["m"],
                "ef": ef,
                "recall": recall_at_k(found, truth),
                "dist_per_query": counter.count / len(queries),
            })
    return results


def format_table(results: list[dict]) -> str:
    rows = [f"{'m':>3} {'ef':>4} {'recall':>7} {'dist/q':>8}"]
    for r in results:
        rows.append(
            f"{r['m']:3} {r['ef']:4} {r['recall']:7.4f} {r['dist_per_query']:8.0f}"
        )
    return "\n".join(rows)
