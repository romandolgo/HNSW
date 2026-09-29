from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt

import brute
from graph import Layer
from metrics import Metric, euclidean
from nsw import NSW


class CountingMetric:
    def __init__(self, metric: Metric = euclidean) -> None:
        self._metric = metric
        self._count = 0

    def __call__(self, a: npt.NDArray, b: npt.NDArray) -> float:
        self._count += 1
        return self._metric(a, b)

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> int:
        count, self._count = self._count, 0
        return count


@dataclass
class Result:
    params: dict[str, Any] = field(default_factory=dict)
    recall: float = 0.0
    dist_per_query: float = 0.0
    build_dist: int = 0
    degree_mean: float = 0.0
    degree_max: int = 0
    reachable: int = 0
    n: int = 0


def recall_at_k(found: Sequence[Sequence[int]], truth: npt.NDArray) -> float:
    k: int = truth.shape[1]
    hits: int = sum(
        len(set(f) & set(t.tolist())) for f, t in zip(found, truth, strict=True)
    )
    return hits / (len(truth) * k)


def reachable_from(graph: Layer, entry_point: int) -> int:
    seen: set[int] = {entry_point}
    stack: list[int] = [entry_point]
    while stack:
        for neighbour in graph[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen)


def degrees(graph: Layer) -> tuple[float, int]:
    sizes: list[int] = [len(neighbours) for neighbours in graph.values()]
    return sum(sizes) / len(sizes), max(sizes)


def evaluate(
    index: NSW,
    queries: npt.NDArray,
    truth: npt.NDArray,
    *,
    ef: int,
    counter: CountingMetric,
    build_dist: int,
    params: dict[str, Any] | None = None,
) -> Result:
    k: int = truth.shape[1]
    counter.reset()
    found: list[list[int]] = [
        [e for _, e in index.search(q, k, ef=ef)] for q in queries
    ]
    search_dist: int = counter.reset()

    degree_mean, degree_max = degrees(index.graph)
    return Result(
        params={**(params or {}), "ef": ef},
        recall=recall_at_k(found, truth),
        dist_per_query=search_dist / len(queries),
        build_dist=build_dist,
        degree_mean=degree_mean,
        degree_max=degree_max,
        # pyrefly: ignore [bad-argument-type]
        reachable=reachable_from(index.graph, index.entry_point),
        n=len(index.graph),
    )


def sweep(
    points: npt.NDArray,
    queries: npt.NDArray,
    *,
    k: int,
    configs: Iterable[dict[str, Any]],
    efs: Iterable[int],
    index_cls: type = NSW,
) -> list[Result]:
    truth: npt.NDArray = brute.knn_batch(points, queries, k)
    results: list[Result] = []

    for config in configs:
        counter = CountingMetric()
        index = index_cls(points, metric=counter, **config)
        index.build()
        build_dist: int = counter.reset()

        params: dict[str, Any] = {
            key: (value.__name__ if callable(value) else value)
            for key, value in config.items()
        }
        for ef in efs:
            if ef < k:
                continue
            results.append(
                evaluate(
                    index,
                    queries,
                    truth,
                    ef=ef,
                    counter=counter,
                    build_dist=build_dist,
                    params=params,
                )
            )
    return results


def format_table(results: Sequence[Result], *, params: Sequence[str]) -> str:
    head: list[str] = [*params, "recall", "dist/q", "build", "deg", "max", "reach"]
    rows: list[list[str]] = [head]
    for r in results:
        rows.append(
            [
                *(str(r.params.get(p, "")) for p in params),
                f"{r.recall:.3f}",
                f"{r.dist_per_query:.0f}",
                f"{r.build_dist}",
                f"{r.degree_mean:.1f}",
                f"{r.degree_max}",
                f"{r.reachable}/{r.n}",
            ]
        )

    widths: list[int] = [max(len(row[i]) for row in rows) for i in range(len(head))]
    return "\n".join(
        "  ".join(cell.rjust(w) for cell, w in zip(row, widths, strict=True))
        for row in rows
    )
