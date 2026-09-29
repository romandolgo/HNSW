"""Quality and cost of search.

Cost is counted in distance computations rather than seconds: on 2D data
wall time drowns in interpreter overhead, while the count depends only on
the graph and the search parameters.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt

import brute
from graph import Layer
from metrics import Metric, euclidean
from nsw import NSW


class CountingMetric:
    """Metric wrapper that counts distance computations.

    Transparent to the graph: build and search take a metric parameter and
    never learn whether it counts anything.

    :param metric: the metric to wrap
    """

    def __init__(self, metric: Metric = euclidean) -> None:
        self._metric = metric
        self._count = 0

    def __call__(self, a: npt.NDArray, b: npt.NDArray) -> float:
        self._count += 1
        return self._metric(a, b)

    @property
    def count(self) -> int:
        """Distances computed since the last reset."""
        return self._count

    def reset(self) -> int:
        """Zero the counter.

        :return: the value before zeroing, so a reading and the start of the
                 next phase fit on one line
        """
        count, self._count = self._count, 0
        return count


@dataclass
class Result:
    """One run at one set of parameters.

    :param params: m, ef_construction, selector, ef
    :param recall: recall@k averaged over queries
    :param dist_per_query: distances computed per query
    :param build_dist: distances computed while building
    :param degree_mean: mean vertex degree
    :param degree_max: largest vertex degree
    :param reachable: vertices reachable from the entry point
    :param n: vertices in the graph
    """

    params: dict[str, Any] = field(default_factory=dict)
    recall: float = 0.0
    dist_per_query: float = 0.0
    build_dist: int = 0
    degree_mean: float = 0.0
    degree_max: int = 0
    reachable: int = 0
    n: int = 0


def recall_at_k(found: Sequence[Sequence[int]], truth: npt.NDArray) -> float:
    """Share of the true k nearest that were found, averaged over queries.

    :param found: one list of indices per query
    :param truth: exact indices (n_queries, k) from brute.knn_batch
    :return: recall@k in [0, 1]
    """
    k: int = truth.shape[1]
    hits: int = sum(
        len(set(f) & set(t.tolist())) for f, t in zip(found, truth, strict=True)
    )
    return hits / (len(truth) * k)


def reachable_from(graph: Layer, entry_point: int) -> int:
    """Vertices a directed walk from the entry point can reach.

    Directed on purpose: shrinking a neighbour's list drops the back-edge, so
    parts of the graph go unreachable while every degree still looks healthy.

    :param graph: a layer
    :param entry_point: where the walk starts
    :return: reachable vertices, the entry point included
    """
    seen: set[int] = {entry_point}
    stack: list[int] = [entry_point]
    while stack:
        for neighbour in graph[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen)


def degrees(graph: Layer) -> tuple[float, int]:
    """Mean and largest vertex degree.

    :param graph: a layer
    :return: (mean, max)
    """
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
    """Run the queries against a built index and collect the metrics.

    :param index: a built graph
    :param queries: query vectors (n_queries, d)
    :param truth: exact answers (n_queries, k)
    :param ef: candidate list size during search
    :param counter: the counter the index was built with; zeroed first, to
                    keep search cost apart from build cost
    :param build_dist: the counter reading taken after the build
    :param params: extra fields for Result.params
    :return: the filled Result
    """
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
    """Sweep a grid of build and search parameters.

    Each config builds once, then every ef runs against that graph. Ground
    truth is computed once for the whole sweep.

    :param points: points (n, d)
    :param queries: query vectors (n_queries, d)
    :param k: how many neighbours
    :param configs: build parameters — m, ef_construction, m_max, selector
    :param efs: search-time candidate list sizes
    :param index_cls: NSW or HNSW
    :return: one Result per (config, ef) pair
    """
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
    """Lay results out as a text table.

    :param results: what sweep returned
    :param params: which Result.params fields become columns
    :return: the table, ready to print
    """
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
