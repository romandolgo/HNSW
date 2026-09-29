"""Замеры качества и стоимости поиска.

Качество — recall@k относительно точного ответа из brute.py. Стоимость —
число вычислений расстояния на запрос: на двумерных данных разница во
времени тонет в накладных расходах интерпретатора, а счётчик зависит
только от структуры графа и параметров поиска.

Отдельно снимаются две характеристики самого графа: распределение степеней
и число вершин, достижимых из точки входа. Второе оказалось важнее
первого — при простом отборе соседей на кластеризованных данных граф
распадается, и recall упирается не в ширину поиска, а в недостижимость.
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
    """Обёртка над метрикой, считающая вычисления расстояния.

    Прозрачна для графа: и построение, и поиск принимают метрику
    параметром и не знают, считает она что-нибудь или нет.

    :param metric: оборачиваемая метрика
    """

    def __init__(self, metric: Metric = euclidean) -> None:
        self._metric = metric
        self._count = 0

    def __call__(self, a: npt.NDArray, b: npt.NDArray) -> float:
        self._count += 1
        return self._metric(a, b)

    @property
    def count(self) -> int:
        """Число вычислений расстояния с момента последнего reset()."""
        return self._count

    def reset(self) -> int:
        """Обнулить счётчик.

        :return: значение до обнуления — чтобы снять показание и начать
                 новую фазу замера одной строкой
        """
        count, self._count = self._count, 0
        return count


@dataclass
class Result:
    """Итог одного прогона на фиксированном наборе параметров.

    :param params: параметры прогона — m, ef_construction, selector, ef
    :param recall: recall@k, усреднённый по запросам
    :param dist_per_query: среднее число вычислений расстояния на запрос
    :param build_dist: число вычислений расстояния при построении графа
    :param degree_mean: средняя степень вершины
    :param degree_max: максимальная степень вершины
    :param reachable: вершин, достижимых из точки входа
    :param n: всего вершин в графе
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
    """Доля истинных k ближайших среди найденных, усреднённая по запросам.

    :param found: по списку индексов на каждый запрос
    :param truth: (n_queries, k) точные индексы из brute.knn_batch
    :return: recall@k в [0, 1]
    """
    k: int = truth.shape[1]
    hits: int = sum(
        len(set(f) & set(t.tolist())) for f, t in zip(found, truth, strict=True)
    )
    return hits / (len(truth) * k)


def reachable_from(graph: Layer, entry_point: int) -> int:
    """Сколько вершин достижимо из точки входа по направлению рёбер.

    Обход именно направленный: усечение связей соседа выбрасывает из его
    списка только что добавленную вершину, а обратная ссылка остаётся, и
    часть графа становится недостижимой при формально ненулевых степенях.

    :param graph: слой графа
    :param entry_point: вершина, с которой начинается обход
    :return: число достижимых вершин, включая саму точку входа
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
    """Средняя и максимальная степень вершины.

    :param graph: слой графа
    :return: (средняя, максимальная)
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
    """Прогнать запросы по готовому индексу и собрать метрики.

    :param index: построенный граф
    :param queries: массив запросов (n_queries, d)
    :param truth: точные ответы (n_queries, k)
    :param ef: размер динамического списка при поиске
    :param counter: счётчик, которым построен индекс; обнуляется перед
                    прогоном, чтобы отделить поиск от построения
    :param build_dist: показание счётчика, снятое после построения
    :param params: дополнительные поля для Result.params
    :return: заполненный Result
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
    """Перебор сетки параметров построения и поиска.

    Для каждого набора из configs граф строится один раз, после чего по
    нему прогоняются все значения efs. Эталон считается один раз на все
    прогоны.

    :param points: массив точек (n, d)
    :param queries: массив запросов (n_queries, d)
    :param k: число соседей
    :param configs: наборы параметров построения — m, ef_construction,
                    m_max, selector
    :param efs: значения ef для поиска
    :param index_cls: класс индекса, NSW или HNSW
    :return: по одному Result на пару (набор из configs, значение ef)
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
    """Собрать результаты в текстовую таблицу.

    :param results: что вернул sweep
    :param params: какие поля Result.params показывать колонками
    :return: готовая к печати таблица
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
