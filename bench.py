"""Замеры: recall и число вычислений расстояния.

Основная метрика качества — recall@k относительно точного ответа из
brute.py, основная метрика стоимости — число вычислений расстояния на
запрос. Время меряется тоже, но на низкой размерности оно шумное.

sweep — самая свободная часть модуля: её форму стоит уточнить после
первых графиков, когда станет понятно, какие срезы реально нужны.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy.typing as npt

from metrics import CountingMetric


@dataclass(frozen=True)
class Result:
    """Итог одного прогона на фиксированном наборе параметров.

    :param params: параметры прогона — m, ef, ef_construction, selector...
    :param recall: recall@k, усреднённый по запросам
    :param dist_per_query: среднее число вычислений расстояния на запрос
    :param build_dist: число вычислений расстояния при построении графа
    :param time_per_query_ms: среднее время на запрос, мс
    """

    params: dict[str, object]
    recall: float
    dist_per_query: float
    build_dist: int
    time_per_query_ms: float


def recall_at_k(found: npt.NDArray, truth: npt.NDArray) -> float:
    """Доля истинных k ближайших среди найденных, усреднённая по запросам.

    :param found: (n_queries, k) индексы, выданные индексом
    :param truth: (n_queries, k) точные индексы из brute.knn_batch
    :return: recall@k в [0, 1]
    """
    raise NotImplementedError


def evaluate(
    index: object,
    queries: npt.NDArray,
    truth: npt.NDArray,
    *,
    k: int,
    ef: int,
    counter: CountingMetric,
    params: dict[str, object] | None = None,
) -> Result:
    """Прогнать запросы по готовому индексу и собрать метрики.

    :param index: построенный NSW или HNSW
    :param queries: массив запросов (n_queries, d)
    :param truth: точные ответы (n_queries, k)
    :param k: число соседей
    :param ef: размер динамического списка при поиске
    :param counter: счётчик, которым построен индекс; сбрасывается перед
                    прогоном запросов, чтобы отделить поиск от построения
    :param params: дополнительные поля для Result.params
    """
    raise NotImplementedError


def sweep(
    factory: Callable[..., object],
    grid: Iterable[dict[str, object]],
    queries: npt.NDArray,
    truth: npt.NDArray,
    *,
    k: int,
    ef_values: Iterable[int],
) -> list[Result]:
    """Перебор сетки параметров построения и поиска.

    Для каждого набора из grid граф строится один раз, после чего по нему
    прогоняются все значения ef_values.

    :param factory: конструктор индекса, принимающий metric и параметры
                    из grid и возвращающий уже построенный граф
    :param grid: наборы параметров построения — m, ef_construction, selector
    :param queries: массив запросов (n_queries, d)
    :param truth: точные ответы (n_queries, k)
    :param k: число соседей
    :param ef_values: значения ef для поиска
    :return: по одному Result на пару (набор из grid, значение ef)
    """
    raise NotImplementedError
