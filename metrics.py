"""Метрика и подсчёт числа вычислений расстояния.

Счётчик — основной измерительный инструмент работы. На двумерных данных
небольшого размера разница во времени тонет в накладных расходах
интерпретатора, тогда как число вычислений расстояния зависит только от
структуры графа и параметров поиска.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type Metric = Callable[[npt.NDArray, npt.NDArray], float]


def euclidean(a: npt.NDArray, b: npt.NDArray) -> float:
    """Евклидово расстояние между двумя точками.

    :param a: вектор (d,)
    :param b: вектор (d,)
    :return: расстояние
    """
    return np.linalg.norm(a - b, ord=2, axis=-1)


class CountingMetric:
    """Обёртка над метрикой, считающая число вызовов.

    Прозрачна для всего остального кода: построение и поиск принимают
    метрику параметром и не знают, считает она что-нибудь или нет.

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

        :return: значение счётчика до обнуления — чтобы снять показание и
                 начать новую фазу замера одной строкой
        """
        count, self._count = self._count, 0
        return count
