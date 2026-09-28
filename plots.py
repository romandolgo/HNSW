"""Отрисовка графа и результатов замеров.

Требует matplotlib: uv add matplotlib
"""

from collections.abc import Sequence

import numpy.typing as npt
from matplotlib.axes import Axes

from bench import Result
from graph import Layer


def plot_layer(
    points: npt.NDArray,
    layer: Layer,
    ax: Axes | None = None,
    *,
    title: str | None = None,
) -> Axes:
    """Нарисовать один слой: точки и рёбра между ними.

    :param points: массив точек (n, 2)
    :param layer: слой графа
    :param ax: оси; None — создать новые
    :param title: заголовок
    """
    raise NotImplementedError


def plot_layers(
    points: npt.NDArray,
    layers: Sequence[Layer],
    *,
    reverse: bool = True,
) -> Sequence[Axes]:
    """Нарисовать все слои рядом — верхний как «магистраль».

    :param points: массив точек (n, 2)
    :param layers: слои графа, layers[0] — нулевой
    :param reverse: рисовать сверху вниз, от разреженного слоя к плотному
    """
    raise NotImplementedError


def plot_path(
    points: npt.NDArray,
    layer: Layer,
    trace: Sequence[int],
    q: npt.NDArray,
    ax: Axes | None = None,
) -> Axes:
    """Подсветить путь поиска поверх слоя.

    :param points: массив точек (n, 2)
    :param layer: слой, по которому шёл поиск
    :param trace: вершины в порядке обхода; источник надо будет вернуть —
                  сейчас search_layer порядок обхода наружу не отдаёт
    :param q: вектор запроса
    :param ax: оси; None — создать новые
    """
    raise NotImplementedError


def plot_tradeoff(
    results: Sequence[Result],
    ax: Axes | None = None,
    *,
    x: str = "dist_per_query",
    label: str | None = None,
) -> Axes:
    """Кривая recall против стоимости запроса.

    :param results: результаты одного прогона sweep
    :param ax: оси; None — создать новые
    :param x: поле Result по оси абсцисс
    :param label: подпись серии
    """
    raise NotImplementedError
