"""Отрисовка слоёв графа и траектории поиска на плоскости.

Картинки строятся только для двумерных данных: координаты точек — это и
есть их положение на холсте, никакого проецирования не делается. Оси
убраны намеренно, единицы измерения здесь ничего не значат, смысл несут
взаимное расположение точек и рисунок рёбер.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots

from graph import Layer

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
CONTEXT = "#d8d7d2"

LAYER_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
"""Порядковая шкала одного тона: светлое — нулевой слой, тёмное — верхний."""

PATH = "#eb6834"
QUERY = "#1baf7a"

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
"""Первые три слота палитры — различимы при дальтонизме по всем парам."""

REACHED, CUT_OFF = CATEGORICAL[0], CATEGORICAL[1]

MARKERS = ("o", "s", "^")
"""Форма маркера как вторичное кодирование серии, помимо цвета."""


def _layer_colour(lc: int, n_layers: int) -> str:
    """Цвет слоя по его высоте: чем выше, тем темнее.

    :param lc: номер слоя
    :param n_layers: всего слоёв
    :return: шестнадцатеричный цвет из LAYER_RAMP
    """
    if n_layers <= 1:
        return LAYER_RAMP[2]
    position: float = lc / (n_layers - 1)
    return LAYER_RAMP[round(position * (len(LAYER_RAMP) - 1))]


def _segments(points: npt.NDArray, layer: Layer) -> list[tuple[npt.NDArray, npt.NDArray]]:
    """Рёбра слоя как отрезки, каждое неупорядоченное ребро один раз.

    :param points: массив точек (n, 2)
    :param layer: слой графа
    :return: список пар концов
    """
    seen: set[tuple[int, int]] = set()
    segments: list[tuple[npt.NDArray, npt.NDArray]] = []
    for a, neighbours in layer.items():
        for b in neighbours:
            key: tuple[int, int] = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            segments.append((points[a], points[b]))
    return segments


def _bare(ax: Axes) -> None:
    """Убрать оси, рамку и засечки, оставив только содержимое.

    :param ax: оси
    """
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(SURFACE)


def plot_layer(
    points: npt.NDArray,
    layer: Layer,
    ax: Axes,
    *,
    colour: str = LAYER_RAMP[2],
    title: str | None = None,
) -> Axes:
    """Нарисовать один слой: рёбра, его вершины и остальные точки фоном.

    :param points: массив точек (n, 2)
    :param layer: слой графа
    :param ax: оси
    :param colour: цвет рёбер и вершин слоя
    :param title: заголовок; None — собрать из числа вершин и рёбер
    :return: те же оси
    """
    _bare(ax)
    ax.scatter(points[:, 0], points[:, 1], s=3, c=CONTEXT, linewidths=0, zorder=1)

    segments = _segments(points, layer)
    ax.add_collection(
        LineCollection(segments, colors=colour, linewidths=0.6, alpha=0.55, zorder=2)
    )

    members: npt.NDArray = np.fromiter(layer, dtype=np.intp, count=len(layer))
    ax.scatter(
        points[members, 0], points[members, 1], s=14, c=colour, linewidths=0, zorder=3
    )

    ax.set_title(
        title if title is not None else f"вершин {len(layer)} · рёбер {len(segments)}",
        color=INK_MUTED,
        fontsize=9,
        pad=6,
    )
    return ax


def plot_reachability(
    points: npt.NDArray,
    layer: Layer,
    entry_point: int,
    ax: Axes,
    *,
    title: str = "",
) -> Axes:
    """Нулевой слой с разделением на достижимое и отрезанное.

    Обход направленный: усечение связей соседа выбрасывает из его списка
    только что добавленную вершину, обратная ссылка остаётся, и часть
    графа оказывается недостижимой при формально ненулевых степенях.

    :param points: массив точек (n, 2)
    :param layer: слой графа
    :param entry_point: вершина, с которой начинается обход
    :param ax: оси
    :param title: заголовок; к нему дописывается доля достижимого
    :return: те же оси
    """
    seen: set[int] = {entry_point}
    stack: list[int] = [entry_point]
    while stack:
        for neighbour in layer[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)

    _bare(ax)
    ax.add_collection(
        LineCollection(
            _segments(points, layer),
            colors=CONTEXT,
            linewidths=0.5,
            alpha=0.7,
            zorder=1,
        )
    )
    for members, colour, size in (
        (sorted(set(layer) - seen), CUT_OFF, 11),
        (sorted(seen), REACHED, 11),
    ):
        if not members:
            continue
        idx: npt.NDArray = np.asarray(members, dtype=np.intp)
        ax.scatter(
            points[idx, 0], points[idx, 1], s=size, c=colour, linewidths=0, zorder=2
        )

    ax.set_title(
        f"{title}\nдостижимо {len(seen)} из {len(layer)}",
        color=INK_MUTED,
        fontsize=9,
        pad=6,
    )
    return ax


def plot_tradeoff(
    series: dict[str, Sequence[tuple[float, float]]],
    ax: Axes,
    *,
    title: str = "",
    xlabel: str = "вычислений расстояния на запрос",
    ylabel: str = "recall@10",
    slots: dict[str, int] | None = None,
) -> Axes:
    """Кривые «точность против стоимости» — по одной на конфигурацию.

    :param series: подпись -> последовательность пар (стоимость, recall)
    :param ax: оси
    :param title: заголовок панели
    :param xlabel: подпись оси абсцисс
    :param ylabel: подпись оси ординат
    :param slots: подпись -> номер слота палитры. Нужен, когда одна и та же
                  конфигурация встречается на нескольких панелях: цвет должен
                  следовать за ней, а не за порядком внутри панели
    :return: те же оси
    """
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=CONTEXT, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CONTEXT)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)

    for position, (label, pairs) in enumerate(series.items()):
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        slot: int = position if slots is None else slots[label]
        colour: str = CATEGORICAL[slot % len(CATEGORICAL)]
        # Форма маркера дублирует цвет: подписи у концов линий на этих
        # данных налезают друг на друга, а различать серии надо не только
        # по цвету.
        ax.plot(
            xs,
            ys,
            marker=MARKERS[slot % len(MARKERS)],
            color=colour,
            linewidth=2.0,
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            zorder=3,
            label=label,
        )

    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="best")
    return ax


def _grid(n: int, width: float, extra: float, ncols: int) -> tuple[Figure, list[Axes]]:
    """Сетка панелей под n слоёв, лишние клетки убираются.

    Ряд из семи панелей даёт фигуру в два десятка дюймов шириной, поэтому
    панели заворачиваются в несколько рядов.

    :param n: сколько панелей нужно
    :param width: сторона панели в дюймах
    :param extra: добавка к высоте фигуры под заголовок и легенду
    :param ncols: максимум панелей в ряду
    :return: (фигура, плоский список осей длины n)
    """
    cols: int = min(n, ncols)
    rows: int = -(-n // cols)
    fig, axes = subplots(
        rows, cols, figsize=(width * cols, width * rows + extra), facecolor=SURFACE
    )
    flat: list[Axes] = list(np.atleast_1d(axes).ravel())
    for ax in flat[n:]:
        ax.remove()
    return fig, flat[:n]


def plot_layers(
    points: npt.NDArray,
    layers: Sequence[Layer],
    *,
    width: float = 3.1,
    ncols: int = 4,
) -> Figure:
    """Все слои от верхнего к нулевому — от разреженной магистрали к плотному низу.

    :param points: массив точек (n, 2)
    :param layers: слои графа, layers[0] нулевой
    :param width: сторона одной панели в дюймах
    :param ncols: максимум панелей в ряду
    :return: готовая фигура
    """
    n: int = len(layers)
    fig, axes = _grid(n, width, 0.7, ncols)

    for panel, lc in enumerate(range(n - 1, -1, -1)):
        plot_layer(
            points,
            layers[lc],
            axes[panel],
            colour=_layer_colour(lc, n),
            title=f"слой {lc} · вершин {len(layers[lc])}",
        )

    fig.suptitle(
        "Слои HNSW: от разреженной магистрали наверху к полному набору точек внизу",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_search(
    points: npt.NDArray,
    layers: Sequence[Layer],
    q: npt.NDArray,
    trace: Sequence[Sequence[int]],
    *,
    found: Sequence[int] = (),
    width: float = 3.1,
    ncols: int = 4,
) -> Figure:
    """Траектория поиска поверх слоёв, по панели на слой.

    Оранжевая ломаная соединяет вершины в порядке разворачивания, а не по
    рёбрам графа: поиск достаёт кандидатов из очереди и может перескочить.

    :param points: массив точек (n, 2)
    :param layers: слои графа, layers[0] нулевой
    :param q: вектор запроса (2,)
    :param trace: траектории по слоям сверху вниз — то, что вернул search
    :param found: индексы найденных соседей, подсвечиваются на нижней панели
    :param width: сторона одной панели в дюймах
    :param ncols: максимум панелей в ряду
    :return: готовая фигура
    """
    n: int = len(layers)
    fig, axes = _grid(n, width, 1.2, ncols)

    for panel, lc in enumerate(range(n - 1, -1, -1)):
        ax: Axes = axes[panel]
        plot_layer(
            points,
            layers[lc],
            ax,
            colour=_layer_colour(lc, n),
            title=f"слой {lc} · развёрнуто {len(trace[panel])}",
        )

        walk: Sequence[int] = trace[panel]
        if walk:
            xy: npt.NDArray = points[np.asarray(walk, dtype=np.intp)]
            ax.plot(
                xy[:, 0], xy[:, 1], "-", color=PATH, linewidth=1.4, alpha=0.9, zorder=4
            )
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=26,
                c=PATH,
                edgecolors=SURFACE,
                linewidths=0.8,
                zorder=5,
            )

        if lc == 0 and len(found):
            hit: npt.NDArray = points[np.asarray(found, dtype=np.intp)]
            ax.scatter(
                hit[:, 0],
                hit[:, 1],
                s=52,
                facecolors="none",
                edgecolors=INK,
                linewidths=1.1,
                zorder=6,
            )

        ax.scatter(
            [q[0]], [q[1]], s=130, c=QUERY, marker="X", edgecolors=SURFACE,
            linewidths=1.2, zorder=7,
        )
        if lc == n - 1:
            ax.annotate(
                "запрос",
                xy=(q[0], q[1]),
                xytext=(7, 7),
                textcoords="offset points",
                color=INK,
                fontsize=8,
            )

    fig.legend(
        handles=[
            Line2D([], [], color=PATH, marker="o", linewidth=1.4, markersize=5,
                   label="порядок разворачивания"),
            Line2D([], [], color=QUERY, marker="X", linewidth=0, markersize=9,
                   label="запрос"),
            Line2D([], [], color="none", marker="o", markersize=8,
                   markerfacecolor="none", markeredgecolor=INK,
                   label="найденные соседи"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=INK_MUTED,
    )
    fig.suptitle(
        "Спуск по слоям: ef=1 на верхних, широкий поиск на нулевом",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig
