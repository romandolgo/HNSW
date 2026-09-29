"""Пересборка всех картинок отчёта одной командой.

    uv run python figures.py

Кладёт png в figures/. Данные и параметры зафиксированы зёрнами, поэтому
повторный запуск даёт те же картинки.
"""

from pathlib import Path

import numpy as np
from matplotlib.pyplot import close, subplots

import bench
import data
import plots
from graph import select_heuristic, select_simple
from hnsw import HNSW
from nsw import NSW

OUT = Path("figures")
K = 10

UNIFORM_N = 700
CLUSTERED_N = 2000
CLUSTERS = 6
SIGMA = 0.03
QUERIES_N = 120
EFS = (10, 15, 20, 30, 50, 80)

# Один и тот же набор точек на картинку со связностью и на кривые: иначе
# читателю пришлось бы держать в голове два разных датасета. Кластеров
# немного и sigma крупная, чтобы сгустки читались как пятна, а не как точки;
# при этом простой отбор теряет половину графа, и это видно.

SLOTS = {"NSW, Alg. 3": 0, "NSW, Alg. 4": 1, "HNSW, Alg. 4": 2}
"""Слот палитры закреплён за конфигурацией, а не за порядком на панели."""


def _clustered(*, seed: int) -> np.ndarray:
    """Кластеризованный набор, общий для картинки связности и для кривых."""
    return data.clusters(CLUSTERED_N, n_clusters=CLUSTERS, sigma=SIGMA, seed=seed)


def hierarchy() -> None:
    """Слои HNSW и спуск по ним на равномерных данных.

    m_l завышен против умолчания: при рекомендованном 1/ln(8) на семистах
    точках слоёв выходит три-четыре, и лестница получается короткой.
    """
    points = data.uniform(UNIFORM_N, seed=0)
    index = HNSW(points, m=8, ef_construction=40, m_l=0.9, seed=5)
    index.build()

    figure = plots.plot_layers(points, index.layers)
    figure.savefig(OUT / "layers.png", dpi=130)
    close(figure)

    query = np.array([0.62, 0.38])
    trace: list[list[int]] = []
    found = index.search(query, 8, ef=24, trace=trace)

    figure = plots.plot_search(
        points, index.layers, query, trace, found=[i for _, i in found]
    )
    figure.savefig(OUT / "search.png", dpi=130)
    close(figure)


def reachability() -> None:
    """Связность на кластерах: Alg. 3 рвёт граф, Alg. 4 держит."""
    points = _clustered(seed=1)

    figure, axes = subplots(1, 2, figsize=(9.2, 5.0), facecolor=plots.SURFACE)
    for ax, (label, selector) in zip(
        axes,
        (("Alg. 3 — ближайшие", select_simple), ("Alg. 4 — эвристика", select_heuristic)),
        strict=True,
    ):
        graph = NSW(points, m=8, ef_construction=40, m_max=16, selector=selector)
        graph.build()
        plots.plot_reachability(points, graph.graph, graph.entry_point, ax, title=label)

    figure.suptitle(
        "Достижимое из точки входа на кластеризованных данных",
        color=plots.INK,
        fontsize=11,
    )
    figure.legend(
        handles=[
            plots.Line2D([], [], color=plots.REACHED, marker="o", linewidth=0,
                         markersize=7, label="достижимо"),
            plots.Line2D([], [], color=plots.CUT_OFF, marker="o", linewidth=0,
                         markersize=7, label="отрезано"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=plots.INK_MUTED,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(OUT / "reachability.png", dpi=130)
    close(figure)


def _curve(results: list[bench.Result]) -> list[tuple[float, float]]:
    """Result-ы одной конфигурации в пары (стоимость, recall)."""
    return [(r.dist_per_query, r.recall) for r in results]


def tradeoff() -> None:
    """Точность против стоимости: сперва селекторы, затем иерархия."""
    points = _clustered(seed=1)
    queries = data.clusters(QUERIES_N, n_clusters=CLUSTERS, sigma=SIGMA, seed=2)

    common = dict(m=8, ef_construction=40)
    runs = {
        "NSW, Alg. 3": bench.sweep(
            points, queries, k=K, index_cls=NSW, efs=EFS,
            configs=[dict(**common, m_max=16, selector=select_simple)],
        ),
        "NSW, Alg. 4": bench.sweep(
            points, queries, k=K, index_cls=NSW, efs=EFS,
            configs=[dict(**common, m_max=16, selector=select_heuristic)],
        ),
        "HNSW, Alg. 4": bench.sweep(
            points, queries, k=K, index_cls=HNSW, efs=EFS,
            configs=[dict(**common, seed=1, selector=select_heuristic)],
        ),
    }

    figure, axes = subplots(1, 2, figsize=(10.4, 4.4), facecolor=plots.SURFACE)
    plots.plot_tradeoff(
        {name: _curve(runs[name]) for name in ("NSW, Alg. 3", "NSW, Alg. 4")},
        axes[0],
        title="Отбор соседей",
        slots=SLOTS,
    )
    plots.plot_tradeoff(
        {name: _curve(runs[name]) for name in ("NSW, Alg. 4", "HNSW, Alg. 4")},
        axes[1],
        title="Иерархия",
        slots=SLOTS,
    )

    figure.suptitle(
        f"Кластеризованные 2D, {CLUSTERED_N} точек в {CLUSTERS} сгустках, "
        f"m=8, ef_construction=40",
        color=plots.INK,
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(OUT / "tradeoff.png", dpi=130)
    close(figure)

    for name, results in runs.items():
        print(f"\n{name}")
        print(bench.format_table(results, params=["ef"]))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    hierarchy()
    reachability()
    tradeoff()
    print(f"\nкартинки в {OUT.resolve()}")


if __name__ == "__main__":
    main()
