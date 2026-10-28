from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots

import bench
import data
from graph import select_heuristic, select_simple
from hnsw import HNSW
from nsw import NSW

M_VALUES = (4, 8, 16)
EFS = (10, 20, 40, 80, 160)
K = 10
QUERIES = 300
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
OUT = Path("figures/pendigits.html")


def main() -> None:
    points, queries = data.pendigits()
    queries = np.random.default_rng(0).permutation(queries)[:QUERIES]
    selectors = (("Simple", select_simple), ("Heuristic", select_heuristic))
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes="all",
        shared_yaxes=True,
        subplot_titles=("Simple: nearest neighbours", "Heuristic: diverse neighbours"),
        horizontal_spacing=0.06,
    )

    for col, (selector_name, selector) in enumerate(selectors, start=1):
        for index_cls, dash, symbol in (
            (NSW, "dash", "circle"),
            (HNSW, "solid", "square"),
        ):
            print(
                f"\n{index_cls.__name__}, {selector_name}: building and searching...",
                flush=True,
            )
            configs = []
            for m in M_VALUES:
                config = {
                    "m": m,
                    "ef_construction": 100,
                    "selector": selector,
                }
                if index_cls is NSW:
                    config["m_max"] = 2 * m
                else:
                    config["seed"] = 0
                configs.append(config)
            results = bench.sweep(
                points, queries, k=K, configs=configs, efs=EFS, index_cls=index_cls
            )
            print(bench.format_table(results))

            for i, m in enumerate(M_VALUES):
                rows = [r for r in results if r["m"] == m]
                name = f"{index_cls.__name__}, m={m}"
                figure.add_scatter(
                    x=[r["dist_per_query"] for r in rows],
                    y=[r["recall"] for r in rows],
                    mode="lines+markers",
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                    line={"color": COLORS[i], "dash": dash},
                    marker={"symbol": symbol, "size": 8},
                    customdata=[r["ef"] for r in rows],
                    hovertemplate=(
                        selector_name + "<br>ef=%{customdata}<br>Recall=%{y:.4f}"
                        "<br>Distances/query=%{x:.1f}<extra>%{fullData.name}</extra>"
                    ),
                    row=1,
                    col=col,
                )

    figure.update_layout(
        template="plotly_white",
        title=f"Pendigits: {len(points)} points in {points.shape[1]}D, {len(queries)} queries",
        legend={"orientation": "h", "y": -0.2, "groupclick": "togglegroup"},
        height=650,
    )
    figure.update_xaxes(title_text="Distance evaluations / query (lower is better)")
    figure.update_yaxes(range=[0, 1.01])
    figure.update_yaxes(title_text=f"Recall@{K}", row=1, col=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        OUT, include_plotlyjs=True, config={"responsive": True, "displaylogo": False}
    )
    print(f"\nSaved: {OUT.resolve()}")


if __name__ == "__main__":
    main()
