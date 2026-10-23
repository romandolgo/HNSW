from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import bench
import data
from graph import select_heuristic
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
    figure = go.Figure()

    for index_cls, dash, symbol in ((NSW, "dash", "circle"), (HNSW, "solid", "square")):
        print(f"\n{index_cls.__name__}: building and searching...", flush=True)
        configs = []
        for m in M_VALUES:
            config = {
                "m": m,
                "ef_construction": 100,
                "selector": select_heuristic,
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
            figure.add_scatter(
                x=[r["dist_per_query"] for r in rows],
                y=[r["recall"] for r in rows],
                mode="lines+markers",
                name=f"{index_cls.__name__}, m={m}",
                line={"color": COLORS[i], "dash": dash},
                marker={"symbol": symbol, "size": 8},
                customdata=[r["ef"] for r in rows],
                hovertemplate="ef=%{customdata}<br>Recall=%{y:.4f}"
                "<br>Distances/query=%{x:.1f}<extra>%{fullData.name}</extra>",
            )

    figure.update_layout(
        template="plotly_white",
        xaxis_title="Distance evaluations per query (lower is better)",
        yaxis_title=f"Recall@{K}",
        title=f"Pendigits: {len(points)} points in {points.shape[1]}D, {len(queries)} queries",
        yaxis={"range": [0.8, 1.01]},
        legend={"x": 1, "y": 0, "xanchor": "right", "yanchor": "bottom"},
        height=600,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        OUT, include_plotlyjs=True, config={"responsive": True, "displaylogo": False}
    )
    print(f"\nSaved: {OUT.resolve()}")


if __name__ == "__main__":
    main()
