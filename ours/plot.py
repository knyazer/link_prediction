import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TEXT_WIDTH_MM = 183
TEXT_WIDTH_IN = TEXT_WIDTH_MM / 25.4

COLORS = {
    "none": "#000000",
    "node_adaptive": "#0072b2",
    "subgraph_adaptive": "#009e73",
    "oracle": "#d55e00",
}

LABELS = {
    "none": "Baseline",
    "node_adaptive": "Node Early Exiting",
    "subgraph_adaptive": "Subgraph Early Exiting",
    "oracle": "Oracle",
}

COLUMN_TITLES = ["Edge fraction by layer", "Cumulative FLOPs", "Metric comparison"]


def setup_style() -> None:
    sns.set_style("ticks")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "standard",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
        }
    )


def plot_edge_fraction(ax: plt.Axes, data: dict) -> None:
    num_layers = data["none"]["num_layers"]
    layers = list(range(1, num_layers + 2))
    for mode in ["node_adaptive", "subgraph_adaptive"]:
        marker = "o" if mode == "node_adaptive" else "s"
        ax.plot(layers, data[mode]["edge_fraction_by_layer"], color=COLORS[mode], label=LABELS[mode], marker=marker)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction resolved")
    ax.set_ylim(-0.05, 1.05)


def plot_flops(ax: plt.Axes, data: dict) -> None:
    num_layers = data["none"]["num_layers"]

    for mode in ["none", "node_adaptive", "subgraph_adaptive"]:
        flops = data[mode]["flops_per_layer"]
        cumulative = list(np.cumsum(flops))
        style = {"linestyle": "--"} if mode == "none" else {"marker": "o" if mode == "node_adaptive" else "s"}
        ax.plot(range(1, num_layers + 1), cumulative, color=COLORS[mode], label=LABELS[mode], **style)
    ax.set_yscale("log", subs=[2, 5])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(minor_thresholds=(2, 1)))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative FLOPs")


def plot_metrics(ax: plt.Axes, data: dict) -> None:
    metric_keys = ["test_mrr", "test_hits_at_1", "test_hits_at_10"]
    oracle_keys = ["oracle_mrr", "oracle_hits_at_1", "oracle_hits_at_10"]
    metric_labels = ["MRR", "Hits@1", "Hits@10"]

    x_pos = np.arange(len(metric_keys))
    bar_width = 0.25

    models = [
        ("none", [data["none"][k] for k in metric_keys]),
        ("node_adaptive", [data["node_adaptive"][k] for k in metric_keys]),
        ("subgraph_adaptive", [data["subgraph_adaptive"][k] for k in metric_keys]),
    ]

    for i, (model_type, values) in enumerate(models):
        ax.bar(x_pos + i * bar_width, values, bar_width, color=COLORS[model_type], label=LABELS[model_type])

    oracle_values = [data["oracle"][k] for k in oracle_keys]
    for j, val in enumerate(oracle_values):
        center = x_pos[j] + bar_width
        left = center - 0.35
        right = center + 0.35
        ax.hlines(
            val, left, right, colors="red", linestyles="--", linewidth=1.0, label=LABELS["oracle"] if j == 0 else None
        )

    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")


def make_figure(all_data: dict, datasets: list[str], backbone_name: str) -> plt.Figure:
    n_rows = len(datasets)
    n_cols = len(COLUMN_TITLES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.65))
    fig.subplots_adjust(
        hspace=0.65,
        wspace=0.45,
        top=0.95,
        bottom=0.15,
        left=0.09,
        right=0.97,
    )

    plot_fns = [plot_edge_fraction, plot_flops, plot_metrics]

    for row, dataset in enumerate(datasets):
        if dataset not in all_data:
            for col in range(n_cols):
                axes[row, col].set_visible(False)
            continue

        data = all_data[dataset]
        for col, plot_fn in enumerate(plot_fns):
            ax = axes[row, col]
            plot_fn(ax, data)

    for row, dataset in enumerate(datasets):
        label = dataset.upper() if dataset.startswith("ogbl") else dataset.capitalize()
        row_center = axes[row, 0].get_position()
        fig.text(
            0.01,
            (row_center.y0 + row_center.y1) / 2,
            label,
            fontsize=9,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    legend_handles = [
        plt.Line2D([0], [0], color=COLORS["none"], linestyle="--", label=LABELS["none"]),
        plt.Line2D([0], [0], color=COLORS["node_adaptive"], marker="o", label=LABELS["node_adaptive"]),
        plt.Line2D([0], [0], color=COLORS["subgraph_adaptive"], marker="s", label=LABELS["subgraph_adaptive"]),
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.0, label=LABELS["oracle"]),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.54, 0.0))

    return fig


def main() -> None:
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ("l12_comparison.json", ["citeseer", "pubmed", "ogbl-ddi"], "SAS", "combined_results_sas"),
        ("gcn_l12_comparison.json", ["citeseer", "pubmed", "ogbl-ddi"], "GCN", "combined_results_gcn"),
    ]

    for json_file, datasets, backbone_name, output_stem in configs:
        json_path = RESULTS_DIR / json_file
        if not json_path.exists():
            print(f"Skipping {backbone_name}: {json_path} not found")
            continue

        with open(json_path) as f:
            all_data = json.load(f)

        fig = make_figure(all_data, datasets, backbone_name)
        fig.savefig(FIGURES_DIR / f"{output_stem}.pdf")
        fig.savefig(FIGURES_DIR / f"{output_stem}.png")
        plt.close(fig)
        print(f"Saved {backbone_name} figure to {FIGURES_DIR / output_stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()
