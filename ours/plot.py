import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TEXT_WIDTH = 5.5  # inches, one-column US letter layout

COLORS = {
    "none": "#000000",
    "node_adaptive": "#0072b2",
    "subgraph_adaptive": "#009e73",
    "oracle": "#d55e00",
}

LABELS = {
    "none": "Baseline",
    "node_adaptive": "Node-Adaptive (A)",
    "subgraph_adaptive": "Subgraph-Adaptive (B)",
    "oracle": "Baseline-Oracle",
}


def setup_style() -> None:
    sns.set_style("ticks")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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
    })


def load_result(dataset: str, model_type: str, num_layers: int) -> dict:
    path = RESULTS_DIR / f"{dataset}_{model_type}_L{num_layers}.json"
    with open(path) as f:
        return json.load(f)


def load_l12_comparison() -> dict:
    with open(RESULTS_DIR / "l12_comparison.json") as f:
        return json.load(f)


def plot_dataset(dataset: str) -> None:
    """Plot 4-panel figure for a dataset. All models at L=12 + oracle."""
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.85))
    fig.subplots_adjust(hspace=0.55, wspace=0.40, top=0.90, bottom=0.08, left=0.10, right=0.97)

    l12 = load_l12_comparison()[dataset]
    baseline = l12["none"]
    node_adp = l12["node_adaptive"]
    sub_adp = l12["subgraph_adaptive"]
    oracle = l12["oracle"]

    num_layers = baseline["num_layers"]

    # --- (a) Edge fraction by layer ---
    ax = axes[0, 0]
    layers = list(range(num_layers + 1))
    ax.plot(layers, node_adp["edge_fraction_by_layer"],
            color=COLORS["node_adaptive"], label=LABELS["node_adaptive"], marker="o")
    ax.plot(layers, sub_adp["edge_fraction_by_layer"],
            color=COLORS["subgraph_adaptive"], label=LABELS["subgraph_adaptive"], marker="s")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of edges resolved")
    ax.set_title("a", fontweight="bold", loc="left")
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.05)

    # --- (b) Cumulative compute cost ---
    ax = axes[0, 1]
    num_nodes = sum(baseline["exit_distribution"])
    baseline_cost = [num_nodes * (l + 1) for l in range(num_layers)]

    ax.plot(range(num_layers), baseline_cost,
            color=COLORS["none"], label=LABELS["none"], linestyle="--")
    ax.plot(range(len(node_adp["cumulative_compute_cost"])), node_adp["cumulative_compute_cost"],
            color=COLORS["node_adaptive"], label=LABELS["node_adaptive"], marker="o")
    ax.plot(range(len(sub_adp["cumulative_compute_cost"])), sub_adp["cumulative_compute_cost"],
            color=COLORS["subgraph_adaptive"], label=LABELS["subgraph_adaptive"], marker="s")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative node-layers")
    ax.set_title("b", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    # --- (c) Metric comparison (baseline + exit modes + oracle) ---
    ax = axes[1, 0]
    metric_keys = ["test_mrr", "test_hits_at_1", "test_hits_at_10"]
    oracle_keys = ["oracle_mrr", "oracle_hits_at_1", "oracle_hits_at_10"]
    metric_labels = ["MRR", "Hits@1", "Hits@10"]

    x_pos = np.arange(len(metric_keys))
    bar_width = 0.20

    models_for_bars = [
        ("none", [baseline[k] for k in metric_keys], LABELS["none"]),
        ("node_adaptive", [node_adp[k] for k in metric_keys], LABELS["node_adaptive"]),
        ("subgraph_adaptive", [sub_adp[k] for k in metric_keys], LABELS["subgraph_adaptive"]),
        ("oracle", [oracle[k] for k in oracle_keys], LABELS["oracle"]),
    ]

    for i, (model_type, values, label) in enumerate(models_for_bars):
        ax.bar(x_pos + i * bar_width, values, bar_width,
               color=COLORS[model_type], label=label)

    ax.set_xticks(x_pos + 1.5 * bar_width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("c", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    # --- (d) Per-layer AUROC ---
    ax = axes[1, 1]
    for model_type, result in [("node_adaptive", node_adp), ("subgraph_adaptive", sub_adp)]:
        auroc = result["per_layer_auroc"]
        counts = result["per_layer_edge_count"]
        if not auroc:
            continue

        sorted_layers = sorted(int(k) for k in auroc)
        auroc_vals = [auroc[str(l)] for l in sorted_layers]
        sizes = [max(3, min(counts[str(l)] / 5, 60)) for l in sorted_layers]

        ax.scatter(sorted_layers, auroc_vals, s=sizes,
                   color=COLORS[model_type], label=LABELS[model_type], alpha=0.8, zorder=3)
        ax.plot(sorted_layers, auroc_vals,
                color=COLORS[model_type], alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Resolution layer")
    ax.set_ylabel("AUROC")
    ax.set_title("d", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    fig.suptitle(dataset.capitalize(), fontsize=12, fontweight="bold")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{dataset}_results.pdf")
    fig.savefig(FIGURES_DIR / f"{dataset}_results.png")
    plt.close(fig)
    print(f"Saved figures for {dataset}")


def main() -> None:
    for dataset in ["cora", "citeseer"]:
        plot_dataset(dataset)


if __name__ == "__main__":
    main()
