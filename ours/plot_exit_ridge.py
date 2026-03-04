import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RIDGE_DIR = RESULTS_DIR / "ridge"

TEXT_WIDTH_MM = 183
TEXT_WIDTH_IN = TEXT_WIDTH_MM / 25.4

DATASET_LABELS: dict[str, str] = {
    "citeseer": "Citeseer",
    "pubmed": "PubMed",
}


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
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "lines.linewidth": 1.2,
        }
    )


def load_multiseed_data(path: Path) -> dict[str, dict[int, list[list[int]]]]:
    with open(path) as f:
        raw = json.load(f)

    return {dataset: {int(depth): dists for depth, dists in depth_data.items()} for dataset, depth_data in raw.items()}


def average_proportions(distributions: list[list[int]]) -> np.ndarray:
    max_len = max(len(d) for d in distributions)
    prop_matrix = np.zeros((len(distributions), max_len))
    for i, dist in enumerate(distributions):
        counts = np.array(dist, dtype=float)
        prop_matrix[i, : len(dist)] = counts / counts.sum()
    return prop_matrix.mean(axis=0)


def plot_multiseed(
    data: dict[str, dict[int, list[list[int]]]],
    out_stem: str,
) -> None:
    setup_style()

    datasets = [d for d in ["citeseer", "pubmed"] if d in data]
    all_depths = sorted({d for dd in data.values() for d in dd if d <= 16})
    pal = sns.color_palette("crest", n_colors=len(all_depths))

    fig, axes = plt.subplots(1, len(datasets), figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.35), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for col, dataset in enumerate(datasets):
        ax = axes[col]
        for depth in all_depths:
            if depth not in data[dataset]:
                continue
            mean_props = average_proportions(data[dataset][depth])
            layers = np.arange(len(mean_props))
            color = pal[all_depths.index(depth)]
            ax.plot(layers, mean_props, color=color, label=f"L = {depth}")

        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel("Exit layer")
        if col == 0:
            ax.set_ylabel("Fraction of nodes")
        ax.legend(frameon=False)

    fig.tight_layout()

    RIDGE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(RIDGE_DIR / f"{out_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {RIDGE_DIR / out_stem}.png")


def main() -> None:
    multiseed_path = RIDGE_DIR / "resgcn_exit_data_multiseed.json"

    if multiseed_path.exists():
        plot_multiseed(load_multiseed_data(multiseed_path), "resgcn_exit_plot")
    else:
        print(f"Missing {multiseed_path}")


if __name__ == "__main__":
    main()
