import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from shared import (
    CHECKPOINTS_DIR,
    HEART_DATASET_DIR,
    RESULTS_DIR,
    ExitMode,
    load_tuning_cache,
    save_checkpoint,
    save_tuning_cache,
)

import numpy as np
import torch

from eval_oracle import evaluate_oracle
from evaluate import evaluate_model
from main import read_data, test, train
from model import (
    BackboneConfig,
    ExitConfig,
    GCNBackbone,
    GCNNodeAdaptiveExit,
    GCNResidualBackbone,
    GCNResidualNodeAdaptiveExit,
    GCNResidualSubgraphAdaptiveExit,
    GCNSubgraphAdaptiveExit,
)
from scoring import mlp_score
from utils import init_seed

FIGURES_DIR = RESULTS_DIR / "figures"

ORACLE_DEPTHS = [1, 2, 4, 8, 12]
DATASETS = ["cora", "citeseer", "pubmed"]


@dataclass(frozen=True)
class HeaRTGCNConfig:
    hidden_channels: int
    num_layers: int
    dropout: float
    lr: float
    l2: float
    num_layers_predictor: int


HEART_CONFIGS: dict[str, HeaRTGCNConfig] = {
    "cora": HeaRTGCNConfig(128, 1, 0.3, 0.01, 1e-4, 3),
    "citeseer": HeaRTGCNConfig(128, 1, 0.3, 0.01, 1e-4, 3),
    "pubmed": HeaRTGCNConfig(256, 1, 0.1, 0.01, 0.0, 2),
}


def build_gcn_model(
    backbone_config: BackboneConfig,
    exit_mode: ExitMode,
    exit_config: ExitConfig | None = None,
) -> torch.nn.Module:
    match exit_mode:
        case ExitMode.NONE:
            return GCNBackbone(backbone_config)
        case ExitMode.NODE_ADAPTIVE:
            assert exit_config is not None
            return GCNNodeAdaptiveExit(backbone_config, exit_config)
        case ExitMode.SUBGRAPH_ADAPTIVE:
            assert exit_config is not None
            return GCNSubgraphAdaptiveExit(backbone_config, exit_config)


def load_gcn_at_depth(
    checkpoint: dict,
    depth: int,
    device: torch.device,
) -> torch.nn.Module:
    bc = checkpoint["backbone_config"]
    backbone_config = BackboneConfig(
        in_channels=bc["in_channels"],
        hidden_channels=bc["hidden_channels"],
        num_layers=depth,
        dropout=bc["dropout"],
    )
    model = GCNBackbone(backbone_config).to(device)

    full_state = checkpoint["model_state_dict"]
    filtered = {k: v for k, v in full_state.items() if not k.startswith("convs.") or int(k.split(".")[1]) < depth}
    model.load_state_dict(filtered)
    return model


def build_residual_gcn_model(
    backbone_config: BackboneConfig,
    exit_mode: ExitMode,
    exit_config: ExitConfig | None = None,
) -> torch.nn.Module:
    match exit_mode:
        case ExitMode.NONE:
            return GCNResidualBackbone(backbone_config)
        case ExitMode.NODE_ADAPTIVE:
            assert exit_config is not None
            return GCNResidualNodeAdaptiveExit(backbone_config, exit_config)
        case ExitMode.SUBGRAPH_ADAPTIVE:
            assert exit_config is not None
            return GCNResidualSubgraphAdaptiveExit(backbone_config, exit_config)


def load_residual_gcn_at_depth(
    checkpoint: dict,
    depth: int,
    device: torch.device,
) -> torch.nn.Module:
    bc = checkpoint["backbone_config"]
    backbone_config = BackboneConfig(
        in_channels=bc["in_channels"],
        hidden_channels=bc["hidden_channels"],
        num_layers=depth,
        dropout=bc["dropout"],
    )
    model = GCNResidualBackbone(backbone_config).to(device)

    full_state = checkpoint["model_state_dict"]
    filtered = {k: v for k, v in full_state.items() if not k.startswith("convs.") or int(k.split(".")[1]) < depth}
    model.load_state_dict(filtered)
    return model


def _backbone_prefix(residual: bool) -> str:
    return "resgcn" if residual else "gcn"


def ckpt_path(dataset: str, exit_mode: ExitMode, num_layers: int, residual: bool = False) -> Path:
    prefix = _backbone_prefix(residual)
    return CHECKPOINTS_DIR / f"{prefix}_{dataset}_{exit_mode.value}_L{num_layers}.pt"


def make_cache_key(
    dataset: str,
    exit_mode: ExitMode,
    num_layers: int,
    dropout: float,
    lr: float,
    tau0: float,
    residual: bool = False,
) -> str:
    prefix = _backbone_prefix(residual)
    return f"{prefix}_{dataset}_{exit_mode.value}_L{num_layers}_d{dropout}_lr{lr}_tau{tau0}"


def train_gcn(
    dataset: str,
    exit_mode: ExitMode,
    num_layers: int,
    hidden_channels: int,
    dropout: float,
    lr: float,
    l2: float,
    tau0: float,
    num_layers_predictor: int,
    data: dict,
    device: torch.device,
    epochs: int = 500,
    kill_cnt_limit: int = 50,
    save: bool = False,
    residual: bool = False,
    confidence_hidden_dim: int = 32,
    seed: int = 999,
) -> tuple[float, float, Path | None]:
    init_seed(seed)

    x = data["x"].to(device)
    train_pos = data["train_pos"].to(device)
    input_channel = x.size(1)

    backbone_config = BackboneConfig(
        in_channels=input_channel,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )

    exit_config = (
        ExitConfig(tau0=tau0, confidence_hidden_dim=confidence_hidden_dim) if exit_mode != ExitMode.NONE else None
    )
    builder = build_residual_gcn_model if residual else build_gcn_model
    model = builder(backbone_config, exit_mode, exit_config).to(device)

    score_func = mlp_score(
        hidden_channels,
        hidden_channels,
        1,
        num_layers_predictor,
        dropout,
    ).to(device)

    model.reset_parameters()
    score_func.reset_parameters()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(score_func.parameters()),
        lr=lr,
        weight_decay=l2,
    )

    from ogb.linkproppred import Evaluator

    evaluator_mrr = Evaluator(name="ogbl-citation2")

    best_val_mrr = 0.0
    test_mrr_at_best = 0.0
    kill_cnt = 0
    best_model_state: dict | None = None
    best_score_state: dict | None = None
    best_epoch = 0

    for epoch in range(1, 1 + epochs):
        train(model, score_func, train_pos, x, optimizer, 1024)

        if epoch % 5 == 0:
            results = test(model, score_func, data, x, evaluator_mrr, 1024)
            val_mrr = results["MRR"][1]
            test_mrr = results["MRR"][2]

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                test_mrr_at_best = test_mrr
                kill_cnt = 0
                if save:
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_score_state = {k: v.cpu().clone() for k, v in score_func.state_dict().items()}
                    best_epoch = epoch
            else:
                kill_cnt += 1
                if kill_cnt > kill_cnt_limit:
                    break

    save_path = None
    if save and best_model_state is not None and best_score_state is not None:
        save_path = ckpt_path(dataset, exit_mode, num_layers, residual=residual)
        save_checkpoint(
            save_path=save_path,
            model_state=best_model_state,
            score_state=best_score_state,
            backbone_config=backbone_config,
            exit_mode=exit_mode,
            dataset=dataset,
            tau0=tau0,
            confidence_hidden_dim=confidence_hidden_dim,
            num_layers_predictor=num_layers_predictor,
            best_epoch=best_epoch,
            backbone_type="residual_gcn" if residual else "gcn",
        )
        print(f"  Saved {save_path.name} (epoch {best_epoch}, val MRR: {best_val_mrr:.4f})")

    return best_val_mrr, test_mrr_at_best, save_path


def plot_gcn_dataset(dataset: str, l12_data: dict) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("ticks")
    matplotlib.rcParams.update(
        {
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
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
        }
    )

    COLORS = {
        "none": "#000000",
        "node_adaptive": "#0072b2",
        "subgraph_adaptive": "#009e73",
        "oracle": "#d55e00",
    }
    MODE_LABELS = {
        "none": "Baseline",
        "node_adaptive": "Node-Adaptive (A)",
        "subgraph_adaptive": "Subgraph-Adaptive (B)",
        "oracle": "Baseline-Oracle",
    }

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.5 * 0.85))
    fig.subplots_adjust(hspace=0.55, wspace=0.40, top=0.90, bottom=0.08, left=0.10, right=0.97)

    baseline = l12_data["none"]
    node_adp = l12_data["node_adaptive"]
    sub_adp = l12_data["subgraph_adaptive"]
    oracle = l12_data["oracle"]
    num_layers = baseline["num_layers"]

    ax = axes[0, 0]
    layers = list(range(num_layers + 1))
    ax.plot(
        layers,
        node_adp["edge_fraction_by_layer"],
        color=COLORS["node_adaptive"],
        label=MODE_LABELS["node_adaptive"],
        marker="o",
    )
    ax.plot(
        layers,
        sub_adp["edge_fraction_by_layer"],
        color=COLORS["subgraph_adaptive"],
        label=MODE_LABELS["subgraph_adaptive"],
        marker="s",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of edges resolved")
    ax.set_title("a", fontweight="bold", loc="left")
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[0, 1]
    num_nodes = sum(baseline["exit_distribution"])
    baseline_cost = [num_nodes * (i + 1) for i in range(num_layers)]
    ax.plot(range(num_layers), baseline_cost, color=COLORS["none"], label=MODE_LABELS["none"], linestyle="--")
    ax.plot(
        range(len(node_adp["cumulative_compute_cost"])),
        node_adp["cumulative_compute_cost"],
        color=COLORS["node_adaptive"],
        label=MODE_LABELS["node_adaptive"],
        marker="o",
    )
    ax.plot(
        range(len(sub_adp["cumulative_compute_cost"])),
        sub_adp["cumulative_compute_cost"],
        color=COLORS["subgraph_adaptive"],
        label=MODE_LABELS["subgraph_adaptive"],
        marker="s",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative node-layers")
    ax.set_title("b", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    metric_keys = ["test_mrr", "test_hits_at_1", "test_hits_at_10"]
    oracle_keys = ["oracle_mrr", "oracle_hits_at_1", "oracle_hits_at_10"]
    metric_labels = ["MRR", "Hits@1", "Hits@10"]
    x_pos = np.arange(len(metric_keys))
    bar_width = 0.20

    for i, (model_type, values, label) in enumerate(
        [
            ("none", [baseline[k] for k in metric_keys], MODE_LABELS["none"]),
            ("node_adaptive", [node_adp[k] for k in metric_keys], MODE_LABELS["node_adaptive"]),
            ("subgraph_adaptive", [sub_adp[k] for k in metric_keys], MODE_LABELS["subgraph_adaptive"]),
            ("oracle", [oracle[k] for k in oracle_keys], MODE_LABELS["oracle"]),
        ]
    ):
        ax.bar(x_pos + i * bar_width, values, bar_width, color=COLORS[model_type], label=label)

    ax.set_xticks(x_pos + 1.5 * bar_width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("c", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for model_type, result in [("node_adaptive", node_adp), ("subgraph_adaptive", sub_adp)]:
        auroc = result["per_layer_auroc"]
        counts = result["per_layer_edge_count"]
        if not auroc:
            continue
        sorted_layers = sorted(int(k) for k in auroc)
        auroc_vals = [auroc[str(ly)] for ly in sorted_layers]
        sizes = [max(3, min(counts[str(ly)] / 5, 60)) for ly in sorted_layers]
        ax.scatter(
            sorted_layers,
            auroc_vals,
            s=sizes,
            color=COLORS[model_type],
            label=MODE_LABELS[model_type],
            alpha=0.8,
            zorder=3,
        )
        ax.plot(sorted_layers, auroc_vals, color=COLORS[model_type], alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Resolution layer")
    ax.set_ylabel("AUROC")
    ax.set_title("d", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    fig.suptitle(f"GCN — {dataset.capitalize()}", fontsize=12, fontweight="bold")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"gcn_{dataset}_results.pdf")
    fig.savefig(FIGURES_DIR / f"gcn_{dataset}_results.png")
    plt.close(fig)
    print(f"  Saved GCN figures for {dataset}")


def run_residual_comparison(device: torch.device) -> None:
    print(f"\n{'=' * 60}")
    print("PHASE 5: RESIDUAL GCN FAIR COMPARISON")
    print(f"{'=' * 60}")

    cache = load_tuning_cache()

    baseline_grid = list(product([1, 2, 3, 4], [0.0, 0.1], [0.001]))
    exit_grid = list(product([2, 3, 4], [0.0, 0.1], [0.0, 0.5, 1.0]))

    datasets = ["pubmed", "cora", "citeseer"]
    all_results: dict[str, dict] = {}

    for dataset in datasets:
        print(f"\n{'─' * 40}")
        print(f"  Dataset: {dataset}")
        print(f"{'─' * 40}")
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        all_results[dataset] = {"baseline": {}, "exit": {}}

        for nl, do, lr in baseline_grid:
            key = make_cache_key(dataset, ExitMode.NONE, nl, do, lr, 0.0, residual=True)
            if key in cache:
                print(f"    {key} (cached): val={cache[key]['val_mrr']:.4f}")
            else:
                print(f"    {key}: training...", end=" ", flush=True)
                val_mrr, test_mrr, _ = train_gcn(
                    dataset=dataset,
                    exit_mode=ExitMode.NONE,
                    num_layers=nl,
                    hidden_channels=256,
                    dropout=do,
                    lr=lr,
                    l2=0.0,
                    tau0=0.0,
                    num_layers_predictor=3,
                    data=data,
                    device=device,
                    epochs=500,
                    kill_cnt_limit=50,
                    residual=True,
                )
                cache[key] = {"val_mrr": val_mrr, "test_mrr": test_mrr}
                save_tuning_cache(cache)
                print(f"val={val_mrr:.4f}, test={test_mrr:.4f}")

        for target_l in [1, 2, 3, 4]:
            best_val = -1.0
            best_do = 0.0
            for nl, do, lr in baseline_grid:
                if nl != target_l:
                    continue
                key = make_cache_key(dataset, ExitMode.NONE, nl, do, lr, 0.0, residual=True)
                if cache[key]["val_mrr"] > best_val:
                    best_val = cache[key]["val_mrr"]
                    best_do = do
            all_results[dataset]["baseline"][target_l] = {
                "dropout": best_do,
                "val_mrr": best_val,
                "test_mrr": cache[
                    make_cache_key(dataset, ExitMode.NONE, target_l, best_do, 0.001, 0.0, residual=True)
                ]["test_mrr"],
            }
            print(f"  Best baseline L={target_l}: do={best_do}, val={best_val:.4f}")

        for target_l in [1, 2, 3, 4]:
            cp = ckpt_path(dataset, ExitMode.NONE, target_l, residual=True)
            if cp.exists():
                print(f"  Checkpoint {cp.name} exists, skipping")
                continue
            best_do = all_results[dataset]["baseline"][target_l]["dropout"]
            print(f"  Training best baseline L={target_l} (do={best_do}) with checkpoint...")
            train_gcn(
                dataset=dataset,
                exit_mode=ExitMode.NONE,
                num_layers=target_l,
                hidden_channels=256,
                dropout=best_do,
                lr=0.001,
                l2=0.0,
                tau0=0.0,
                num_layers_predictor=3,
                data=data,
                device=device,
                save=True,
                residual=True,
            )

        for nl, do, tau0 in exit_grid:
            key = make_cache_key(dataset, ExitMode.NODE_ADAPTIVE, nl, do, 0.001, tau0, residual=True)
            if key in cache:
                print(f"    {key} (cached): val={cache[key]['val_mrr']:.4f}")
            else:
                print(f"    {key}: training...", end=" ", flush=True)
                val_mrr, test_mrr, _ = train_gcn(
                    dataset=dataset,
                    exit_mode=ExitMode.NODE_ADAPTIVE,
                    num_layers=nl,
                    hidden_channels=256,
                    dropout=do,
                    lr=0.001,
                    l2=0.0,
                    tau0=tau0,
                    num_layers_predictor=3,
                    data=data,
                    device=device,
                    epochs=500,
                    kill_cnt_limit=50,
                    residual=True,
                )
                cache[key] = {"val_mrr": val_mrr, "test_mrr": test_mrr}
                save_tuning_cache(cache)
                print(f"val={val_mrr:.4f}, test={test_mrr:.4f}")

        for target_l in [2, 3, 4]:
            best_val = -1.0
            best_do, best_tau0 = 0.0, 0.0
            for nl, do, tau0 in exit_grid:
                if nl != target_l:
                    continue
                key = make_cache_key(dataset, ExitMode.NODE_ADAPTIVE, nl, do, 0.001, tau0, residual=True)
                if cache[key]["val_mrr"] > best_val:
                    best_val = cache[key]["val_mrr"]
                    best_do, best_tau0 = do, tau0
            best_key = make_cache_key(
                dataset, ExitMode.NODE_ADAPTIVE, target_l, best_do, 0.001, best_tau0, residual=True
            )
            all_results[dataset]["exit"][target_l] = {
                "dropout": best_do,
                "tau0": best_tau0,
                "val_mrr": best_val,
                "test_mrr": cache[best_key]["test_mrr"],
            }
            print(f"  Best exit L={target_l}: do={best_do}, tau0={best_tau0}, val={best_val:.4f}")

        for target_l in [2, 3, 4]:
            cp = ckpt_path(dataset, ExitMode.NODE_ADAPTIVE, target_l, residual=True)
            if cp.exists():
                print(f"  Checkpoint {cp.name} exists, skipping")
                continue
            info = all_results[dataset]["exit"][target_l]
            print(f"  Training best exit L={target_l} (do={info['dropout']}, tau0={info['tau0']}) with checkpoint...")
            train_gcn(
                dataset=dataset,
                exit_mode=ExitMode.NODE_ADAPTIVE,
                num_layers=target_l,
                hidden_channels=256,
                dropout=info["dropout"],
                lr=0.001,
                l2=0.0,
                tau0=info["tau0"],
                num_layers_predictor=3,
                data=data,
                device=device,
                save=True,
                residual=True,
            )

        print(f"\n  Evaluating {dataset}...")
        for target_l in [1, 2, 3, 4]:
            cp = ckpt_path(dataset, ExitMode.NONE, target_l, residual=True)
            result = evaluate_model(cp, data, device)
            all_results[dataset]["baseline"][target_l]["eval"] = asdict(result)
            print(f"    Fixed L={target_l}: MRR={result.test_mrr * 100:.2f}%")

        for target_l in [2, 3, 4]:
            cp = ckpt_path(dataset, ExitMode.NODE_ADAPTIVE, target_l, residual=True)
            result = evaluate_model(cp, data, device)
            all_results[dataset]["exit"][target_l]["eval"] = asdict(result)
            cost = result.total_compute_cost
            bl_cost = all_results[dataset]["baseline"][target_l]["eval"]["total_compute_cost"]
            reduction = (1 - cost / bl_cost) * 100 if bl_cost > 0 else 0
            print(
                f"    Exit  L={target_l}: MRR={result.test_mrr * 100:.2f}% "
                f"(compute: {reduction:.0f}% less than fixed L={target_l})"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "residual_gcn_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("RESIDUAL GCN COMPARISON SUMMARY")
    print(f"{'=' * 60}")

    for dataset in datasets:
        r = all_results[dataset]
        print(f"\n{dataset.capitalize()}")
        print(f"{'Model':<35} {'MRR':>8} {'Hits@1':>8} {'Hits@10':>8} {'Compute':>10}")
        print("─" * 76)

        for target_l in [1, 2, 3, 4]:
            e = r["baseline"][target_l]["eval"]
            print(
                f"{'Fixed Residual-GCN L=' + str(target_l):<35} "
                f"{e['test_mrr'] * 100:>7.2f}% {e['test_hits_at_1'] * 100:>7.2f}% "
                f"{e['test_hits_at_10'] * 100:>7.2f}% {e['total_compute_cost']:>9.0f}"
            )

        for target_l in [2, 3, 4]:
            e = r["exit"][target_l]["eval"]
            bl_cost = r["baseline"][target_l]["eval"]["total_compute_cost"]
            red = (1 - e["total_compute_cost"] / bl_cost) * 100 if bl_cost > 0 else 0
            print(
                f"{'Exit Residual-GCN  L=' + str(target_l):<35} "
                f"{e['test_mrr'] * 100:>7.2f}% {e['test_hits_at_1'] * 100:>7.2f}% "
                f"{e['test_hits_at_10'] * 100:>7.2f}% {e['total_compute_cost']:>9.0f} ({red:.0f}%↓)"
            )

    print("\nDone! Results saved to", RESULTS_DIR / "residual_gcn_comparison.json")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\n{'=' * 60}")
    print("PHASE 0: VALIDATING GCN BASELINE AGAINST HeaRT")
    print(f"{'=' * 60}")

    expected_mrr = {"cora": 16.65, "citeseer": 21.02}

    for dataset in ["cora", "citeseer"]:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        hcfg = HEART_CONFIGS[dataset]
        print(
            f"\n  {dataset}: h={hcfg.hidden_channels}, L={hcfg.num_layers}, "
            f"do={hcfg.dropout}, lr={hcfg.lr}, l2={hcfg.l2}, pred={hcfg.num_layers_predictor}"
        )

        val_mrr, test_mrr, _ = train_gcn(
            dataset=dataset,
            exit_mode=ExitMode.NONE,
            num_layers=hcfg.num_layers,
            hidden_channels=hcfg.hidden_channels,
            dropout=hcfg.dropout,
            lr=hcfg.lr,
            l2=hcfg.l2,
            tau0=0.0,
            num_layers_predictor=hcfg.num_layers_predictor,
            data=data,
            device=device,
        )
        delta = abs(test_mrr * 100 - expected_mrr[dataset])
        status = "PASS" if delta < 3.0 else "WARNING"
        print(
            f"  → val={val_mrr * 100:.2f}%, test={test_mrr * 100:.2f}% "
            f"(expected ~{expected_mrr[dataset]}%, Δ={delta:.2f}%) [{status}]"
        )

    print(f"\n{'=' * 60}")
    print("PHASE 1: HYPERPARAMETER TUNING (h=256)")
    print(f"{'=' * 60}")

    cache = load_tuning_cache()

    baseline_grid = list(product([1, 2, 3, 4], [0.0, 0.1, 0.2], [0.001]))
    exit_grid_small = list(product([4, 8], [0.0, 0.1], [0.0, 0.5, 1.0]))
    exit_grid_full = list(product([4, 8, 12], [0.0, 0.1], [0.0, 0.5, 1.0]))

    best_configs: dict[str, dict[str, tuple[int, float, float, float, float]]] = {}

    for dataset in DATASETS:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        best_configs[dataset] = {}

        is_pubmed = dataset == "pubmed"
        tune_epochs = 200 if is_pubmed else 500
        tune_kill_cnt = 20 if is_pubmed else 50
        exit_grid = exit_grid_small if is_pubmed else exit_grid_full

        for exit_mode in ExitMode:
            grid = (
                [(nl, do, 0.001, 0.0) for nl, do, lr in baseline_grid]
                if exit_mode == ExitMode.NONE
                else [(nl, do, 0.001, tau0) for nl, do, tau0 in exit_grid]
            )

            print(f"\n  Tuning GCN {exit_mode.value} on {dataset} ({len(grid)} configs)")
            best_val = -1.0
            best_entry: tuple[int, float, float, float, float] | None = None

            for i, (nl, do, lr, tau0) in enumerate(grid):
                key = make_cache_key(dataset, exit_mode, nl, do, lr, tau0)
                if key in cache:
                    val_mrr = cache[key]["val_mrr"]
                    test_mrr = cache[key]["test_mrr"]
                    print(f"    [{i + 1}/{len(grid)}] {key} (cached): val={val_mrr:.4f}")
                else:
                    print(f"    [{i + 1}/{len(grid)}] {key}: training...", end=" ", flush=True)
                    val_mrr, test_mrr, _ = train_gcn(
                        dataset=dataset,
                        exit_mode=exit_mode,
                        num_layers=nl,
                        hidden_channels=256,
                        dropout=do,
                        lr=lr,
                        l2=0.0,
                        tau0=tau0,
                        num_layers_predictor=3,
                        data=data,
                        device=device,
                        epochs=tune_epochs,
                        kill_cnt_limit=tune_kill_cnt,
                    )
                    cache[key] = {"val_mrr": val_mrr, "test_mrr": test_mrr}
                    save_tuning_cache(cache)
                    print(f"val={val_mrr:.4f}, test={test_mrr:.4f}")

                if val_mrr > best_val:
                    best_val = val_mrr
                    best_entry = (nl, do, lr, tau0, test_mrr)

            assert best_entry is not None
            best_configs[dataset][exit_mode.value] = best_entry
            nl, do, lr, tau0, test_mrr = best_entry
            print(
                f"  BEST GCN {exit_mode.value} on {dataset}: L={nl}, do={do}, tau0={tau0}, "
                f"val={best_val:.4f}, test={test_mrr:.4f}"
            )

    print(f"\n{'=' * 60}")
    print("PHASE 2: TRAINING L=12 MODELS")
    print(f"{'=' * 60}")

    for dataset in DATASETS:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")

        for exit_mode in ExitMode:
            mode_str = exit_mode.value
            _, do, _, tau0, _ = best_configs[dataset][mode_str]
            l12_tau0 = tau0 if exit_mode != ExitMode.NONE else 0.0

            cp = ckpt_path(dataset, exit_mode, 12)
            if cp.exists():
                ckpt = torch.load(cp, map_location="cpu", weights_only=False)
                if ckpt.get("backbone_type") == "gcn" and ckpt["backbone_config"]["hidden_channels"] == 256:
                    print(f"\n  GCN {mode_str} L=12 on {dataset}: checkpoint exists, skipping")
                    continue

            print(f"\n  Training GCN {mode_str} L=12 on {dataset} (do={do}, tau0={l12_tau0})")
            train_gcn(
                dataset=dataset,
                exit_mode=exit_mode,
                num_layers=12,
                hidden_channels=256,
                dropout=do,
                lr=0.001,
                l2=0.0,
                tau0=l12_tau0,
                num_layers_predictor=3,
                data=data,
                device=device,
                save=True,
            )

    print(f"\n{'=' * 60}")
    print("PHASE 3: EVALUATION")
    print(f"{'=' * 60}")

    all_l12_results: dict[str, dict] = {}

    for dataset in DATASETS:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        all_l12_results[dataset] = {}

        for mode_str in ["none", "node_adaptive", "subgraph_adaptive"]:
            cp = CHECKPOINTS_DIR / f"gcn_{dataset}_{mode_str}_L12.pt"
            print(f"\n  Evaluating GCN {mode_str} L=12 on {dataset}...")
            result = evaluate_model(cp, data, device)
            all_l12_results[dataset][mode_str] = asdict(result)
            print(
                f"    MRR={result.test_mrr * 100:.2f}%, Hits@1={result.test_hits_at_1 * 100:.2f}%, "
                f"Hits@10={result.test_hits_at_10 * 100:.2f}%, cost={result.total_compute_cost:.0f}"
            )

        print(f"\n  Computing GCN per-edge oracle on {dataset} (depths={ORACLE_DEPTHS})...")
        cp = CHECKPOINTS_DIR / f"gcn_{dataset}_none_L12.pt"
        oracle_result = evaluate_oracle(cp, data, device, ORACLE_DEPTHS, model_loader=load_gcn_at_depth)
        all_l12_results[dataset]["oracle"] = oracle_result
        print(f"    Oracle MRR={oracle_result['oracle_mrr'] * 100:.2f}%")
        print(f"    Depth distribution: {oracle_result['depth_distribution']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gcn_l12_comparison.json", "w") as f:
        json.dump(all_l12_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PHASE 4: PLOTTING")
    print(f"{'=' * 60}")

    for dataset in DATASETS:
        try:
            plot_gcn_dataset(dataset, all_l12_results[dataset])
        except Exception as e:
            print(f"  Warning: failed to plot {dataset}: {e}")

    print(f"\n{'=' * 60}")
    print("SUMMARY: GCN BACKBONE (all L=12)")
    print(f"{'=' * 60}")

    for dataset in DATASETS:
        print(f"\n{dataset.capitalize()}")
        print(f"{'Model':<30} {'MRR':>8} {'Hits@1':>8} {'Hits@10':>8} {'Cost':>8}")
        print("-" * 72)

        r = all_l12_results[dataset]
        bl = r["none"]
        print(
            f"{'GCN Baseline (L=12)':<30} {bl['test_mrr'] * 100:>7.2f}% "
            f"{bl['test_hits_at_1'] * 100:>7.2f}% "
            f"{bl['test_hits_at_10'] * 100:>7.2f}% "
            f"{bl['total_compute_cost']:>7.0f}"
        )

        for mode in ["node_adaptive", "subgraph_adaptive"]:
            mr = r[mode]
            red = (1 - mr["total_compute_cost"] / bl["total_compute_cost"]) * 100
            label = {"node_adaptive": "GCN Node-Adpt", "subgraph_adaptive": "GCN Sub-Adpt"}[mode]
            print(
                f"{label + ' (L=12)':<30} {mr['test_mrr'] * 100:>7.2f}% "
                f"{mr['test_hits_at_1'] * 100:>7.2f}% "
                f"{mr['test_hits_at_10'] * 100:>7.2f}% "
                f"{mr['total_compute_cost']:>7.0f} ({red:.0f}%↓)"
            )

        o = r["oracle"]
        print(
            f"{'GCN Oracle':<30} {o['oracle_mrr'] * 100:>7.2f}% "
            f"{o['oracle_hits_at_1'] * 100:>7.2f}% "
            f"{o['oracle_hits_at_10'] * 100:>7.2f}% "
            f"{'—':>8}"
        )
        print(f"  Depth distribution: {o['depth_distribution']}")

    tuning_summary: dict[str, dict] = {}
    for dataset in DATASETS:
        tuning_summary[dataset] = {}
        for exit_mode in ExitMode:
            nl, do, lr, tau0, test_mrr = best_configs[dataset][exit_mode.value]
            tuning_summary[dataset][exit_mode.value] = {
                "best_L": nl,
                "dropout": do,
                "tau0": tau0,
                "test_mrr": test_mrr,
            }
    with open(RESULTS_DIR / "gcn_best_configs.json", "w") as f:
        json.dump(tuning_summary, f, indent=2)

    print("\nPhases 0-4 done!")

    run_residual_comparison(device)


if __name__ == "__main__":
    if "--phase5" in sys.argv:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        run_residual_comparison(device)
    else:
        main()
