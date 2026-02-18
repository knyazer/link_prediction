"""GCN backbone experiments: validate against HeaRT, tune, train, evaluate L=12 + oracle, plot.

Runs the same evaluation pipeline as SAS experiments but with standard GCN backbone
(independent per-layer weights, GELU activation) instead of weight-shared SAS.

Phases:
0. Validate: GCN baseline with HeaRT's exact hyperparameters
1. Tune: Grid search over {baseline, node_adaptive, subgraph_adaptive} x hyperparameters
2. Train L=12: Final training at L=12 for all modes on all datasets
3. Evaluate L=12 + per-edge oracle
4. Plot: Generate 4-panel figures
"""

import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

OURS_DIR = Path(__file__).resolve().parent
HEART_BENCHMARKING_DIR = OURS_DIR.parent / "HeaRT" / "benchmarking"
sys.path.insert(0, str(OURS_DIR))
sys.path.insert(0, str(HEART_BENCHMARKING_DIR))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evalutors import evaluate_mrr
from main import ExitMode, get_metric_score, read_data, test, test_edge, train
from model import (
    BackboneConfig,
    ExitConfig,
    ForwardResult,
    GCNBackbone,
    GCNNodeAdaptiveExit,
    GCNSubgraphAdaptiveExit,
)
from scoring import mlp_score
from utils import init_seed

HEART_DATASET_DIR = HEART_BENCHMARKING_DIR.parent / "dataset"
CHECKPOINTS_DIR = OURS_DIR.parent / "checkpoints"
RESULTS_DIR = OURS_DIR.parent / "results"
TUNING_CACHE = RESULTS_DIR / "tuning_cache.json"
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
    checkpoint: dict, depth: int, device: torch.device,
) -> torch.nn.Module:
    """Load a GCNBackbone from checkpoint, truncated to first `depth` layers.

    Since GCN has independent per-layer weights, we load only the first k layers
    from the L=12 checkpoint to evaluate at intermediate depths.
    """
    bc = checkpoint["backbone_config"]
    backbone_config = BackboneConfig(
        in_channels=bc["in_channels"],
        hidden_channels=bc["hidden_channels"],
        num_layers=depth,
        dropout=bc["dropout"],
    )
    model = GCNBackbone(backbone_config).to(device)

    full_state = checkpoint["model_state_dict"]
    filtered = {
        k: v for k, v in full_state.items()
        if not k.startswith("convs.") or int(k.split(".")[1]) < depth
    }
    model.load_state_dict(filtered)
    return model


def ckpt_path(dataset: str, exit_mode: ExitMode, num_layers: int) -> Path:
    return CHECKPOINTS_DIR / f"gcn_{dataset}_{exit_mode.value}_L{num_layers}.pt"


def make_cache_key(
    dataset: str, exit_mode: ExitMode,
    num_layers: int, dropout: float, lr: float, tau0: float,
) -> str:
    return f"gcn_{dataset}_{exit_mode.value}_L{num_layers}_d{dropout}_lr{lr}_tau{tau0}"


def load_tuning_cache() -> dict:
    if TUNING_CACHE.exists():
        with open(TUNING_CACHE) as f:
            return json.load(f)
    return {}


def save_tuning_cache(cache: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TUNING_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


# ============================================================================
# Training
# ============================================================================


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
    save_checkpoint: bool = False,
) -> tuple[float, float, Path | None]:
    """Train GCN model. Returns (best_val_mrr, test_mrr_at_best_val, checkpoint_path)."""
    init_seed(999)

    x = data["x"].to(device)
    train_pos = data["train_pos"].to(device)
    input_channel = x.size(1)

    backbone_config = BackboneConfig(
        in_channels=input_channel,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )

    exit_config = ExitConfig(tau0=tau0, confidence_hidden_dim=32) if exit_mode != ExitMode.NONE else None
    model = build_gcn_model(backbone_config, exit_mode, exit_config).to(device)

    score_func = mlp_score(
        hidden_channels, hidden_channels, 1, num_layers_predictor, dropout,
    ).to(device)

    model.reset_parameters()
    score_func.reset_parameters()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(score_func.parameters()),
        lr=lr, weight_decay=l2,
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
        loss = train(model, score_func, train_pos, x, optimizer, 1024)

        if epoch % 5 == 0:
            results = test(model, score_func, data, x, evaluator_mrr, 1024)
            val_mrr = results["MRR"][1]
            test_mrr = results["MRR"][2]

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                test_mrr_at_best = test_mrr
                kill_cnt = 0
                if save_checkpoint:
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_score_state = {k: v.cpu().clone() for k, v in score_func.state_dict().items()}
                    best_epoch = epoch
            else:
                kill_cnt += 1
                if kill_cnt > kill_cnt_limit:
                    break

    save_path = None
    if save_checkpoint and best_model_state is not None and best_score_state is not None:
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = ckpt_path(dataset, exit_mode, num_layers)
        torch.save({
            "model_state_dict": best_model_state,
            "score_func_state_dict": best_score_state,
            "backbone_config": {
                "in_channels": input_channel,
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            "exit_mode": exit_mode.value,
            "backbone_type": "gcn",
            "data_name": dataset,
            "tau0": tau0,
            "confidence_hidden_dim": 32,
            "num_layers_predictor": num_layers_predictor,
            "best_epoch": best_epoch,
        }, save_path)
        print(f"  Saved {save_path.name} (epoch {best_epoch}, val MRR: {best_val_mrr:.4f})")

    return best_val_mrr, test_mrr_at_best, save_path


# ============================================================================
# Evaluation
# ============================================================================


@dataclass
class GCNEvalResult:
    dataset: str
    model_type: str
    num_layers: int
    test_mrr: float
    test_hits_at_1: float
    test_hits_at_3: float
    test_hits_at_10: float
    test_hits_at_100: float
    exit_distribution: list[int]
    active_nodes_per_layer: list[int]
    edge_fraction_by_layer: list[float]
    cumulative_compute_cost: list[float]
    total_compute_cost: float
    per_layer_auroc: dict[str, float]
    per_layer_edge_count: dict[str, int]


@torch.no_grad()
def evaluate_gcn_checkpoint(
    checkpoint_path: Path, data: dict, device: torch.device,
) -> GCNEvalResult:
    init_seed(999)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bc = checkpoint["backbone_config"]

    backbone_config = BackboneConfig(
        in_channels=bc["in_channels"],
        hidden_channels=bc["hidden_channels"],
        num_layers=bc["num_layers"],
        dropout=bc["dropout"],
    )

    exit_mode = ExitMode(checkpoint["exit_mode"])
    exit_config = ExitConfig(
        tau0=checkpoint["tau0"],
        confidence_hidden_dim=checkpoint["confidence_hidden_dim"],
    ) if exit_mode != ExitMode.NONE else None

    model = build_gcn_model(backbone_config, exit_mode, exit_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    score_func = mlp_score(
        bc["hidden_channels"], bc["hidden_channels"], 1,
        checkpoint["num_layers_predictor"], bc["dropout"],
    )
    score_func.load_state_dict(checkpoint["score_func_state_dict"])
    score_func = score_func.to(device)
    score_func.eval()

    x = data["x"].to(device)
    adj = data["adj"].to(device)
    num_layers = bc["num_layers"]
    num_nodes = x.size(0)

    model_out = model(x, adj)

    if isinstance(model_out, ForwardResult):
        h = model_out.node_embeddings
        exit_layers = model_out.exit_layers
        active_per_layer = model_out.active_nodes_per_layer
    else:
        h = model_out
        exit_layers = torch.full((num_nodes,), num_layers, dtype=torch.long, device=device)
        active_per_layer = [num_nodes] * num_layers

    batch_size = 1024
    pos_train_pred, _ = test_edge(score_func, data["train_val"], h, batch_size)
    pos_valid_pred, neg_valid_pred = test_edge(
        score_func, data["valid_pos"], h, batch_size, data["valid_neg"],
    )
    pos_test_pred, neg_test_pred = test_edge(
        score_func, data["test_pos"], h, batch_size, data["test_neg"],
    )

    from ogb.linkproppred import Evaluator
    evaluator_mrr = Evaluator(name="ogbl-citation2")

    metrics = get_metric_score(
        evaluator_mrr,
        torch.flatten(pos_train_pred),
        torch.flatten(pos_valid_pred),
        neg_valid_pred.squeeze(-1),
        torch.flatten(pos_test_pred),
        neg_test_pred.squeeze(-1),
    )

    test_pos = data["test_pos"]
    exit_layers_cpu = exit_layers.cpu()
    src_exit = exit_layers_cpu[test_pos[:, 0]]
    dst_exit = exit_layers_cpu[test_pos[:, 1]]
    resolution_layers = torch.maximum(src_exit, dst_exit)

    total_edges = resolution_layers.size(0)
    edge_fracs = [
        float((resolution_layers <= layer).sum().item()) / total_edges
        for layer in range(num_layers + 1)
    ]

    cum_cost: list[float] = []
    running = 0.0
    for count in active_per_layer:
        running += count
        cum_cost.append(running)

    exit_dist = [int((exit_layers == layer).sum().item()) for layer in range(num_layers + 1)]

    pos_flat = pos_test_pred.cpu().flatten()
    neg_2d = neg_test_pred.cpu().squeeze(-1)
    auroc_by_layer: dict[str, float] = {}
    count_by_layer: dict[str, int] = {}

    for layer in range(num_layers + 1):
        mask = resolution_layers == layer
        count = int(mask.sum().item())
        if count < 2:
            continue
        layer_pos = pos_flat[mask]
        layer_neg = neg_2d[mask]
        num_neg = layer_neg.size(1)
        labels = np.concatenate([np.ones(count), np.zeros(count * num_neg)])
        preds = np.concatenate([layer_pos.numpy(), layer_neg.reshape(-1).numpy()])
        try:
            auroc_by_layer[str(layer)] = float(roc_auc_score(labels, preds))
        except ValueError:
            pass
        count_by_layer[str(layer)] = count

    return GCNEvalResult(
        dataset=checkpoint["data_name"],
        model_type=ExitMode(checkpoint["exit_mode"]).value,
        num_layers=num_layers,
        test_mrr=metrics["MRR"][2],
        test_hits_at_1=metrics["Hits@1"][2],
        test_hits_at_3=metrics["Hits@3"][2],
        test_hits_at_10=metrics["Hits@10"][2],
        test_hits_at_100=metrics["Hits@100"][2],
        exit_distribution=exit_dist,
        active_nodes_per_layer=active_per_layer,
        edge_fraction_by_layer=edge_fracs,
        cumulative_compute_cost=cum_cost,
        total_compute_cost=float(sum(active_per_layer)),
        per_layer_auroc=auroc_by_layer,
        per_layer_edge_count=count_by_layer,
    )


@torch.no_grad()
def compute_edge_scores(
    score_func: torch.nn.Module,
    h: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_scores: list[torch.Tensor] = []
    neg_scores: list[torch.Tensor] = []
    for perm in DataLoader(range(test_pos.size(0)), batch_size):
        pos_edges = test_pos[perm].t()
        neg_edges = torch.permute(test_neg[perm], (2, 0, 1))
        pos_scores.append(score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu())
        neg_scores.append(score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu())
    return torch.cat(pos_scores, dim=0), torch.cat(neg_scores, dim=0)


@torch.no_grad()
def evaluate_gcn_oracle(
    checkpoint_path: Path, data: dict, device: torch.device, oracle_depths: list[int],
) -> dict:
    """Per-edge oracle using first k layers of the L=12 GCN checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bc = checkpoint["backbone_config"]

    score_func = mlp_score(
        bc["hidden_channels"], bc["hidden_channels"], 1,
        checkpoint["num_layers_predictor"], bc["dropout"],
    )
    score_func.load_state_dict(checkpoint["score_func_state_dict"])
    score_func = score_func.to(device)
    score_func.eval()

    x = data["x"].to(device)
    adj = data["adj"].to(device)
    test_pos = data["test_pos"]
    test_neg = data["test_neg"]
    num_edges = test_pos.size(0)

    all_ranks = torch.full((num_edges, len(oracle_depths)), 999999, dtype=torch.long)

    for depth_idx, depth in enumerate(oracle_depths):
        model = load_gcn_at_depth(checkpoint, depth, device)
        model.eval()

        h = model(x, adj)
        pos_scores, neg_scores = compute_edge_scores(score_func, h, test_pos, test_neg)

        pos_flat = pos_scores.flatten()
        neg_2d = neg_scores.squeeze(-1)
        ranks = (neg_2d >= pos_flat[:, None]).sum(dim=1) + 1
        all_ranks[:, depth_idx] = ranks
        print(f"    Depth {depth}: MRR={float((1.0 / ranks.float()).mean()):.4f}")

    oracle_best_depth_idx = all_ranks.argmin(dim=1)
    oracle_ranks = all_ranks[torch.arange(num_edges), oracle_best_depth_idx]

    oracle_pos = torch.zeros(num_edges)
    oracle_neg = torch.zeros(num_edges, test_neg.size(1))

    for depth_idx, depth in enumerate(oracle_depths):
        mask = oracle_best_depth_idx == depth_idx
        if not mask.any():
            continue
        model = load_gcn_at_depth(checkpoint, depth, device)
        model.eval()
        h = model(x, adj)
        pos_scores, neg_scores = compute_edge_scores(score_func, h, test_pos, test_neg)
        oracle_pos[mask] = pos_scores.flatten()[mask]
        oracle_neg[mask] = neg_scores.squeeze(-1)[mask]

    from ogb.linkproppred import Evaluator
    evaluator_mrr = Evaluator(name="ogbl-citation2")
    result_mrr = evaluate_mrr(evaluator_mrr, oracle_pos.unsqueeze(-1), oracle_neg)

    depth_distribution = {
        str(oracle_depths[i]): int((oracle_best_depth_idx == i).sum().item())
        for i in range(len(oracle_depths))
    }
    per_depth_mrr = {
        str(depth): float((1.0 / all_ranks[:, depth_idx].float()).mean())
        for depth_idx, depth in enumerate(oracle_depths)
    }

    return {
        "oracle_mrr": result_mrr["MRR"],
        "oracle_hits_at_1": result_mrr["mrr_hit1"],
        "oracle_hits_at_10": result_mrr["mrr_hit10"],
        "oracle_hits_at_100": result_mrr["mrr_hit100"],
        "depth_distribution": depth_distribution,
        "per_depth_mrr": per_depth_mrr,
        "oracle_depths": oracle_depths,
    }


# ============================================================================
# Plotting
# ============================================================================


def plot_gcn_dataset(dataset: str, l12_data: dict) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("ticks")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "standard",
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.5, "lines.linewidth": 1.0, "lines.markersize": 3,
    })

    COLORS = {
        "none": "#000000", "node_adaptive": "#0072b2",
        "subgraph_adaptive": "#009e73", "oracle": "#d55e00",
    }
    MODE_LABELS = {
        "none": "Baseline", "node_adaptive": "Node-Adaptive (A)",
        "subgraph_adaptive": "Subgraph-Adaptive (B)", "oracle": "Baseline-Oracle",
    }

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.5 * 0.85))
    fig.subplots_adjust(hspace=0.55, wspace=0.40, top=0.90, bottom=0.08, left=0.10, right=0.97)

    baseline = l12_data["none"]
    node_adp = l12_data["node_adaptive"]
    sub_adp = l12_data["subgraph_adaptive"]
    oracle = l12_data["oracle"]
    num_layers = baseline["num_layers"]

    # (a) Edge fraction by layer
    ax = axes[0, 0]
    layers = list(range(num_layers + 1))
    ax.plot(layers, node_adp["edge_fraction_by_layer"],
            color=COLORS["node_adaptive"], label=MODE_LABELS["node_adaptive"], marker="o")
    ax.plot(layers, sub_adp["edge_fraction_by_layer"],
            color=COLORS["subgraph_adaptive"], label=MODE_LABELS["subgraph_adaptive"], marker="s")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of edges resolved")
    ax.set_title("a", fontweight="bold", loc="left")
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.05)

    # (b) Cumulative compute cost
    ax = axes[0, 1]
    num_nodes = sum(baseline["exit_distribution"])
    baseline_cost = [num_nodes * (l + 1) for l in range(num_layers)]
    ax.plot(range(num_layers), baseline_cost,
            color=COLORS["none"], label=MODE_LABELS["none"], linestyle="--")
    ax.plot(range(len(node_adp["cumulative_compute_cost"])), node_adp["cumulative_compute_cost"],
            color=COLORS["node_adaptive"], label=MODE_LABELS["node_adaptive"], marker="o")
    ax.plot(range(len(sub_adp["cumulative_compute_cost"])), sub_adp["cumulative_compute_cost"],
            color=COLORS["subgraph_adaptive"], label=MODE_LABELS["subgraph_adaptive"], marker="s")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative node-layers")
    ax.set_title("b", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    # (c) Metric comparison
    ax = axes[1, 0]
    metric_keys = ["test_mrr", "test_hits_at_1", "test_hits_at_10"]
    oracle_keys = ["oracle_mrr", "oracle_hits_at_1", "oracle_hits_at_10"]
    metric_labels = ["MRR", "Hits@1", "Hits@10"]
    x_pos = np.arange(len(metric_keys))
    bar_width = 0.20

    for i, (model_type, values, label) in enumerate([
        ("none", [baseline[k] for k in metric_keys], MODE_LABELS["none"]),
        ("node_adaptive", [node_adp[k] for k in metric_keys], MODE_LABELS["node_adaptive"]),
        ("subgraph_adaptive", [sub_adp[k] for k in metric_keys], MODE_LABELS["subgraph_adaptive"]),
        ("oracle", [oracle[k] for k in oracle_keys], MODE_LABELS["oracle"]),
    ]):
        ax.bar(x_pos + i * bar_width, values, bar_width, color=COLORS[model_type], label=label)

    ax.set_xticks(x_pos + 1.5 * bar_width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("c", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    # (d) Per-layer AUROC
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
                   color=COLORS[model_type], label=MODE_LABELS[model_type], alpha=0.8, zorder=3)
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


# ============================================================================
# Main pipeline
# ============================================================================


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================================================================
    # Phase 0: Validate GCN baseline matches HeaRT
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 0: VALIDATING GCN BASELINE AGAINST HeaRT")
    print(f"{'='*60}")

    expected_mrr = {"cora": 16.65, "citeseer": 21.02}

    for dataset in ["cora", "citeseer"]:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        hcfg = HEART_CONFIGS[dataset]
        print(f"\n  {dataset}: h={hcfg.hidden_channels}, L={hcfg.num_layers}, "
              f"do={hcfg.dropout}, lr={hcfg.lr}, l2={hcfg.l2}, pred={hcfg.num_layers_predictor}")

        val_mrr, test_mrr, _ = train_gcn(
            dataset=dataset, exit_mode=ExitMode.NONE,
            num_layers=hcfg.num_layers, hidden_channels=hcfg.hidden_channels,
            dropout=hcfg.dropout, lr=hcfg.lr, l2=hcfg.l2, tau0=0.0,
            num_layers_predictor=hcfg.num_layers_predictor,
            data=data, device=device,
        )
        delta = abs(test_mrr * 100 - expected_mrr[dataset])
        status = "PASS" if delta < 3.0 else "WARNING"
        print(f"  → val={val_mrr*100:.2f}%, test={test_mrr*100:.2f}% "
              f"(expected ~{expected_mrr[dataset]}%, Δ={delta:.2f}%) [{status}]")

    # ================================================================
    # Phase 1: Hyperparameter tuning (h=256)
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: HYPERPARAMETER TUNING (h=256)")
    print(f"{'='*60}")

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
                    print(f"    [{i+1}/{len(grid)}] {key} (cached): val={val_mrr:.4f}")
                else:
                    print(f"    [{i+1}/{len(grid)}] {key}: training...", end=" ", flush=True)
                    val_mrr, test_mrr, _ = train_gcn(
                        dataset=dataset, exit_mode=exit_mode,
                        num_layers=nl, hidden_channels=256, dropout=do,
                        lr=lr, l2=0.0, tau0=tau0, num_layers_predictor=3,
                        data=data, device=device,
                        epochs=tune_epochs, kill_cnt_limit=tune_kill_cnt,
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
            print(f"  BEST GCN {exit_mode.value} on {dataset}: L={nl}, do={do}, tau0={tau0}, "
                  f"val={best_val:.4f}, test={test_mrr:.4f}")

    # ================================================================
    # Phase 2: Train L=12 models
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: TRAINING L=12 MODELS")
    print(f"{'='*60}")

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
                dataset=dataset, exit_mode=exit_mode,
                num_layers=12, hidden_channels=256, dropout=do,
                lr=0.001, l2=0.0, tau0=l12_tau0, num_layers_predictor=3,
                data=data, device=device, save_checkpoint=True,
            )

    # ================================================================
    # Phase 3: Evaluate L=12 + oracle
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 3: EVALUATION")
    print(f"{'='*60}")

    all_l12_results: dict[str, dict] = {}

    for dataset in DATASETS:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        all_l12_results[dataset] = {}

        for mode_str in ["none", "node_adaptive", "subgraph_adaptive"]:
            cp = CHECKPOINTS_DIR / f"gcn_{dataset}_{mode_str}_L12.pt"
            print(f"\n  Evaluating GCN {mode_str} L=12 on {dataset}...")
            result = evaluate_gcn_checkpoint(cp, data, device)
            all_l12_results[dataset][mode_str] = asdict(result)
            print(f"    MRR={result.test_mrr*100:.2f}%, Hits@1={result.test_hits_at_1*100:.2f}%, "
                  f"Hits@10={result.test_hits_at_10*100:.2f}%, cost={result.total_compute_cost:.0f}")

        print(f"\n  Computing GCN per-edge oracle on {dataset} (depths={ORACLE_DEPTHS})...")
        cp = CHECKPOINTS_DIR / f"gcn_{dataset}_none_L12.pt"
        oracle_result = evaluate_gcn_oracle(cp, data, device, ORACLE_DEPTHS)
        all_l12_results[dataset]["oracle"] = oracle_result
        print(f"    Oracle MRR={oracle_result['oracle_mrr']*100:.2f}%")
        print(f"    Depth distribution: {oracle_result['depth_distribution']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gcn_l12_comparison.json", "w") as f:
        json.dump(all_l12_results, f, indent=2)

    # ================================================================
    # Phase 4: Plots
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 4: PLOTTING")
    print(f"{'='*60}")

    for dataset in DATASETS:
        try:
            plot_gcn_dataset(dataset, all_l12_results[dataset])
        except Exception as e:
            print(f"  Warning: failed to plot {dataset}: {e}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("SUMMARY: GCN BACKBONE (all L=12)")
    print(f"{'='*60}")

    for dataset in DATASETS:
        print(f"\n{dataset.capitalize()}")
        print(f"{'Model':<30} {'MRR':>8} {'Hits@1':>8} {'Hits@10':>8} {'Cost':>8}")
        print("-" * 72)

        r = all_l12_results[dataset]
        bl = r["none"]
        print(f"{'GCN Baseline (L=12)':<30} {bl['test_mrr']*100:>7.2f}% "
              f"{bl['test_hits_at_1']*100:>7.2f}% "
              f"{bl['test_hits_at_10']*100:>7.2f}% "
              f"{bl['total_compute_cost']:>7.0f}")

        for mode in ["node_adaptive", "subgraph_adaptive"]:
            mr = r[mode]
            red = (1 - mr["total_compute_cost"] / bl["total_compute_cost"]) * 100
            label = {"node_adaptive": "GCN Node-Adpt", "subgraph_adaptive": "GCN Sub-Adpt"}[mode]
            print(f"{label + ' (L=12)':<30} {mr['test_mrr']*100:>7.2f}% "
                  f"{mr['test_hits_at_1']*100:>7.2f}% "
                  f"{mr['test_hits_at_10']*100:>7.2f}% "
                  f"{mr['total_compute_cost']:>7.0f} ({red:.0f}%↓)")

        o = r["oracle"]
        print(f"{'GCN Oracle':<30} {o['oracle_mrr']*100:>7.2f}% "
              f"{o['oracle_hits_at_1']*100:>7.2f}% "
              f"{o['oracle_hits_at_10']*100:>7.2f}% "
              f"{'—':>8}")
        print(f"  Depth distribution: {o['depth_distribution']}")

    # Tuning summary
    tuning_summary: dict[str, dict] = {}
    for dataset in DATASETS:
        tuning_summary[dataset] = {}
        for exit_mode in ExitMode:
            nl, do, lr, tau0, test_mrr = best_configs[dataset][exit_mode.value]
            tuning_summary[dataset][exit_mode.value] = {
                "best_L": nl, "dropout": do, "tau0": tau0, "test_mrr": test_mrr,
            }
    with open(RESULTS_DIR / "gcn_best_configs.json", "w") as f:
        json.dump(tuning_summary, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
