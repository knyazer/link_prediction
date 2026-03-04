import json
from dataclasses import asdict, dataclass
from pathlib import Path

from shared import (
    CHECKPOINTS_DIR,
    HEART_DATASET_DIR,
    RESULTS_DIR,
    ExitMode,
    build_sas_model,
)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from main import get_metric_score, read_data, test_edge
from model import BackboneConfig, ExitConfig, ForwardResult
from scoring import mlp_score
from utils import init_seed


@dataclass
class EvalResult:
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
    active_edges_per_layer: list[int]
    flops_per_layer: list[float]
    edge_fraction_by_layer: list[float]
    cumulative_compute_cost: list[float]
    total_compute_cost: float
    per_layer_auroc: dict[str, float]
    per_layer_edge_count: dict[str, int]


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    override_num_layers: int | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    bc = checkpoint["backbone_config"]
    num_layers = override_num_layers if override_num_layers is not None else bc["num_layers"]
    backbone_config = BackboneConfig(
        in_channels=bc["in_channels"],
        hidden_channels=bc["hidden_channels"],
        num_layers=num_layers,
        dropout=bc["dropout"],
    )

    exit_mode = ExitMode(checkpoint["exit_mode"])
    exit_config = (
        ExitConfig(
            tau0=checkpoint["tau0"],
            confidence_hidden_dim=checkpoint["confidence_hidden_dim"],
        )
        if exit_mode != ExitMode.NONE
        else None
    )

    backbone_type = checkpoint.get("backbone_type")
    if backbone_type == "gcn":
        from model import GCNBackbone, GCNNodeAdaptiveExit, GCNSubgraphAdaptiveExit

        match exit_mode:
            case ExitMode.NONE:
                model = GCNBackbone(backbone_config)
            case ExitMode.NODE_ADAPTIVE:
                model = GCNNodeAdaptiveExit(backbone_config, exit_config)
            case ExitMode.SUBGRAPH_ADAPTIVE:
                model = GCNSubgraphAdaptiveExit(backbone_config, exit_config)
    elif backbone_type == "residual_gcn":
        from model import GCNResidualBackbone, GCNResidualNodeAdaptiveExit, GCNResidualSubgraphAdaptiveExit

        match exit_mode:
            case ExitMode.NONE:
                model = GCNResidualBackbone(backbone_config)
            case ExitMode.NODE_ADAPTIVE:
                model = GCNResidualNodeAdaptiveExit(backbone_config, exit_config)
            case ExitMode.SUBGRAPH_ADAPTIVE:
                model = GCNResidualSubgraphAdaptiveExit(backbone_config, exit_config)
    else:
        model = build_sas_model(backbone_config, exit_mode, exit_config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    score_func = mlp_score(
        bc["hidden_channels"],
        bc["hidden_channels"],
        1,
        checkpoint["num_layers_predictor"],
        bc["dropout"],
    )
    score_func.load_state_dict(checkpoint["score_func_state_dict"])
    score_func = score_func.to(device)

    return model, score_func, checkpoint


def compute_per_layer_auroc(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    resolution_layers: torch.Tensor,
    num_layers: int,
) -> tuple[dict[str, float], dict[str, int]]:
    pos_flat = pos_scores.cpu().flatten()
    neg_2d = neg_scores.cpu().squeeze(-1)

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

    return auroc_by_layer, count_by_layer


@torch.no_grad()
def evaluate_model(
    checkpoint_path: Path,
    data: dict,
    device: torch.device,
    override_num_layers: int | None = None,
) -> EvalResult:
    init_seed(999)

    model, score_func, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        device,
        override_num_layers,
    )
    model.eval()
    score_func.eval()

    x = data["x"].to(device)
    adj = data["adj"].to(device)
    bc = checkpoint["backbone_config"]
    num_layers = override_num_layers if override_num_layers is not None else bc["num_layers"]
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
        score_func,
        data["valid_pos"],
        h,
        batch_size,
        data["valid_neg"],
    )
    pos_test_pred, neg_test_pred = test_edge(
        score_func,
        data["test_pos"],
        h,
        batch_size,
        data["test_neg"],
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

    row, col, _ = adj.coo()
    col_cpu = col.cpu()
    hidden = bc["hidden_channels"]

    active_edges_per_layer: list[int] = []
    for layer in range(num_layers):
        active_mask = exit_layers_cpu > layer
        active_edges_per_layer.append(int(active_mask[col_cpu].sum().item()))

    flops_per_layer = [an * hidden * hidden + ae * hidden for an, ae in zip(active_per_layer, active_edges_per_layer)]

    src_exit = exit_layers_cpu[test_pos[:, 0]]
    dst_exit = exit_layers_cpu[test_pos[:, 1]]
    resolution_layers = torch.maximum(src_exit, dst_exit)

    total_edges = resolution_layers.size(0)
    edge_fracs = [float((resolution_layers <= layer).sum().item()) / total_edges for layer in range(num_layers + 1)]

    cum_cost: list[float] = []
    running = 0.0
    for count in active_per_layer:
        running += count
        cum_cost.append(running)

    exit_dist = [int((exit_layers == layer).sum().item()) for layer in range(num_layers + 1)]

    auroc_by_layer, count_by_layer = compute_per_layer_auroc(
        pos_test_pred,
        neg_test_pred,
        resolution_layers,
        num_layers,
    )

    return EvalResult(
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
        active_edges_per_layer=active_edges_per_layer,
        flops_per_layer=flops_per_layer,
        edge_fraction_by_layer=edge_fracs,
        cumulative_compute_cost=cum_cost,
        total_compute_cost=float(sum(active_per_layer)),
        per_layer_auroc=auroc_by_layer,
        per_layer_edge_count=count_by_layer,
    )


def find_compute_balanced_L(
    checkpoint_path: Path,
    data: dict,
    device: torch.device,
    target_cost: float,
) -> int:
    x = data["x"].to(device)
    adj = data["adj"].to(device)

    best_l = 1
    best_diff = float("inf")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    max_layers = checkpoint["backbone_config"]["num_layers"]

    for num_layers in range(1, max_layers + 1):
        init_seed(999)
        model, _, _ = load_model_from_checkpoint(checkpoint_path, device, override_num_layers=num_layers)
        model.eval()

        with torch.no_grad():
            out = model(x, adj)

        if isinstance(out, ForwardResult):
            cost = float(sum(out.active_nodes_per_layer))
        else:
            cost = float(x.size(0) * num_layers)

        diff = abs(cost - target_cost)
        if diff < best_diff:
            best_diff = diff
            best_l = num_layers

    return best_l


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    l12_path = RESULTS_DIR / "l12_comparison.json"
    if l12_path.exists():
        with open(l12_path) as f:
            all_l12 = json.load(f)
    else:
        all_l12 = {}

    for dataset in ["cora", "citeseer", "pubmed"]:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")

        for exit_mode in ExitMode:
            ckpt = CHECKPOINTS_DIR / f"{dataset}_{exit_mode.value}_L12.pt"
            if not ckpt.exists():
                print(f"  Skipping {exit_mode.value} on {dataset} (no checkpoint)")
                continue
            print(f"Evaluating {exit_mode.value} on {dataset}...")
            result = evaluate_model(ckpt, data, device)
            print(f"  MRR={result.test_mrr:.4f}, cost={result.total_compute_cost:.0f}")

            all_l12.setdefault(dataset, {})[exit_mode.value] = asdict(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(l12_path, "w") as f:
        json.dump(all_l12, f, indent=2)

    print(f"\nResults saved to {l12_path}")


if __name__ == "__main__":
    main()
