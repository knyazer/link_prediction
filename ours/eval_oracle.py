import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from shared import (
    CHECKPOINTS_DIR,
    HEART_DATASET_DIR,
    RESULTS_DIR,
    ExitMode,
    HyperConfig,
    compute_edge_scores,
)

import torch

from evalutors import evaluate_mrr
from main import read_data
from model import BackboneConfig, WeightSharedSAS
from scoring import mlp_score

ORACLE_DEPTHS = [1, 2, 4, 8, 12]


def _load_sas_at_depth(checkpoint: dict, depth: int, device: torch.device) -> torch.nn.Module:
    bc = checkpoint["backbone_config"]
    backbone_config = BackboneConfig(
        in_channels=bc["in_channels"],
        hidden_channels=bc["hidden_channels"],
        num_layers=depth,
        dropout=bc["dropout"],
    )
    model = WeightSharedSAS(backbone_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def checkpoint_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["backbone_config"]["hidden_channels"] == 256


@torch.no_grad()
def evaluate_oracle(
    checkpoint_path: Path,
    data: dict,
    device: torch.device,
    oracle_depths: list[int],
    model_loader: Callable[[dict, int, torch.device], torch.nn.Module] | None = None,
) -> dict:
    if model_loader is None:
        model_loader = _load_sas_at_depth

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bc = checkpoint["backbone_config"]

    score_func = mlp_score(
        bc["hidden_channels"],
        bc["hidden_channels"],
        1,
        checkpoint["num_layers_predictor"],
        bc["dropout"],
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
    all_pos_scores = torch.zeros(num_edges, len(oracle_depths))
    all_neg_scores = torch.zeros(num_edges, len(oracle_depths), test_neg.size(1))

    for depth_idx, depth in enumerate(oracle_depths):
        model = model_loader(checkpoint, depth, device)
        model.eval()

        h = model(x, adj)
        pos_scores, neg_scores = compute_edge_scores(score_func, h, test_pos, test_neg)

        pos_flat = pos_scores.flatten()
        neg_2d = neg_scores.squeeze(-1)
        ranks = (neg_2d >= pos_flat[:, None]).sum(dim=1) + 1

        all_ranks[:, depth_idx] = ranks
        all_pos_scores[:, depth_idx] = pos_flat
        all_neg_scores[:, depth_idx] = neg_2d

        print(f"    Depth {depth}: MRR={float((1.0 / ranks.float()).mean()):.4f}")

    oracle_best_depth_idx = all_ranks.argmin(dim=1)

    oracle_pos = all_pos_scores[torch.arange(num_edges), oracle_best_depth_idx]
    oracle_neg = all_neg_scores[torch.arange(num_edges), oracle_best_depth_idx]

    from ogb.linkproppred import Evaluator

    evaluator_mrr = Evaluator(name="ogbl-citation2")

    result_mrr = evaluate_mrr(
        evaluator_mrr,
        oracle_pos.unsqueeze(-1),
        oracle_neg,
    )

    depth_distribution = {
        str(oracle_depths[i]): int((oracle_best_depth_idx == i).sum().item()) for i in range(len(oracle_depths))
    }

    per_depth_mrr = {}
    for depth_idx, depth in enumerate(oracle_depths):
        per_depth_mrr[str(depth)] = float((1.0 / all_ranks[:, depth_idx].float()).mean())

    return {
        "oracle_mrr": result_mrr["MRR"],
        "oracle_hits_at_1": result_mrr["mrr_hit1"],
        "oracle_hits_at_10": result_mrr["mrr_hit10"],
        "oracle_hits_at_100": result_mrr["mrr_hit100"],
        "depth_distribution": depth_distribution,
        "per_depth_mrr": per_depth_mrr,
        "oracle_depths": oracle_depths,
    }


def main() -> None:
    from evaluate import evaluate_model
    from run_all import train_sas_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    l12_configs: dict[str, dict[str, tuple[float, float]]] = {
        "cora": {
            "none": (0.0, 0.0),
            "node_adaptive": (0.0, 0.5),
            "subgraph_adaptive": (0.0, 1.0),
        },
        "citeseer": {
            "none": (0.1, 0.0),
            "node_adaptive": (0.1, 0.0),
            "subgraph_adaptive": (0.1, 0.0),
        },
    }

    all_results: dict[str, dict] = {}

    for dataset in ["cora", "citeseer"]:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 60}")

        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        all_results[dataset] = {}

        for mode_str, exit_mode in [
            ("none", ExitMode.NONE),
            ("node_adaptive", ExitMode.NODE_ADAPTIVE),
            ("subgraph_adaptive", ExitMode.SUBGRAPH_ADAPTIVE),
        ]:
            ckpt_path = CHECKPOINTS_DIR / f"{dataset}_{mode_str}_L12.pt"
            if not checkpoint_is_valid(ckpt_path):
                dropout, tau0 = l12_configs[dataset][mode_str]
                print(f"\nTraining {mode_str} L=12 (dropout={dropout}, tau0={tau0})...")
                config = HyperConfig(num_layers=12, dropout=dropout, lr=0.001, tau0=tau0)
                train_sas_model(dataset, exit_mode, config, data, device, save=True)
            else:
                print(f"\n{mode_str} L=12 checkpoint exists (hidden=256)")

        for mode_str in ["none", "node_adaptive", "subgraph_adaptive"]:
            ckpt_path = CHECKPOINTS_DIR / f"{dataset}_{mode_str}_L12.pt"
            print(f"\nEvaluating {mode_str} L=12...")
            result = evaluate_model(ckpt_path, data, device)
            all_results[dataset][mode_str] = asdict(result)
            print(
                f"  MRR={result.test_mrr:.4f}, Hits@1={result.test_hits_at_1:.4f}, "
                f"Hits@10={result.test_hits_at_10:.4f}, cost={result.total_compute_cost:.0f}"
            )

        print(f"\nComputing per-edge oracle (depths={ORACLE_DEPTHS})...")
        ckpt_path = CHECKPOINTS_DIR / f"{dataset}_none_L12.pt"
        oracle_result = evaluate_oracle(ckpt_path, data, device, ORACLE_DEPTHS)
        all_results[dataset]["oracle"] = oracle_result
        print(
            f"  Oracle MRR={oracle_result['oracle_mrr']:.4f}, "
            f"Hits@1={oracle_result['oracle_hits_at_1']:.4f}, "
            f"Hits@10={oracle_result['oracle_hits_at_10']:.4f}"
        )
        print(f"  Depth distribution: {oracle_result['depth_distribution']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "l12_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SUMMARY (all L=12)")
    print(f"{'=' * 60}")
    for dataset in ["cora", "citeseer"]:
        print(f"\n{dataset.capitalize()}")
        print(f"{'Model':<30} {'MRR':>8} {'Hits@1':>8} {'Hits@10':>8} {'Cost':>8}")
        print("-" * 72)

        r = all_results[dataset]

        baseline = r["none"]
        print(
            f"{'Baseline (L=12)':<30} {baseline['test_mrr'] * 100:>7.2f}% "
            f"{baseline['test_hits_at_1'] * 100:>7.2f}% "
            f"{baseline['test_hits_at_10'] * 100:>7.2f}% "
            f"{baseline['total_compute_cost']:>7.0f}"
        )

        node = r["node_adaptive"]
        reduction = (1 - node["total_compute_cost"] / baseline["total_compute_cost"]) * 100
        print(
            f"{'Node-Adaptive (L=12)':<30} {node['test_mrr'] * 100:>7.2f}% "
            f"{node['test_hits_at_1'] * 100:>7.2f}% "
            f"{node['test_hits_at_10'] * 100:>7.2f}% "
            f"{node['total_compute_cost']:>7.0f} ({reduction:.0f}%↓)"
        )

        sub = r["subgraph_adaptive"]
        reduction = (1 - sub["total_compute_cost"] / baseline["total_compute_cost"]) * 100
        print(
            f"{'Subgraph-Adaptive (L=12)':<30} {sub['test_mrr'] * 100:>7.2f}% "
            f"{sub['test_hits_at_1'] * 100:>7.2f}% "
            f"{sub['test_hits_at_10'] * 100:>7.2f}% "
            f"{sub['total_compute_cost']:>7.0f} ({reduction:.0f}%↓)"
        )

        oracle = r["oracle"]
        print(
            f"{'Baseline-Oracle':<30} {oracle['oracle_mrr'] * 100:>7.2f}% "
            f"{oracle['oracle_hits_at_1'] * 100:>7.2f}% "
            f"{oracle['oracle_hits_at_10'] * 100:>7.2f}% "
            f"{'—':>8}"
        )
        print(f"  Oracle depth distribution: {oracle['depth_distribution']}")
        print(
            f"  Per-depth MRR: {', '.join(f'L={d}={v * 100:.2f}%' for d, v in zip(ORACLE_DEPTHS, [oracle['per_depth_mrr'][str(d)] for d in ORACLE_DEPTHS]))}"
        )


if __name__ == "__main__":
    main()
