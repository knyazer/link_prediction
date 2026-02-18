"""Evaluate all models at L=12 + per-edge oracle for the baseline.

Models evaluated:
- Baseline (L=12): no exit, all nodes process all 12 layers
- Node-adaptive (L=12): per-node early exit
- Subgraph-adaptive (L=12): two-stage neighborhood-aware exit
- Baseline-oracle: baseline L=12 weights, but for each test edge, pick
  the depth L in {1,2,4,8,12} that gives the best rank (per-edge cheating)
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

OURS_DIR = Path(__file__).resolve().parent
HEART_BENCHMARKING_DIR = OURS_DIR.parent / "HeaRT" / "benchmarking"
sys.path.insert(0, str(OURS_DIR))
sys.path.insert(0, str(HEART_BENCHMARKING_DIR))

import torch
from torch.utils.data import DataLoader

from evalutors import evaluate_mrr
from main import ExitMode, read_data, train, test
from model import BackboneConfig, ExitConfig, NodeAdaptiveExit, SubgraphAdaptiveExit, WeightSharedSAS
from scoring import mlp_score
from utils import init_seed

HEART_DATASET_DIR = HEART_BENCHMARKING_DIR.parent / "dataset"
CHECKPOINTS_DIR = OURS_DIR.parent / "checkpoints"
RESULTS_DIR = OURS_DIR.parent / "results"

ORACLE_DEPTHS = [1, 2, 4, 8, 12]


def train_and_save(
    dataset: str,
    exit_mode: ExitMode,
    num_layers: int,
    dropout: float,
    tau0: float,
    data: dict,
    device: torch.device,
) -> Path:
    """Train a model and save checkpoint. Returns checkpoint path."""
    init_seed(999)

    x = data["x"].to(device)
    train_pos = data["train_pos"].to(device)
    input_channel = x.size(1)

    backbone_config = BackboneConfig(
        in_channels=input_channel,
        hidden_channels=256,
        num_layers=num_layers,
        dropout=dropout,
    )

    match exit_mode:
        case ExitMode.NONE:
            model = WeightSharedSAS(backbone_config).to(device)
        case ExitMode.NODE_ADAPTIVE:
            exit_config = ExitConfig(tau0=tau0, confidence_hidden_dim=32)
            model = NodeAdaptiveExit(backbone_config, exit_config).to(device)
        case ExitMode.SUBGRAPH_ADAPTIVE:
            exit_config = ExitConfig(tau0=tau0, confidence_hidden_dim=32)
            model = SubgraphAdaptiveExit(backbone_config, exit_config).to(device)

    score_func = mlp_score(256, 256, 1, 3, dropout).to(device)

    model.reset_parameters()
    score_func.reset_parameters()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(score_func.parameters()),
        lr=0.001,
    )

    from ogb.linkproppred import Evaluator
    evaluator_mrr = Evaluator(name="ogbl-citation2")

    best_valid = 0.0
    kill_cnt = 0
    best_model_state: dict | None = None
    best_score_state: dict | None = None
    best_epoch = 0

    for epoch in range(1, 151):
        loss = train(model, score_func, train_pos, x, optimizer, 1024)

        if epoch % 5 == 0:
            results = test(model, score_func, data, x, evaluator_mrr, 1024)
            val_mrr = results["MRR"][1]

            if val_mrr > best_valid:
                best_valid = val_mrr
                kill_cnt = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_score_state = {k: v.cpu().clone() for k, v in score_func.state_dict().items()}
                best_epoch = epoch
            else:
                kill_cnt += 1
                if kill_cnt > 10:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    assert best_model_state is not None
    assert best_score_state is not None

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINTS_DIR / f"{dataset}_{exit_mode.value}_L{num_layers}.pt"

    torch.save({
        "model_state_dict": best_model_state,
        "score_func_state_dict": best_score_state,
        "backbone_config": {
            "in_channels": input_channel,
            "hidden_channels": 256,
            "num_layers": num_layers,
            "dropout": dropout,
        },
        "exit_mode": exit_mode.value,
        "data_name": dataset,
        "tau0": tau0,
        "confidence_hidden_dim": 32,
        "num_layers_predictor": 3,
        "best_epoch": best_epoch,
    }, save_path)

    print(f"  Saved {save_path.name} (epoch {best_epoch}, val MRR: {best_valid:.4f})")
    return save_path


def checkpoint_is_valid(path: Path) -> bool:
    """Check if checkpoint exists with hidden_channels=256."""
    if not path.exists():
        return False
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["backbone_config"]["hidden_channels"] == 256


@torch.no_grad()
def compute_edge_scores(
    score_func: torch.nn.Module,
    h: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-edge positive and negative scores."""
    pos_scores: list[torch.Tensor] = []
    neg_scores: list[torch.Tensor] = []

    for perm in DataLoader(range(test_pos.size(0)), batch_size):
        pos_edges = test_pos[perm].t()
        neg_edges = torch.permute(test_neg[perm], (2, 0, 1))
        pos_scores.append(score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu())
        neg_scores.append(score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu())

    return torch.cat(pos_scores, dim=0), torch.cat(neg_scores, dim=0)


@torch.no_grad()
def compute_per_edge_ranks(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """Compute rank of each positive score among its negatives.

    Returns tensor of shape (num_edges,) with 1-indexed ranks.
    """
    pos_flat = pos_scores.flatten()
    neg_2d = neg_scores.squeeze(-1)
    ranks = (neg_2d >= pos_flat[:, None]).sum(dim=1) + 1
    return ranks


@torch.no_grad()
def evaluate_oracle(
    checkpoint_path: Path,
    data: dict,
    device: torch.device,
    oracle_depths: list[int],
) -> dict:
    """Per-edge oracle: for each test edge, pick depth with best rank."""
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
    all_pos_scores = torch.zeros(num_edges, len(oracle_depths))
    all_neg_scores = torch.zeros(num_edges, len(oracle_depths), test_neg.size(1))

    for depth_idx, depth in enumerate(oracle_depths):
        backbone_config = BackboneConfig(
            in_channels=bc["in_channels"],
            hidden_channels=bc["hidden_channels"],
            num_layers=depth,
            dropout=bc["dropout"],
        )
        model = WeightSharedSAS(backbone_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        h = model(x, adj)
        pos_scores, neg_scores = compute_edge_scores(score_func, h, test_pos, test_neg)
        ranks = compute_per_edge_ranks(pos_scores, neg_scores)

        all_ranks[:, depth_idx] = ranks
        all_pos_scores[:, depth_idx] = pos_scores.flatten()
        all_neg_scores[:, depth_idx] = neg_scores.squeeze(-1)

        print(f"    Depth {depth}: MRR={float((1.0 / ranks.float()).mean()):.4f}")

    oracle_best_depth_idx = all_ranks.argmin(dim=1)
    oracle_ranks = all_ranks[torch.arange(num_edges), oracle_best_depth_idx]

    oracle_mrr = float((1.0 / oracle_ranks.float()).mean())

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
        str(oracle_depths[i]): int((oracle_best_depth_idx == i).sum().item())
        for i in range(len(oracle_depths))
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Best hyperparams for L=12 from tuning cache
    l12_configs: dict[str, dict[str, tuple[float, float]]] = {
        "cora": {
            "none": (0.0, 0.0),        # (dropout, tau0)
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
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        all_results[dataset] = {}

        # Train missing L=12 checkpoints
        for mode_str, exit_mode in [
            ("none", ExitMode.NONE),
            ("node_adaptive", ExitMode.NODE_ADAPTIVE),
            ("subgraph_adaptive", ExitMode.SUBGRAPH_ADAPTIVE),
        ]:
            ckpt_path = CHECKPOINTS_DIR / f"{dataset}_{mode_str}_L12.pt"
            if not checkpoint_is_valid(ckpt_path):
                dropout, tau0 = l12_configs[dataset][mode_str]
                print(f"\nTraining {mode_str} L=12 (dropout={dropout}, tau0={tau0})...")
                train_and_save(dataset, exit_mode, 12, dropout, tau0, data, device)
            else:
                print(f"\n{mode_str} L=12 checkpoint exists (hidden=256)")

        # Evaluate all L=12 models
        from evaluate import evaluate_model
        for mode_str in ["none", "node_adaptive", "subgraph_adaptive"]:
            ckpt_path = CHECKPOINTS_DIR / f"{dataset}_{mode_str}_L12.pt"
            print(f"\nEvaluating {mode_str} L=12...")
            result = evaluate_model(ckpt_path, data, device)
            all_results[dataset][mode_str] = asdict(result)
            print(f"  MRR={result.test_mrr:.4f}, Hits@1={result.test_hits_at_1:.4f}, "
                  f"Hits@10={result.test_hits_at_10:.4f}, cost={result.total_compute_cost:.0f}")

        # Per-edge oracle for baseline
        print(f"\nComputing per-edge oracle (depths={ORACLE_DEPTHS})...")
        ckpt_path = CHECKPOINTS_DIR / f"{dataset}_none_L12.pt"
        oracle_result = evaluate_oracle(ckpt_path, data, device, ORACLE_DEPTHS)
        all_results[dataset]["oracle"] = oracle_result
        print(f"  Oracle MRR={oracle_result['oracle_mrr']:.4f}, "
              f"Hits@1={oracle_result['oracle_hits_at_1']:.4f}, "
              f"Hits@10={oracle_result['oracle_hits_at_10']:.4f}")
        print(f"  Depth distribution: {oracle_result['depth_distribution']}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "l12_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY (all L=12)")
    print(f"{'='*60}")
    for dataset in ["cora", "citeseer"]:
        print(f"\n{dataset.capitalize()}")
        print(f"{'Model':<30} {'MRR':>8} {'Hits@1':>8} {'Hits@10':>8} {'Cost':>8}")
        print("-" * 72)

        r = all_results[dataset]

        baseline = r["none"]
        print(f"{'Baseline (L=12)':<30} {baseline['test_mrr']*100:>7.2f}% "
              f"{baseline['test_hits_at_1']*100:>7.2f}% "
              f"{baseline['test_hits_at_10']*100:>7.2f}% "
              f"{baseline['total_compute_cost']:>7.0f}")

        node = r["node_adaptive"]
        reduction = (1 - node["total_compute_cost"] / baseline["total_compute_cost"]) * 100
        print(f"{'Node-Adaptive (L=12)':<30} {node['test_mrr']*100:>7.2f}% "
              f"{node['test_hits_at_1']*100:>7.2f}% "
              f"{node['test_hits_at_10']*100:>7.2f}% "
              f"{node['total_compute_cost']:>7.0f} ({reduction:.0f}%↓)")

        sub = r["subgraph_adaptive"]
        reduction = (1 - sub["total_compute_cost"] / baseline["total_compute_cost"]) * 100
        print(f"{'Subgraph-Adaptive (L=12)':<30} {sub['test_mrr']*100:>7.2f}% "
              f"{sub['test_hits_at_1']*100:>7.2f}% "
              f"{sub['test_hits_at_10']*100:>7.2f}% "
              f"{sub['total_compute_cost']:>7.0f} ({reduction:.0f}%↓)")

        oracle = r["oracle"]
        print(f"{'Baseline-Oracle':<30} {oracle['oracle_mrr']*100:>7.2f}% "
              f"{oracle['oracle_hits_at_1']*100:>7.2f}% "
              f"{oracle['oracle_hits_at_10']*100:>7.2f}% "
              f"{'—':>8}")
        print(f"  Oracle depth distribution: {oracle['depth_distribution']}")
        print(f"  Per-depth MRR: {', '.join(f'L={d}={v*100:.2f}%' for d,v in zip(ORACLE_DEPTHS, [oracle['per_depth_mrr'][str(d)] for d in ORACLE_DEPTHS]))}")


if __name__ == "__main__":
    main()
