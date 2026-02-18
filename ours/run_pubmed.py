"""Run all experiments for pubmed: tune, train, evaluate L=12 + oracle, plot.

Also regenerates plots for cora and citeseer with the updated layout.
"""

import json
import sys
from dataclasses import asdict
from itertools import product
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
TUNING_CACHE = RESULTS_DIR / "tuning_cache.json"

ORACLE_DEPTHS = [1, 2, 4, 8, 12]
DATASET = "pubmed"

LABELS = {
    "node_adaptive": "Node-Adaptive (A)",
    "subgraph_adaptive": "Subgraph-Adaptive (B)",
}


def load_tuning_cache() -> dict:
    if TUNING_CACHE.exists():
        with open(TUNING_CACHE) as f:
            return json.load(f)
    return {}


def save_tuning_cache(cache: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TUNING_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def config_key(dataset: str, exit_mode: ExitMode, num_layers: int, dropout: float, lr: float, tau0: float) -> str:
    return f"{dataset}_{exit_mode.value}_L{num_layers}_d{dropout}_lr{lr}_tau{tau0}"


def tune_single(
    dataset: str, exit_mode: ExitMode,
    num_layers: int, dropout: float, lr: float, tau0: float,
    data: dict, device: torch.device,
) -> tuple[float, float]:
    """Train with given config, return (best_val_mrr, test_mrr_at_best_val)."""
    init_seed(999)

    x = data["x"].to(device)
    train_pos = data["train_pos"].to(device)
    input_channel = x.size(1)

    backbone_config = BackboneConfig(
        in_channels=input_channel, hidden_channels=256,
        num_layers=num_layers, dropout=dropout,
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
        list(model.parameters()) + list(score_func.parameters()), lr=lr,
    )

    from ogb.linkproppred import Evaluator
    evaluator_mrr = Evaluator(name="ogbl-citation2")

    best_val_mrr = 0.0
    test_mrr_at_best_val = 0.0
    kill_cnt = 0

    for epoch in range(1, 81):
        loss = train(model, score_func, train_pos, x, optimizer, 1024)
        if epoch % 5 == 0:
            results = test(model, score_func, data, x, evaluator_mrr, 1024)
            val_mrr = results["MRR"][1]
            test_mrr = results["MRR"][2]
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                test_mrr_at_best_val = test_mrr
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt > 6:
                    break

    return best_val_mrr, test_mrr_at_best_val


def train_and_save(
    dataset: str, exit_mode: ExitMode,
    num_layers: int, dropout: float, tau0: float,
    data: dict, device: torch.device,
) -> Path:
    init_seed(999)

    x = data["x"].to(device)
    train_pos = data["train_pos"].to(device)
    input_channel = x.size(1)

    backbone_config = BackboneConfig(
        in_channels=input_channel, hidden_channels=256,
        num_layers=num_layers, dropout=dropout,
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
        list(model.parameters()) + list(score_func.parameters()), lr=0.001,
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
def evaluate_oracle(
    checkpoint_path: Path, data: dict, device: torch.device, oracle_depths: list[int],
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bc = checkpoint["backbone_config"]

    score_func = mlp_score(bc["hidden_channels"], bc["hidden_channels"], 1, checkpoint["num_layers_predictor"], bc["dropout"])
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
        backbone_config = BackboneConfig(
            in_channels=bc["in_channels"], hidden_channels=bc["hidden_channels"],
            num_layers=depth, dropout=bc["dropout"],
        )
        model = WeightSharedSAS(backbone_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
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

    # Recompute scores at oracle-selected depths for proper OGB evaluation
    oracle_pos = torch.zeros(num_edges)
    oracle_neg = torch.zeros(num_edges, test_neg.size(1))

    for depth_idx, depth in enumerate(oracle_depths):
        mask = oracle_best_depth_idx == depth_idx
        if not mask.any():
            continue
        backbone_config = BackboneConfig(
            in_channels=bc["in_channels"], hidden_channels=bc["hidden_channels"],
            num_layers=depth, dropout=bc["dropout"],
        )
        model = WeightSharedSAS(backbone_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
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
    dataset = DATASET

    print(f"\n{'='*60}")
    print(f"RUNNING ALL EXPERIMENTS FOR {dataset.upper()}")
    print(f"{'='*60}")

    data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
    cache = load_tuning_cache()

    # ================================================================
    # Phase 1: Hyperparameter tuning
    # ================================================================
    print(f"\n--- Phase 1: Hyperparameter tuning ---")

    baseline_grid = list(product([2, 3, 4], [0.0, 0.1, 0.2], [0.001]))
    # Skip L=12 in tuning: too slow on pubmed (75s/epoch × 150 epochs × 6 configs = ~18h per mode)
    # We'll use the best L=8 hyperparameters for L=12 final training instead.
    exit_grid = list(product([4, 8], [0.0, 0.1], [0.0, 0.5, 1.0]))

    best_configs: dict[str, tuple[int, float, float, float, float]] = {}

    for exit_mode in ExitMode:
        grid = [(nl, do, 0.001, 0.0) for nl, do, lr in baseline_grid] if exit_mode == ExitMode.NONE \
            else [(nl, do, 0.001, tau0) for nl, do, tau0 in exit_grid]

        print(f"\n  Tuning {exit_mode.value} ({len(grid)} configs)")
        best_val = -1.0
        best_entry = None

        for i, (nl, do, lr, tau0) in enumerate(grid):
            key = config_key(dataset, exit_mode, nl, do, lr, tau0)
            if key in cache:
                val_mrr, test_mrr = cache[key]["val_mrr"], cache[key]["test_mrr"]
                print(f"    [{i+1}/{len(grid)}] {key} (cached): val={val_mrr:.4f}")
            else:
                print(f"    [{i+1}/{len(grid)}] {key}: training...", end=" ", flush=True)
                val_mrr, test_mrr = tune_single(dataset, exit_mode, nl, do, lr, tau0, data, device)
                cache[key] = {"val_mrr": val_mrr, "test_mrr": test_mrr}
                save_tuning_cache(cache)
                print(f"val={val_mrr:.4f}, test={test_mrr:.4f}")

            if val_mrr > best_val:
                best_val = val_mrr
                best_entry = (nl, do, lr, tau0, test_mrr)

        assert best_entry is not None
        nl, do, lr, tau0, test_mrr = best_entry
        best_configs[exit_mode.value] = best_entry
        print(f"  BEST {exit_mode.value}: L={nl}, do={do}, tau0={tau0}, val={best_val:.4f}, test={test_mrr:.4f}")

    # ================================================================
    # Phase 2: Train final L=12 models for all modes
    # ================================================================
    print(f"\n--- Phase 2: Training L=12 models ---")

    # For L=12, use the best dropout/tau0 from the grid search
    # (pick the best L=12 config, or if no L=12 was best, use the best config's dropout/tau0)
    for exit_mode in ExitMode:
        mode_str = exit_mode.value

        # Find best L=12 config from cache
        best_l12_val = -1.0
        best_l12_do = 0.0
        best_l12_tau0 = 0.0

        # Use best config's dropout/tau0 for L=12 (L=12 tuning skipped for pubmed)
        nl, do, lr, tau0, _ = best_configs[mode_str]
        best_l12_do = do
        best_l12_tau0 = tau0 if exit_mode != ExitMode.NONE else 0.0

        print(f"\n  Training {mode_str} L=12 (dropout={best_l12_do}, tau0={best_l12_tau0})")
        train_and_save(dataset, exit_mode, 12, best_l12_do, best_l12_tau0, data, device)

    # ================================================================
    # Phase 3: Evaluate all L=12 + oracle
    # ================================================================
    print(f"\n--- Phase 3: Evaluation ---")

    from evaluate import evaluate_model

    l12_results: dict[str, dict] = {}

    for mode_str in ["none", "node_adaptive", "subgraph_adaptive"]:
        ckpt_path = CHECKPOINTS_DIR / f"{dataset}_{mode_str}_L12.pt"
        print(f"\n  Evaluating {mode_str} L=12...")
        result = evaluate_model(ckpt_path, data, device)
        l12_results[mode_str] = asdict(result)

        # Also save as individual result JSON
        filename = f"{dataset}_{mode_str}_L12.json"
        with open(RESULTS_DIR / filename, "w") as f:
            json.dump(asdict(result), f, indent=2)

        print(f"    MRR={result.test_mrr:.4f}, Hits@1={result.test_hits_at_1:.4f}, "
              f"Hits@10={result.test_hits_at_10:.4f}, cost={result.total_compute_cost:.0f}")

    # Oracle
    print(f"\n  Computing per-edge oracle (depths={ORACLE_DEPTHS})...")
    ckpt_path = CHECKPOINTS_DIR / f"{dataset}_none_L12.pt"
    oracle_result = evaluate_oracle(ckpt_path, data, device, ORACLE_DEPTHS)
    l12_results["oracle"] = oracle_result
    print(f"    Oracle MRR={oracle_result['oracle_mrr']:.4f}, "
          f"Hits@1={oracle_result['oracle_hits_at_1']:.4f}, "
          f"Hits@10={oracle_result['oracle_hits_at_10']:.4f}")
    print(f"    Depth distribution: {oracle_result['depth_distribution']}")

    # Update l12_comparison.json with pubmed
    l12_path = RESULTS_DIR / "l12_comparison.json"
    if l12_path.exists():
        with open(l12_path) as f:
            all_l12 = json.load(f)
    else:
        all_l12 = {}
    all_l12[dataset] = l12_results
    with open(l12_path, "w") as f:
        json.dump(all_l12, f, indent=2)

    # ================================================================
    # Phase 4: Regenerate ALL plots
    # ================================================================
    print(f"\n--- Phase 4: Regenerating all plots ---")

    from plot import plot_dataset as plot_ds
    for ds in ["cora", "citeseer", "pubmed"]:
        try:
            plot_ds(ds)
        except Exception as e:
            print(f"  Warning: failed to plot {ds}: {e}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset.upper()} (all L=12)")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'MRR':>8} {'Hits@1':>8} {'Hits@10':>8} {'Cost':>8}")
    print("-" * 72)

    bl = l12_results["none"]
    print(f"{'Baseline (L=12)':<30} {bl['test_mrr']*100:>7.2f}% "
          f"{bl['test_hits_at_1']*100:>7.2f}% "
          f"{bl['test_hits_at_10']*100:>7.2f}% "
          f"{bl['total_compute_cost']:>7.0f}")

    for mode in ["node_adaptive", "subgraph_adaptive"]:
        r = l12_results[mode]
        red = (1 - r["total_compute_cost"] / bl["total_compute_cost"]) * 100
        print(f"{LABELS[mode] + ' (L=12)':<30} {r['test_mrr']*100:>7.2f}% "
              f"{r['test_hits_at_1']*100:>7.2f}% "
              f"{r['test_hits_at_10']*100:>7.2f}% "
              f"{r['total_compute_cost']:>7.0f} ({red:.0f}%↓)")

    o = l12_results["oracle"]
    print(f"{'Baseline-Oracle':<30} {o['oracle_mrr']*100:>7.2f}% "
          f"{o['oracle_hits_at_1']*100:>7.2f}% "
          f"{o['oracle_hits_at_10']*100:>7.2f}% "
          f"{'—':>8}")
    print(f"\nOracle depth distribution: {o['depth_distribution']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
