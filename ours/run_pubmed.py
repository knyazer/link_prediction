import json
from dataclasses import asdict
from itertools import product

from shared import (
    CHECKPOINTS_DIR,
    HEART_DATASET_DIR,
    RESULTS_DIR,
    ExitMode,
    HyperConfig,
    config_key,
    load_tuning_cache,
    save_tuning_cache,
)

import torch

from main import read_data
from run_all import train_sas_model

ORACLE_DEPTHS = [1, 2, 4, 8, 12]
DATASET = "pubmed"

LABELS = {
    "node_adaptive": "Node-Adaptive (A)",
    "subgraph_adaptive": "Subgraph-Adaptive (B)",
}


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = DATASET

    print(f"\n{'=' * 60}")
    print(f"RUNNING ALL EXPERIMENTS FOR {dataset.upper()}")
    print(f"{'=' * 60}")

    data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
    cache = load_tuning_cache()

    print("\n--- Phase 1: Hyperparameter tuning ---")

    baseline_grid = [
        HyperConfig(num_layers=nl, dropout=do, lr=0.001, tau0=0.0, epochs=80, kill_cnt=6)
        for nl, do in product([2, 3, 4], [0.0, 0.1, 0.2])
    ]
    exit_grid = [
        HyperConfig(num_layers=nl, dropout=do, lr=0.001, tau0=tau0, epochs=80, kill_cnt=6)
        for nl, do, tau0 in product([4, 8], [0.0, 0.1], [0.0, 0.5, 1.0])
    ]

    best_configs: dict[str, tuple[HyperConfig, float]] = {}

    for exit_mode in ExitMode:
        grid = baseline_grid if exit_mode == ExitMode.NONE else exit_grid

        print(f"\n  Tuning {exit_mode.value} ({len(grid)} configs)")
        best_val = -1.0
        best_cfg: HyperConfig | None = None
        best_test = 0.0

        for i, cfg in enumerate(grid):
            key = config_key(dataset, exit_mode, cfg)
            if key in cache:
                val_mrr, test_mrr = cache[key]["val_mrr"], cache[key]["test_mrr"]
                print(f"    [{i + 1}/{len(grid)}] {key} (cached): val={val_mrr:.4f}")
            else:
                print(f"    [{i + 1}/{len(grid)}] {key}: training...", end=" ", flush=True)
                val_mrr, test_mrr, _ = train_sas_model(dataset, exit_mode, cfg, data, device)
                cache[key] = {"val_mrr": val_mrr, "test_mrr": test_mrr}
                save_tuning_cache(cache)
                print(f"val={val_mrr:.4f}, test={test_mrr:.4f}")

            if val_mrr > best_val:
                best_val = val_mrr
                best_cfg = cfg
                best_test = test_mrr

        assert best_cfg is not None
        best_configs[exit_mode.value] = (best_cfg, best_test)
        print(
            f"  BEST {exit_mode.value}: L={best_cfg.num_layers}, do={best_cfg.dropout}, "
            f"tau0={best_cfg.tau0}, val={best_val:.4f}, test={best_test:.4f}"
        )

    print("\n--- Phase 2: Training L=12 models ---")

    for exit_mode in ExitMode:
        best_cfg, _ = best_configs[exit_mode.value]
        l12_config = HyperConfig(
            num_layers=12,
            dropout=best_cfg.dropout,
            lr=0.001,
            tau0=best_cfg.tau0 if exit_mode != ExitMode.NONE else 0.0,
        )
        print(f"\n  Training {exit_mode.value} L=12 (dropout={l12_config.dropout}, tau0={l12_config.tau0})")
        train_sas_model(dataset, exit_mode, l12_config, data, device, save=True)

    print("\n--- Phase 3: Evaluation ---")

    from evaluate import evaluate_model

    l12_results: dict[str, dict] = {}

    for mode_str in ["none", "node_adaptive", "subgraph_adaptive"]:
        ckpt_path = CHECKPOINTS_DIR / f"{dataset}_{mode_str}_L12.pt"
        print(f"\n  Evaluating {mode_str} L=12...")
        result = evaluate_model(ckpt_path, data, device)
        l12_results[mode_str] = asdict(result)

        filename = f"{dataset}_{mode_str}_L12.json"
        with open(RESULTS_DIR / filename, "w") as f:
            json.dump(asdict(result), f, indent=2)

        print(
            f"    MRR={result.test_mrr:.4f}, Hits@1={result.test_hits_at_1:.4f}, "
            f"Hits@10={result.test_hits_at_10:.4f}, cost={result.total_compute_cost:.0f}"
        )

    print(f"\n  Computing per-edge oracle (depths={ORACLE_DEPTHS})...")
    from eval_oracle import evaluate_oracle

    ckpt_path = CHECKPOINTS_DIR / f"{dataset}_none_L12.pt"
    oracle_result = evaluate_oracle(ckpt_path, data, device, ORACLE_DEPTHS)
    l12_results["oracle"] = oracle_result
    print(
        f"    Oracle MRR={oracle_result['oracle_mrr']:.4f}, "
        f"Hits@1={oracle_result['oracle_hits_at_1']:.4f}, "
        f"Hits@10={oracle_result['oracle_hits_at_10']:.4f}"
    )
    print(f"    Depth distribution: {oracle_result['depth_distribution']}")

    l12_path = RESULTS_DIR / "l12_comparison.json"
    if l12_path.exists():
        with open(l12_path) as f:
            all_l12 = json.load(f)
    else:
        all_l12 = {}
    all_l12[dataset] = l12_results
    with open(l12_path, "w") as f:
        json.dump(all_l12, f, indent=2)

    print("\n--- Phase 4: Regenerating all plots ---")

    from plot import plot_dataset as plot_ds

    for ds in ["cora", "citeseer", "pubmed"]:
        try:
            plot_ds(ds)
        except Exception as e:
            print(f"  Warning: failed to plot {ds}: {e}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {dataset.upper()} (all L=12)")
    print(f"{'=' * 60}")
    print(f"{'Model':<30} {'MRR':>8} {'Hits@1':>8} {'Hits@10':>8} {'Cost':>8}")
    print("-" * 72)

    bl = l12_results["none"]
    print(
        f"{'Baseline (L=12)':<30} {bl['test_mrr'] * 100:>7.2f}% "
        f"{bl['test_hits_at_1'] * 100:>7.2f}% "
        f"{bl['test_hits_at_10'] * 100:>7.2f}% "
        f"{bl['total_compute_cost']:>7.0f}"
    )

    for mode in ["node_adaptive", "subgraph_adaptive"]:
        r = l12_results[mode]
        red = (1 - r["total_compute_cost"] / bl["total_compute_cost"]) * 100
        print(
            f"{LABELS[mode] + ' (L=12)':<30} {r['test_mrr'] * 100:>7.2f}% "
            f"{r['test_hits_at_1'] * 100:>7.2f}% "
            f"{r['test_hits_at_10'] * 100:>7.2f}% "
            f"{r['total_compute_cost']:>7.0f} ({red:.0f}%↓)"
        )

    o = l12_results["oracle"]
    print(
        f"{'Baseline-Oracle':<30} {o['oracle_mrr'] * 100:>7.2f}% "
        f"{o['oracle_hits_at_1'] * 100:>7.2f}% "
        f"{o['oracle_hits_at_10'] * 100:>7.2f}% "
        f"{'—':>8}"
    )
    print(f"\nOracle depth distribution: {o['depth_distribution']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
