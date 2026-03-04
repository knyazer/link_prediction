import json
from itertools import product
from pathlib import Path

from shared import (
    CHECKPOINTS_DIR,
    HEART_DATASET_DIR,
    RESULTS_DIR,
    ExitMode,
    HyperConfig,
    build_sas_model,
    config_key,
    load_tuning_cache,
    save_checkpoint,
    save_tuning_cache,
)

import torch

from main import read_data, train, test
from model import BackboneConfig, ExitConfig
from scoring import mlp_score
from utils import init_seed


def train_sas_model(
    dataset: str,
    exit_mode: ExitMode,
    config: HyperConfig,
    data: dict,
    device: torch.device,
    save: bool = False,
    l2: float = 0.0,
) -> tuple[float, float, Path | None]:
    init_seed(config.seed)

    x = data["x"].to(device)
    train_pos = data["train_pos"].to(device)
    input_channel = x.size(1)

    backbone_config = BackboneConfig(
        in_channels=input_channel,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    exit_config = (
        ExitConfig(tau0=config.tau0, confidence_hidden_dim=config.confidence_hidden_dim)
        if exit_mode != ExitMode.NONE
        else None
    )
    model = build_sas_model(backbone_config, exit_mode, exit_config).to(device)

    score_func = mlp_score(
        config.hidden_channels,
        config.hidden_channels,
        1,
        config.num_layers_predictor,
        config.dropout,
    ).to(device)

    model.reset_parameters()
    score_func.reset_parameters()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(score_func.parameters()),
        lr=config.lr,
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

    for epoch in range(1, 1 + config.epochs):
        train(model, score_func, train_pos, x, optimizer, config.batch_size)

        if epoch % config.eval_steps == 0:
            results = test(model, score_func, data, x, evaluator_mrr, config.batch_size)
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
                if kill_cnt > config.kill_cnt:
                    if save:
                        print(f"  Early stopping at epoch {epoch}")
                    break

    save_path = None
    if save and best_model_state is not None and best_score_state is not None:
        save_path = CHECKPOINTS_DIR / f"{dataset}_{exit_mode.value}_L{config.num_layers}.pt"
        save_checkpoint(
            save_path=save_path,
            model_state=best_model_state,
            score_state=best_score_state,
            backbone_config=backbone_config,
            exit_mode=exit_mode,
            dataset=dataset,
            tau0=config.tau0,
            confidence_hidden_dim=config.confidence_hidden_dim,
            num_layers_predictor=config.num_layers_predictor,
            best_epoch=best_epoch,
        )
        print(f"  Saved {save_path.name} (epoch {best_epoch}, val MRR: {best_val_mrr:.4f})")

    return best_val_mrr, test_mrr_at_best, save_path


BASELINE_GRID: dict[str, list[HyperConfig]] = {
    "cora": [
        HyperConfig(num_layers=nl, dropout=do, lr=lr, tau0=0.0)
        for nl, do, lr in product([2, 3, 4], [0.0, 0.1, 0.2], [0.001])
    ],
    "citeseer": [
        HyperConfig(num_layers=nl, dropout=do, lr=lr, tau0=0.0)
        for nl, do, lr in product([2, 3, 4], [0.0, 0.1, 0.2], [0.001])
    ],
}

EXIT_GRID: dict[str, list[HyperConfig]] = {
    "cora": [
        HyperConfig(num_layers=nl, dropout=do, lr=0.001, tau0=tau0)
        for nl, do, tau0 in product([4, 8, 12], [0.0, 0.1], [0.0, 0.5, 1.0])
    ],
    "citeseer": [
        HyperConfig(num_layers=nl, dropout=do, lr=0.001, tau0=tau0)
        for nl, do, tau0 in product([4, 8, 12], [0.0, 0.1], [0.0, 0.5, 1.0])
    ],
}


def phase_tune() -> dict[str, dict[str, tuple[HyperConfig, float, float]]]:
    print("\n" + "=" * 70)
    print("PHASE 1: HYPERPARAMETER TUNING")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache = load_tuning_cache()
    best_configs: dict[str, dict[str, tuple[HyperConfig, float, float]]] = {}

    for dataset in ["cora", "citeseer"]:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        best_configs[dataset] = {}

        for exit_mode in ExitMode:
            grid = BASELINE_GRID[dataset] if exit_mode == ExitMode.NONE else EXIT_GRID[dataset]

            print(f"\n--- Tuning {exit_mode.value} on {dataset} ({len(grid)} configs) ---")

            best_val = -1.0
            best_test = -1.0
            best_cfg: HyperConfig | None = None

            for i, cfg in enumerate(grid):
                key = config_key(dataset, exit_mode, cfg)

                if key in cache:
                    val_mrr, test_mrr = cache[key]["val_mrr"], cache[key]["test_mrr"]
                    print(f"  [{i + 1}/{len(grid)}] {key} (cached): val={val_mrr:.4f}, test={test_mrr:.4f}")
                else:
                    print(f"  [{i + 1}/{len(grid)}] {key}: training...", end=" ", flush=True)
                    val_mrr, test_mrr, _ = train_sas_model(dataset, exit_mode, cfg, data, device)
                    cache[key] = {"val_mrr": val_mrr, "test_mrr": test_mrr}
                    save_tuning_cache(cache)
                    print(f"val={val_mrr:.4f}, test={test_mrr:.4f}")

                if val_mrr > best_val:
                    best_val = val_mrr
                    best_test = test_mrr
                    best_cfg = cfg

            assert best_cfg is not None
            best_configs[dataset][exit_mode.value] = (best_cfg, best_val, best_test)
            print(
                f"  BEST: L={best_cfg.num_layers}, do={best_cfg.dropout}, "
                f"tau0={best_cfg.tau0}, val={best_val:.4f}, test={best_test:.4f}"
            )

    return best_configs


def phase_train(
    best_configs: dict[str, dict[str, tuple[HyperConfig, float, float]]],
) -> dict[str, dict[str, int]]:
    print("\n" + "=" * 70)
    print("PHASE 2: FINAL TRAINING")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layer_counts: dict[str, dict[str, int]] = {}

    for dataset in ["cora", "citeseer"]:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")
        layer_counts[dataset] = {}

        for exit_mode in ExitMode:
            cfg, best_val, best_test = best_configs[dataset][exit_mode.value]
            print(f"\n--- Training final {exit_mode.value} on {dataset} ---")
            print(f"  Config: L={cfg.num_layers}, do={cfg.dropout}, tau0={cfg.tau0}, lr={cfg.lr}")

            train_sas_model(dataset, exit_mode, cfg, data, device, save=True)
            layer_counts[dataset][exit_mode.value] = cfg.num_layers

    return layer_counts


def phase_evaluate(layer_counts: dict[str, dict[str, int]]) -> None:
    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATION")
    print("=" * 70)

    from dataclasses import asdict

    from evaluate import evaluate_model, find_compute_balanced_L

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    compute_balanced_info: dict[str, dict] = {}

    for dataset in ["cora", "citeseer"]:
        data = read_data(dataset, HEART_DATASET_DIR, "samples.npy")

        for exit_mode in ExitMode:
            num_layers = layer_counts[dataset][exit_mode.value]
            ckpt = CHECKPOINTS_DIR / f"{dataset}_{exit_mode.value}_L{num_layers}.pt"
            print(f"\nEvaluating {exit_mode.value} on {dataset} (L={num_layers})...")
            result = evaluate_model(ckpt, data, device)
            results.append(result)
            print(
                f"  MRR={result.test_mrr:.4f}, Hits@1={result.test_hits_at_1:.4f}, "
                f"Hits@10={result.test_hits_at_10:.4f}, cost={result.total_compute_cost:.0f}"
            )

        node_l = layer_counts[dataset]["node_adaptive"]
        node_result = next(r for r in results if r.dataset == dataset and r.model_type == "node_adaptive")

        sub_l = layer_counts[dataset]["subgraph_adaptive"]
        sub_ckpt = CHECKPOINTS_DIR / f"{dataset}_subgraph_adaptive_L{sub_l}.pt"

        balanced_l = find_compute_balanced_L(sub_ckpt, data, device, node_result.total_compute_cost)
        print(f"  Compute-balanced L for subgraph_adaptive on {dataset}: {balanced_l}")

        if balanced_l != sub_l:
            balanced_result = evaluate_model(sub_ckpt, data, device, override_num_layers=balanced_l)
            results.append(balanced_result)
            print(f"  Balanced MRR={balanced_result.test_mrr:.4f}, cost={balanced_result.total_compute_cost:.0f}")

        compute_balanced_info[dataset] = {
            "node_adaptive_L": node_l,
            "node_adaptive_cost": node_result.total_compute_cost,
            "subgraph_adaptive_balanced_L": balanced_l,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for result in results:
        filename = f"{result.dataset}_{result.model_type}_L{result.num_layers}.json"
        with open(RESULTS_DIR / filename, "w") as f:
            json.dump(asdict(result), f, indent=2)

    with open(RESULTS_DIR / "compute_balanced.json", "w") as f:
        json.dump(compute_balanced_info, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}")


def phase_plot(layer_counts: dict[str, dict[str, int]]) -> None:
    print("\n" + "=" * 70)
    print("PHASE 4: PLOTTING")
    print("=" * 70)

    from plot import plot_dataset_with_layers

    for dataset in ["cora", "citeseer"]:
        plot_dataset_with_layers(dataset, layer_counts[dataset])


def phase_update_tables(
    best_configs: dict[str, dict[str, tuple[HyperConfig, float, float]]],
    layer_counts: dict[str, dict[str, int]],
) -> None:
    print("\n" + "=" * 70)
    print("PHASE 5: UPDATING TABLES")
    print("=" * 70)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ["cora", "citeseer"]:
        for exit_mode in ExitMode:
            cfg, val, tst = best_configs[dataset][exit_mode.value]
            num_layers = layer_counts[dataset][exit_mode.value]
            result_file = RESULTS_DIR / f"{dataset}_{exit_mode.value}_L{num_layers}.json"
            if result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)
                print(
                    f"  {dataset} {exit_mode.value}: MRR={result['test_mrr']:.4f}, "
                    f"Hits@1={result['test_hits_at_1']:.4f}, Hits@10={result['test_hits_at_10']:.4f}, "
                    f"cost={result['total_compute_cost']:.0f}"
                )

    ours_md = tables_dir / "ours.md"
    lines = ["# Our Results vs HeaRT Baselines\n"]
    lines.append("All results under HeaRT evaluation (500 hard negatives per positive sample).")
    lines.append("Values are percentages. Baselines reproduced with 5 seeds; our model uses 1 seed.")
    lines.append("SEAL not reproduced (torch compatibility issue); paper numbers used.\n")

    for dataset in ["cora", "citeseer"]:
        lines.append(f"## {dataset.capitalize()}\n")
        lines.append("| Model | MRR | Hits@1 | Hits@10 | Compute Cost |")
        lines.append("|-------|-----|--------|---------|----|")

        if dataset == "cora":
            lines.append("| GCN (reproduced) | 16.65±0.37 | 7.78±0.54 | 36.13±0.78 | — |")
            lines.append("| SEAL (paper) | 10.67±3.46 | 3.89±2.04 | 24.27±6.74 | — |")
            lines.append("| NCNC (reproduced) | 15.50±0.69 | 5.61±0.94 | 36.74±1.55 | — |")
        else:
            lines.append("| GCN (reproduced) | 21.02±0.58 | 9.36±0.51 | 47.16±0.46 | — |")
            lines.append("| SEAL (paper) | 13.16±1.66 | 5.08±1.31 | 27.37±3.20 | — |")
            lines.append("| NCNC (reproduced) | 23.48±1.23 | 10.29±1.19 | 53.98±0.80 | — |")

        baseline_cost = None
        for exit_mode in ExitMode:
            num_layers = layer_counts[dataset][exit_mode.value]
            result_file = RESULTS_DIR / f"{dataset}_{exit_mode.value}_L{num_layers}.json"
            if result_file.exists():
                with open(result_file) as f:
                    r = json.load(f)
                cost = r["total_compute_cost"]
                if exit_mode == ExitMode.NONE:
                    baseline_cost = cost
                    cost_str = f"{cost:.0f}"
                else:
                    if baseline_cost and baseline_cost > 0:
                        reduction = (1 - cost / baseline_cost) * 100
                        cost_str = f"{cost:.0f} ({reduction:.1f}% ↓)"
                    else:
                        cost_str = f"{cost:.0f}"

                mode_label = {
                    ExitMode.NONE: f"Ours (no exit, L={num_layers})",
                    ExitMode.NODE_ADAPTIVE: f"Ours (node-adaptive, L={num_layers})",
                    ExitMode.SUBGRAPH_ADAPTIVE: f"Ours (subgraph-adaptive, L={num_layers})",
                }[exit_mode]

                lines.append(
                    f"| {mode_label} | {r['test_mrr'] * 100:.2f} | "
                    f"{r['test_hits_at_1'] * 100:.2f} | {r['test_hits_at_10'] * 100:.2f} | {cost_str} |"
                )
        lines.append("")

    with open(ours_md, "w") as f:
        f.write("\n".join(lines))

    print(f"  Updated {ours_md}")

    config_summary = {}
    for dataset in ["cora", "citeseer"]:
        config_summary[dataset] = {}
        for exit_mode in ExitMode:
            cfg, val, tst = best_configs[dataset][exit_mode.value]
            config_summary[dataset][exit_mode.value] = {
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "tau0": cfg.tau0,
                "best_val_mrr": val,
                "test_mrr_at_best_val": tst,
            }

    with open(RESULTS_DIR / "best_configs.json", "w") as f:
        json.dump(config_summary, f, indent=2)
    print(f"  Saved best configs to {RESULTS_DIR / 'best_configs.json'}")


def main() -> None:
    best_configs = phase_tune()
    layer_counts = phase_train(best_configs)
    phase_evaluate(layer_counts)
    phase_plot(layer_counts)
    phase_update_tables(best_configs, layer_counts)

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
