import sys
from dataclasses import dataclass
from pathlib import Path

OURS_DIR = Path(__file__).resolve().parent
HEART_BENCHMARKING_DIR = OURS_DIR.parent / "HeaRT" / "benchmarking"
sys.path.insert(0, str(OURS_DIR))
sys.path.insert(0, str(HEART_BENCHMARKING_DIR))

import torch

from main import ExitMode, read_data, train, test
from model import BackboneConfig, ExitConfig, NodeAdaptiveExit, SubgraphAdaptiveExit, WeightSharedSAS
from scoring import mlp_score
from utils import init_seed

HEART_DATASET_DIR = HEART_BENCHMARKING_DIR.parent / "dataset"
CHECKPOINTS_DIR = OURS_DIR.parent / "checkpoints"


@dataclass(frozen=True)
class TrainConfig:
    data_name: str
    exit_mode: ExitMode
    num_layers: int = 12
    hidden_channels: int = 32
    dropout: float = 0.2
    tau0: float = 0.0
    confidence_hidden_dim: int = 32
    lr: float = 0.003
    epochs: int = 150
    batch_size: int = 1024
    num_layers_predictor: int = 3
    seed: int = 999
    eval_steps: int = 5
    kill_cnt: int = 10


def train_and_save(config: TrainConfig) -> Path:
    init_seed(config.seed)
    device = torch.device("cpu")

    data = read_data(config.data_name, HEART_DATASET_DIR, "samples.npy")
    x = data["x"].to(device)
    train_pos = data["train_pos"].to(device)
    input_channel = x.size(1)

    backbone_config = BackboneConfig(
        in_channels=input_channel,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    match config.exit_mode:
        case ExitMode.NONE:
            model = WeightSharedSAS(backbone_config).to(device)
        case ExitMode.NODE_ADAPTIVE:
            exit_config = ExitConfig(tau0=config.tau0, confidence_hidden_dim=config.confidence_hidden_dim)
            model = NodeAdaptiveExit(backbone_config, exit_config).to(device)
        case ExitMode.SUBGRAPH_ADAPTIVE:
            exit_config = ExitConfig(tau0=config.tau0, confidence_hidden_dim=config.confidence_hidden_dim)
            model = SubgraphAdaptiveExit(backbone_config, exit_config).to(device)

    score_func = mlp_score(
        config.hidden_channels, config.hidden_channels, 1,
        config.num_layers_predictor, config.dropout,
    ).to(device)

    model.reset_parameters()
    score_func.reset_parameters()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(score_func.parameters()),
        lr=config.lr,
    )

    from ogb.linkproppred import Evaluator
    evaluator_mrr = Evaluator(name="ogbl-citation2")

    best_valid = 0.0
    kill_cnt = 0
    best_model_state: dict | None = None
    best_score_state: dict | None = None
    best_epoch = 0

    for epoch in range(1, 1 + config.epochs):
        loss = train(model, score_func, train_pos, x, optimizer, config.batch_size)

        if epoch % config.eval_steps == 0:
            results = test(model, score_func, data, x, evaluator_mrr, config.batch_size)
            val_mrr = results["MRR"][1]
            print(f"  Epoch {epoch}: loss={loss:.4f}, val_MRR={val_mrr:.4f}")

            if val_mrr > best_valid:
                best_valid = val_mrr
                kill_cnt = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_score_state = {k: v.cpu().clone() for k, v in score_func.state_dict().items()}
                best_epoch = epoch
            else:
                kill_cnt += 1
                if kill_cnt > config.kill_cnt:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    assert best_model_state is not None
    assert best_score_state is not None

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINTS_DIR / f"{config.data_name}_{config.exit_mode.value}_L{config.num_layers}.pt"

    torch.save({
        "model_state_dict": best_model_state,
        "score_func_state_dict": best_score_state,
        "backbone_config": {
            "in_channels": backbone_config.in_channels,
            "hidden_channels": backbone_config.hidden_channels,
            "num_layers": backbone_config.num_layers,
            "dropout": backbone_config.dropout,
        },
        "exit_mode": config.exit_mode.value,
        "data_name": config.data_name,
        "tau0": config.tau0,
        "confidence_hidden_dim": config.confidence_hidden_dim,
        "num_layers_predictor": config.num_layers_predictor,
        "best_epoch": best_epoch,
    }, save_path)

    print(f"Saved {save_path.name} (epoch {best_epoch}, val MRR: {best_valid:.4f})")
    return save_path


def main() -> None:
    for dataset in ["cora", "citeseer"]:
        for mode in ExitMode:
            print(f"\n{'='*60}")
            print(f"Training {mode.value} on {dataset}")
            print(f"{'='*60}")
            train_and_save(TrainConfig(data_name=dataset, exit_mode=mode))


if __name__ == "__main__":
    main()
