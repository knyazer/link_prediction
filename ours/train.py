from pathlib import Path

from shared import HEART_DATASET_DIR, ExitMode, HyperConfig

import torch

from main import read_data
from run_all import train_sas_model


def train_and_save(
    data_name: str,
    exit_mode: ExitMode,
    num_layers: int = 12,
    hidden_channels: int = 256,
    dropout: float = 0.2,
    tau0: float = 0.0,
    confidence_hidden_dim: int = 32,
    lr: float = 0.003,
    epochs: int = 150,
    batch_size: int = 1024,
    num_layers_predictor: int = 3,
    seed: int = 999,
    eval_steps: int = 5,
    kill_cnt: int = 10,
) -> Path:
    config = HyperConfig(
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        tau0=tau0,
        confidence_hidden_dim=confidence_hidden_dim,
        hidden_channels=hidden_channels,
        num_layers_predictor=num_layers_predictor,
        batch_size=batch_size,
        epochs=epochs,
        eval_steps=eval_steps,
        kill_cnt=kill_cnt,
        seed=seed,
    )

    data = read_data(data_name, HEART_DATASET_DIR, "samples.npy")
    device = torch.device("cpu")

    print(f"Training {exit_mode.value} on {data_name}")
    _, _, save_path = train_sas_model(data_name, exit_mode, config, data, device, save=True)
    assert save_path is not None
    return save_path


def main() -> None:
    for dataset in ["cora", "citeseer"]:
        for mode in ExitMode:
            print(f"\n{'=' * 60}")
            print(f"Training {mode.value} on {dataset}")
            print(f"{'=' * 60}")
            train_and_save(dataset, mode)


if __name__ == "__main__":
    main()
