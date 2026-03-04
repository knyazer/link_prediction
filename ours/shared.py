import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

OURS_DIR = Path(__file__).resolve().parent
HEART_BENCHMARKING_DIR = OURS_DIR.parent / "HeaRT" / "benchmarking"
sys.path.insert(0, str(OURS_DIR))
sys.path.insert(0, str(HEART_BENCHMARKING_DIR))

HEART_DATASET_DIR = HEART_BENCHMARKING_DIR.parent / "dataset"
CHECKPOINTS_DIR = OURS_DIR.parent / "checkpoints"
RESULTS_DIR = OURS_DIR.parent / "results"
TUNING_CACHE = RESULTS_DIR / "tuning_cache.json"

import torch
from torch.utils.data import DataLoader

from model import BackboneConfig, ExitConfig, NodeAdaptiveExit, SubgraphAdaptiveExit, WeightSharedSAS


class ExitMode(Enum):
    NONE = "none"
    NODE_ADAPTIVE = "node_adaptive"
    SUBGRAPH_ADAPTIVE = "subgraph_adaptive"


@dataclass(frozen=True)
class HyperConfig:
    num_layers: int
    dropout: float
    lr: float
    tau0: float
    confidence_hidden_dim: int = 32
    hidden_channels: int = 256
    num_layers_predictor: int = 3
    batch_size: int = 1024
    epochs: int = 150
    eval_steps: int = 5
    kill_cnt: int = 10
    seed: int = 999


def config_key(dataset: str, exit_mode: ExitMode, config: HyperConfig) -> str:
    return f"{dataset}_{exit_mode.value}_L{config.num_layers}_d{config.dropout}_lr{config.lr}_tau{config.tau0}"


def load_tuning_cache() -> dict:
    if TUNING_CACHE.exists():
        with open(TUNING_CACHE) as f:
            return json.load(f)
    return {}


def save_tuning_cache(cache: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TUNING_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def build_sas_model(
    backbone_config: BackboneConfig,
    exit_mode: ExitMode,
    exit_config: ExitConfig | None = None,
) -> torch.nn.Module:
    match exit_mode:
        case ExitMode.NONE:
            return WeightSharedSAS(backbone_config)
        case ExitMode.NODE_ADAPTIVE:
            assert exit_config is not None
            return NodeAdaptiveExit(backbone_config, exit_config)
        case ExitMode.SUBGRAPH_ADAPTIVE:
            assert exit_config is not None
            return SubgraphAdaptiveExit(backbone_config, exit_config)


def save_checkpoint(
    save_path: Path,
    model_state: dict,
    score_state: dict,
    backbone_config: BackboneConfig,
    exit_mode: ExitMode,
    dataset: str,
    tau0: float,
    confidence_hidden_dim: int,
    num_layers_predictor: int,
    best_epoch: int,
    backbone_type: str | None = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict = {
        "model_state_dict": model_state,
        "score_func_state_dict": score_state,
        "backbone_config": {
            "in_channels": backbone_config.in_channels,
            "hidden_channels": backbone_config.hidden_channels,
            "num_layers": backbone_config.num_layers,
            "dropout": backbone_config.dropout,
        },
        "exit_mode": exit_mode.value,
        "data_name": dataset,
        "tau0": tau0,
        "confidence_hidden_dim": confidence_hidden_dim,
        "num_layers_predictor": num_layers_predictor,
        "best_epoch": best_epoch,
    }
    if backbone_type is not None:
        ckpt["backbone_type"] = backbone_type
    torch.save(ckpt, save_path)


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
