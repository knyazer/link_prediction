import sys
import time
from pathlib import Path

OURS_DIR = Path(__file__).resolve().parent
HEART_BENCHMARKING_DIR = OURS_DIR.parent / "HeaRT" / "benchmarking"
sys.path.insert(0, str(OURS_DIR))
sys.path.insert(0, str(HEART_BENCHMARKING_DIR))

import torch
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from main import read_data
from model import BackboneConfig, ExitConfig, ForwardResult, NodeAdaptiveExit, SubgraphAdaptiveExit, WeightSharedSAS
from scoring import mlp_score
from utils import init_seed

HEART_DATASET_DIR = HEART_BENCHMARKING_DIR.parent / "dataset"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_exit_distribution(out: ForwardResult, num_layers: int) -> None:
    exit_layers = out.exit_layers
    for layer in range(num_layers + 1):
        count = (exit_layers == layer).sum().item()
        frac = count / exit_layers.size(0)
        bar = "#" * int(frac * 50)
        print(f"  layer {layer:2d}: {frac:6.2%} ({count:5d} nodes) {bar}")
    print(f"  Active nodes per layer: {out.active_nodes_per_layer}")


def train_model(
    model: torch.nn.Module,
    score_func: torch.nn.Module,
    x: torch.Tensor,
    train_pos: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(score_func.parameters()),
        lr=lr,
    )
    num_nodes = x.size(0)

    model.train()
    score_func.train()
    for epoch in range(1, epochs + 1):
        for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
            optimizer.zero_grad()

            mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
            mask[perm] = 0
            train_edge_mask = train_pos[mask].transpose(1, 0)
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1, 0]]), dim=1)
            edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float)
            batch_adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes])

            model_out = model(x, batch_adj)
            h = model_out.node_embeddings if isinstance(model_out, ForwardResult) else model_out

            edge = train_pos[perm].t()
            pos_out = score_func(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            neg_edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long, device=h.device)
            neg_out = score_func(h[neg_edge[0]], h[neg_edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
            optimizer.step()


def diagnose(
    data_name: str = "cora",
    num_layers: int = 12,
    hidden_channels: int = 32,
    epochs: int = 150,
    batch_size: int = 1024,
    lr: float = 0.003,
    dropout: float = 0.2,
    tau0: float = 0.0,
) -> None:
    init_seed(999)
    device = torch.device("cpu")

    data = read_data(data_name, HEART_DATASET_DIR, "samples.npy")
    x = data["x"].to(device)
    train_pos = data["train_pos"].to(device)
    adj = data["adj"].to(device)
    input_channel = x.size(1)

    print(f"Dataset: {data_name}, nodes: {x.size(0)}, train edges: {train_pos.size(0)}")
    print(f"Hyperparams: hidden={hidden_channels}, L={num_layers}, tau0={tau0}, "
          f"lr={lr}, dropout={dropout}, epochs={epochs}")

    backbone_config = BackboneConfig(
        in_channels=input_channel,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )

    models: dict[str, torch.nn.Module] = {}
    models["none"] = WeightSharedSAS(backbone_config)

    for hard in [False, True]:
        label = "hard" if hard else "soft"
        exit_config = ExitConfig(tau0=tau0, confidence_hidden_dim=hidden_channels, hard_gumbel=hard)
        models[f"node_adaptive ({label})"] = NodeAdaptiveExit(backbone_config, exit_config)
        models[f"subgraph_adaptive ({label})"] = SubgraphAdaptiveExit(backbone_config, exit_config)

    for name, model in models.items():
        init_seed(999)
        model = model.to(device)
        model.reset_parameters()

        score_func = mlp_score(hidden_channels, hidden_channels, 1, 3, dropout).to(device)
        score_func.reset_parameters()

        param_count = count_parameters(model)
        print(f"\n{'='*60}")
        print(f"Model: {name} | params: {param_count:,}")

        train_model(model, score_func, x, train_pos, epochs, lr, batch_size)

        model.eval()
        timings: list[float] = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                out = model(x, adj)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        avg_time = sum(timings) / len(timings)
        print(f"After {epochs} epochs — forward time: {avg_time * 1000:.1f} ms")

        if isinstance(out, ForwardResult):
            print(f"Exit distribution:")
            print_exit_distribution(out, num_layers)
        else:
            print("(baseline — no exit layers)")


if __name__ == "__main__":
    diagnose()
