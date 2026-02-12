from dataclasses import dataclass
from enum import Enum
import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, remove_self_loops
from torch_sparse import SparseTensor


@dataclass(frozen=True)
class BackboneConfig:
    in_channels: int
    hidden_channels: int
    num_layers: int
    dropout: float


@dataclass(frozen=True)
class ExitConfig:
    tau0: float = 1.0
    confidence_hidden_dim: int = 64


class AntiSymmetric(nn.Module):
    """Linear transform with antisymmetric weight matrix (W - W^T)."""

    def __init__(self, num_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_features, num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        antisym_W = self.W - self.W.T
        return x @ antisym_W.T + self.bias


class _PairwiseParametrization(nn.Module):
    """Constrains weight to be symmetric with controlled diagonal."""

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        W0 = W[:, :-2].triu(1)
        W0 = W0 + W0.T
        q = W[:, -2]
        r = W[:, -1]
        w_diag = torch.diag(q * torch.sum(torch.abs(W0), 1) + r)
        return W0 + w_diag


class Pairwise(nn.Module):
    """Symmetric linear layer with pairwise parametrization."""

    def __init__(self, num_hidden: int):
        super().__init__()
        self.lin = nn.Linear(num_hidden + 2, num_hidden, bias=False)
        parametrize.register_parametrization(self.lin, "weight", _PairwiseParametrization(), unsafe=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _sparse_tensor_to_edge_index(adj_t: SparseTensor) -> torch.Tensor:
    row, col, _ = adj_t.coo()
    return torch.stack([row, col], dim=0)


class SASConv(MessagePassing):
    """SAS-GNN convolution: antisymmetric update + symmetric message passing.

    output_u = -AntiSymmetric(x_u) + sum_{v in N(u)} norm_uv * Pairwise(x_v)
    """

    def __init__(self, hidden_channels: int):
        super().__init__(aggr="add")
        self.antisymmetric_update = AntiSymmetric(hidden_channels)
        self.symmetric_aggr = Pairwise(hidden_channels)

    def reset_parameters(self) -> None:
        self.antisymmetric_update.reset_parameters()
        self.symmetric_aggr.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: SparseTensor | torch.Tensor) -> torch.Tensor:
        edge_index = _sparse_tensor_to_edge_index(adj_t) if isinstance(adj_t, SparseTensor) else adj_t

        out = self.symmetric_aggr(x)
        edge_index_no_self, _ = remove_self_loops(edge_index)
        row, col = edge_index_no_self
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index_no_self, x=out, norm=norm)
        return -self.antisymmetric_update(x) + out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class WeightSharedSAS(nn.Module):
    """Weight-shared SAS-GNN backbone for link prediction.

    Matches HeaRT's GNN interface: forward(x, adj_t) -> node_embeddings.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.in_channels, config.hidden_channels)
        self.conv = SASConv(config.hidden_channels)

    def reset_parameters(self) -> None:
        self.input_proj.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: SparseTensor | torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.input_proj(x.float()))
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        for _ in range(self.config.num_layers):
            delta = self.conv(x, adj_t)
            x = x + F.gelu(delta)
        return x


# ---------------------------------------------------------------------------
# Exit mechanism components
# ---------------------------------------------------------------------------


class ConfidenceHead(nn.Module):
    """MLP producing 2D logits (continue, exit) per node. Shared across layers."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )

    def reset_parameters(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemperatureHead(nn.Module):
    """Produces per-node inverse temperature for Gumbel-Softmax.

    temp = 1 / (softplus(linear(x)) + tau0)
    """

    def __init__(self, in_dim: int, tau0: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=False)
        self.softplus = nn.Softplus(beta=1)
        self.tau0 = tau0

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val = self.softplus(self.linear(x)) + self.tau0
        temp = val.pow(-1)
        return temp.masked_fill(temp == float("inf"), 0.0)


class ForwardResult(NamedTuple):
    node_embeddings: torch.Tensor
    exit_layers: torch.Tensor
    active_nodes_per_layer: list[int]


class NodeAdaptiveExit(nn.Module):
    """Algorithm A: Node-Adaptive Early Exit.

    Each node independently decides when to stop via Gumbel-Softmax.
    forward(x, adj_t) returns ForwardResult with per-node frozen embeddings.
    """

    def __init__(self, backbone_config: BackboneConfig, exit_config: ExitConfig):
        super().__init__()
        self.backbone_config = backbone_config
        self.exit_config = exit_config

        self.input_proj = nn.Linear(backbone_config.in_channels, backbone_config.hidden_channels)
        self.conv = SASConv(backbone_config.hidden_channels)

        hidden = backbone_config.hidden_channels
        self.confidence_head = ConfidenceHead(hidden, exit_config.confidence_hidden_dim)
        self.temperature_head = TemperatureHead(hidden, exit_config.tau0)

    def reset_parameters(self) -> None:
        self.input_proj.reset_parameters()
        self.conv.reset_parameters()
        self.confidence_head.reset_parameters()
        self.temperature_head.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: SparseTensor | torch.Tensor) -> ForwardResult:
        num_nodes = x.size(0)
        device = x.device
        num_layers = self.backbone_config.num_layers

        x = F.gelu(self.input_proj(x.float()))
        x = F.dropout(x, p=self.backbone_config.dropout, training=self.training)

        z = torch.zeros_like(x)
        continue_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        step_size = torch.ones(num_nodes, 1, device=device)
        exit_layers = torch.full((num_nodes,), num_layers, dtype=torch.long, device=device)
        active_nodes_per_layer: list[int] = []

        for layer_idx in range(num_layers):
            active_nodes_per_layer.append(int(continue_mask.sum().item()))

            logits = self.confidence_head(x)
            temp = self.temperature_head(x)
            gumbel_out = F.gumbel_softmax(logits=logits, tau=temp, hard=True)

            tau = gumbel_out[:, 0:1]
            step_size = step_size * tau

            exit_decision = gumbel_out[:, 1] > gumbel_out[:, 0]
            newly_exited = exit_decision & continue_mask
            z = z + x * newly_exited.unsqueeze(1).float()
            exit_layers = torch.where(newly_exited, torch.tensor(layer_idx, device=device), exit_layers)

            continue_mask = continue_mask & ~exit_decision

            delta = self.conv(x, adj_t)
            x = x + step_size * F.gelu(delta)

        z = z + x * continue_mask.unsqueeze(1).float()

        return ForwardResult(
            node_embeddings=z,
            exit_layers=exit_layers,
            active_nodes_per_layer=active_nodes_per_layer,
        )
