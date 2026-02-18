from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import degree, remove_self_loops
from torch_sparse import SparseTensor


def relu_tanh(x: Tensor) -> Tensor:
    return F.relu(torch.tanh(x))


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
    hard_gumbel: bool = False


class AntiSymmetric(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_features, num_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x: Float[Tensor, "nodes features"]) -> Float[Tensor, "nodes features"]:
        antisym_W = self.W - self.W.T
        return F.relu(x @ antisym_W.T)


class PairwiseParametrization(nn.Module):
    def forward(self, W: Float[Tensor, "out_features in_features_plus2"]) -> Float[Tensor, "out_features out_features"]:
        W0 = W[:, :-2].triu(1)
        W0 = W0 + W0.T
        q = W[:, -2]
        r = W[:, -1]
        w_diag = torch.diag(q * torch.sum(torch.abs(W0), 1) + r)
        return W0 + w_diag


class Pairwise(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()
        self.lin = nn.Linear(num_hidden + 2, num_hidden, bias=False)
        parametrize.register_parametrization(self.lin, "weight", PairwiseParametrization(), unsafe=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def forward(self, x: Float[Tensor, "nodes features"]) -> Float[Tensor, "nodes features"]:
        return self.lin(x)


def sparse_tensor_to_edge_index(adj_t: SparseTensor) -> Int[Tensor, "2 edges"]:
    row, col, _ = adj_t.coo()
    return torch.stack([row, col], dim=0)


class SASConv(MessagePassing):
    def __init__(self, hidden_channels: int):
        super().__init__(aggr="add")
        self.antisymmetric_update = AntiSymmetric(hidden_channels)
        self.symmetric_aggr = Pairwise(hidden_channels)

    def reset_parameters(self) -> None:
        self.antisymmetric_update.reset_parameters()
        self.symmetric_aggr.reset_parameters()

    def forward(
        self, x: Float[Tensor, "nodes features"], adj_t: SparseTensor | Int[Tensor, "2 edges"]
    ) -> Float[Tensor, "nodes features"]:
        edge_index = sparse_tensor_to_edge_index(adj_t) if isinstance(adj_t, SparseTensor) else adj_t

        out = self.symmetric_aggr(x)
        edge_index_no_self, _ = remove_self_loops(edge_index)
        row, col = edge_index_no_self
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index_no_self, x=out, norm=norm)
        return -self.antisymmetric_update(x) + out

    def message(
        self, x_j: Float[Tensor, "edges features"], norm: Float[Tensor, " edges"]
    ) -> Float[Tensor, "edges features"]:
        return norm[:, None] * x_j


class WeightSharedSAS(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.in_channels, config.hidden_channels)
        self.conv = SASConv(config.hidden_channels)

    def reset_parameters(self) -> None:
        self.input_proj.reset_parameters()
        self.conv.reset_parameters()

    def forward(
        self, x: Float[Tensor, "nodes in_features"], adj_t: SparseTensor | Int[Tensor, "2 edges"]
    ) -> Float[Tensor, "nodes hidden"]:
        x = F.relu(self.input_proj(x.float()))
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        for _ in range(self.config.num_layers):
            delta = self.conv(x, adj_t)
            x = x + relu_tanh(delta)
        return x


class ConfidenceHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def reset_parameters(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x: Float[Tensor, "nodes features"]) -> Float[Tensor, "nodes 2"]:
        return self.mlp(x)


class TemperatureHead(nn.Module):
    def __init__(self, in_dim: int, tau0: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=False)
        self.softplus = nn.Softplus(beta=1)
        self.tau0 = tau0

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, x: Float[Tensor, "nodes features"]) -> Float[Tensor, "nodes 1"]:
        val = self.softplus(self.linear(x)) + self.tau0
        temp = val.pow(-1)
        return temp.masked_fill(temp == float("inf"), 0.0)


class ForwardResult(NamedTuple):
    node_embeddings: Float[Tensor, "nodes hidden"]
    exit_layers: Int[Tensor, " nodes"]
    active_nodes_per_layer: list[int]


class NodeAdaptiveExit(nn.Module):
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

    def forward(
        self, x: Float[Tensor, "nodes in_features"], adj_t: SparseTensor | Int[Tensor, "2 edges"]
    ) -> ForwardResult:
        num_nodes = x.size(0)
        device = x.device
        num_layers = self.backbone_config.num_layers
        hard = self.exit_config.hard_gumbel

        x = F.relu(self.input_proj(x.float()))
        x = F.dropout(x, p=self.backbone_config.dropout, training=self.training)

        z = torch.zeros_like(x)
        continue_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        exit_layers = torch.full((num_nodes,), num_layers, dtype=torch.long, device=device)
        active_nodes_per_layer: list[int] = []

        for layer_idx in range(num_layers):
            active_nodes_per_layer.append(int(continue_mask.sum().item()))

            logits = self.confidence_head(x)
            temp = self.temperature_head(x)
            gumbel_out = F.gumbel_softmax(logits=logits, tau=temp, hard=hard)

            step_size = gumbel_out[:, 0:1]

            exit_decision = gumbel_out[:, 1] > gumbel_out[:, 0]
            newly_exited = torch.logical_and(exit_decision, continue_mask)
            z = z + x * newly_exited[:, None].float()
            exit_layers = torch.where(newly_exited, torch.tensor(layer_idx, device=device), exit_layers)

            continue_mask = torch.logical_and(continue_mask, torch.logical_not(exit_decision))

            delta = self.conv(x, adj_t)
            x = x + step_size * relu_tanh(delta)

        z = z + x * continue_mask[:, None].float()

        return ForwardResult(
            node_embeddings=z,
            exit_layers=exit_layers,
            active_nodes_per_layer=active_nodes_per_layer,
        )


class HardExitHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def reset_parameters(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(
        self, x: Float[Tensor, "nodes features"], readiness: Float[Tensor, "nodes 1"]
    ) -> Float[Tensor, "nodes 2"]:
        return self.mlp(torch.cat([x, readiness], dim=-1))


def compute_neighbor_readiness(
    soft_exited: Bool[Tensor, " nodes"], edge_index: Int[Tensor, "2 edges"], num_nodes: int
) -> Float[Tensor, "nodes 1"]:
    row, col = edge_index
    neighbor_soft_exited = soft_exited[col].float()

    readiness_sum = torch.zeros(num_nodes, device=edge_index.device)
    readiness_sum.scatter_add_(0, row, neighbor_soft_exited)

    degree_count = torch.zeros(num_nodes, device=edge_index.device)
    degree_count.scatter_add_(0, row, torch.ones_like(neighbor_soft_exited))
    degree_count = degree_count.clamp(min=1)

    return (readiness_sum / degree_count)[:, None]


class SubgraphAdaptiveExit(nn.Module):
    def __init__(self, backbone_config: BackboneConfig, exit_config: ExitConfig):
        super().__init__()
        self.backbone_config = backbone_config
        self.exit_config = exit_config

        self.input_proj = nn.Linear(backbone_config.in_channels, backbone_config.hidden_channels)
        self.conv = SASConv(backbone_config.hidden_channels)

        hidden = backbone_config.hidden_channels
        self.soft_exit_head = ConfidenceHead(hidden, exit_config.confidence_hidden_dim)
        self.hard_exit_head = HardExitHead(hidden, exit_config.confidence_hidden_dim)
        self.temperature_head = TemperatureHead(hidden, exit_config.tau0)

    def reset_parameters(self) -> None:
        self.input_proj.reset_parameters()
        self.conv.reset_parameters()
        self.soft_exit_head.reset_parameters()
        self.hard_exit_head.reset_parameters()
        self.temperature_head.reset_parameters()

    def forward(
        self, x: Float[Tensor, "nodes in_features"], adj_t: SparseTensor | Int[Tensor, "2 edges"]
    ) -> ForwardResult:
        num_nodes = x.size(0)
        device = x.device
        num_layers = self.backbone_config.num_layers
        hard = self.exit_config.hard_gumbel

        edge_index = sparse_tensor_to_edge_index(adj_t) if isinstance(adj_t, SparseTensor) else adj_t
        edge_index_no_self, _ = remove_self_loops(edge_index)

        x = F.relu(self.input_proj(x.float()))
        x = F.dropout(x, p=self.backbone_config.dropout, training=self.training)

        z = torch.zeros_like(x)
        soft_exited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        hard_exited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        exit_layers = torch.full((num_nodes,), num_layers, dtype=torch.long, device=device)
        active_nodes_per_layer: list[int] = []

        for layer_idx in range(num_layers):
            active_nodes_per_layer.append(int(torch.logical_not(hard_exited).sum().item()))

            temp = self.temperature_head(x)

            soft_logits = self.soft_exit_head(x)
            soft_gumbel = F.gumbel_softmax(logits=soft_logits, tau=temp, hard=hard)
            soft_exit_decision = soft_gumbel[:, 1] > soft_gumbel[:, 0]
            newly_soft = torch.logical_and(soft_exit_decision, torch.logical_not(soft_exited))
            soft_exited = torch.logical_or(soft_exited, newly_soft)

            readiness = compute_neighbor_readiness(soft_exited, edge_index_no_self, num_nodes).detach()
            hard_logits = self.hard_exit_head(x, readiness)

            continue_logits = torch.zeros(num_nodes, 2, device=device)
            continue_logits[:, 0] = 1e6
            effective_hard_logits = torch.where(
                soft_exited[:, None], hard_logits, continue_logits
            )

            hard_gumbel_out = F.gumbel_softmax(logits=effective_hard_logits, tau=temp, hard=hard)
            step_size = hard_gumbel_out[:, 0:1]

            hard_exit_decision = hard_gumbel_out[:, 1] > hard_gumbel_out[:, 0]
            newly_hard = torch.logical_and(hard_exit_decision, torch.logical_not(hard_exited))
            z = z + x * newly_hard[:, None].float()
            exit_layers = torch.where(newly_hard, torch.tensor(layer_idx, device=device), exit_layers)
            hard_exited = torch.logical_or(hard_exited, newly_hard)

            delta = self.conv(x, adj_t)
            x = x + step_size * relu_tanh(delta)

        z = z + x * torch.logical_not(hard_exited)[:, None].float()

        return ForwardResult(
            node_embeddings=z,
            exit_layers=exit_layers,
            active_nodes_per_layer=active_nodes_per_layer,
        )


# ── GCN backbone (independent per-layer weights, matching HeaRT GCN) ──


class GCNBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        h = config.hidden_channels

        self.convs = nn.ModuleList()
        if config.num_layers == 1:
            self.convs.append(GCNConv(config.in_channels, h))
        else:
            self.convs.append(GCNConv(config.in_channels, h))
            for _ in range(config.num_layers - 2):
                self.convs.append(GCNConv(h, h))
            self.convs.append(GCNConv(h, h))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self, x: Float[Tensor, "nodes in_features"], adj_t: SparseTensor | Int[Tensor, "2 edges"]
    ) -> Float[Tensor, "nodes hidden"]:
        for conv in self.convs[:-1]:
            x = conv(x.float(), adj_t)
            x = F.gelu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GCNNodeAdaptiveExit(nn.Module):
    def __init__(self, backbone_config: BackboneConfig, exit_config: ExitConfig):
        super().__init__()
        self.backbone_config = backbone_config
        self.exit_config = exit_config
        h = backbone_config.hidden_channels

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(backbone_config.in_channels, h))
        for _ in range(backbone_config.num_layers - 1):
            self.convs.append(GCNConv(h, h))

        self.confidence_head = ConfidenceHead(h, exit_config.confidence_hidden_dim)
        self.temperature_head = TemperatureHead(h, exit_config.tau0)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        self.confidence_head.reset_parameters()
        self.temperature_head.reset_parameters()

    def forward(
        self, x: Float[Tensor, "nodes in_features"], adj_t: SparseTensor | Int[Tensor, "2 edges"]
    ) -> ForwardResult:
        num_nodes = x.size(0)
        device = x.device
        num_layers = self.backbone_config.num_layers
        hard = self.exit_config.hard_gumbel

        z = torch.zeros(num_nodes, self.backbone_config.hidden_channels, device=device)
        continue_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        exit_layers = torch.full((num_nodes,), num_layers, dtype=torch.long, device=device)
        active_nodes_per_layer: list[int] = []

        for layer_idx in range(num_layers):
            x = self.convs[layer_idx](x.float(), adj_t)

            if layer_idx < num_layers - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.backbone_config.dropout, training=self.training)

            active_nodes_per_layer.append(int(continue_mask.sum().item()))

            logits = self.confidence_head(x)
            temp = self.temperature_head(x)
            gumbel_out = F.gumbel_softmax(logits=logits, tau=temp, hard=hard)

            exit_decision = gumbel_out[:, 1] > gumbel_out[:, 0]
            newly_exited = torch.logical_and(exit_decision, continue_mask)
            z = z + x * newly_exited[:, None].float()
            exit_layers = torch.where(newly_exited, torch.tensor(layer_idx, device=device), exit_layers)

            continue_mask = torch.logical_and(continue_mask, torch.logical_not(exit_decision))

        z = z + x * continue_mask[:, None].float()

        return ForwardResult(
            node_embeddings=z,
            exit_layers=exit_layers,
            active_nodes_per_layer=active_nodes_per_layer,
        )


class GCNSubgraphAdaptiveExit(nn.Module):
    def __init__(self, backbone_config: BackboneConfig, exit_config: ExitConfig):
        super().__init__()
        self.backbone_config = backbone_config
        self.exit_config = exit_config
        h = backbone_config.hidden_channels

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(backbone_config.in_channels, h))
        for _ in range(backbone_config.num_layers - 1):
            self.convs.append(GCNConv(h, h))

        self.soft_exit_head = ConfidenceHead(h, exit_config.confidence_hidden_dim)
        self.hard_exit_head = HardExitHead(h, exit_config.confidence_hidden_dim)
        self.temperature_head = TemperatureHead(h, exit_config.tau0)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        self.soft_exit_head.reset_parameters()
        self.hard_exit_head.reset_parameters()
        self.temperature_head.reset_parameters()

    def forward(
        self, x: Float[Tensor, "nodes in_features"], adj_t: SparseTensor | Int[Tensor, "2 edges"]
    ) -> ForwardResult:
        num_nodes = x.size(0)
        device = x.device
        num_layers = self.backbone_config.num_layers
        hard = self.exit_config.hard_gumbel

        edge_index = sparse_tensor_to_edge_index(adj_t) if isinstance(adj_t, SparseTensor) else adj_t
        edge_index_no_self, _ = remove_self_loops(edge_index)

        z = torch.zeros(num_nodes, self.backbone_config.hidden_channels, device=device)
        soft_exited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        hard_exited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        exit_layers = torch.full((num_nodes,), num_layers, dtype=torch.long, device=device)
        active_nodes_per_layer: list[int] = []

        for layer_idx in range(num_layers):
            x = self.convs[layer_idx](x.float(), adj_t)

            if layer_idx < num_layers - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.backbone_config.dropout, training=self.training)

            active_nodes_per_layer.append(int(torch.logical_not(hard_exited).sum().item()))

            temp = self.temperature_head(x)

            soft_logits = self.soft_exit_head(x)
            soft_gumbel = F.gumbel_softmax(logits=soft_logits, tau=temp, hard=hard)
            soft_exit_decision = soft_gumbel[:, 1] > soft_gumbel[:, 0]
            newly_soft = torch.logical_and(soft_exit_decision, torch.logical_not(soft_exited))
            soft_exited = torch.logical_or(soft_exited, newly_soft)

            readiness = compute_neighbor_readiness(soft_exited, edge_index_no_self, num_nodes).detach()
            hard_logits = self.hard_exit_head(x, readiness)

            continue_logits = torch.zeros(num_nodes, 2, device=device)
            continue_logits[:, 0] = 1e6
            effective_hard_logits = torch.where(
                soft_exited[:, None], hard_logits, continue_logits
            )

            hard_gumbel_out = F.gumbel_softmax(logits=effective_hard_logits, tau=temp, hard=hard)

            hard_exit_decision = hard_gumbel_out[:, 1] > hard_gumbel_out[:, 0]
            newly_hard = torch.logical_and(hard_exit_decision, torch.logical_not(hard_exited))
            z = z + x * newly_hard[:, None].float()
            exit_layers = torch.where(newly_hard, torch.tensor(layer_idx, device=device), exit_layers)
            hard_exited = torch.logical_or(hard_exited, newly_hard)

        z = z + x * torch.logical_not(hard_exited)[:, None].float()

        return ForwardResult(
            node_embeddings=z,
            exit_layers=exit_layers,
            active_nodes_per_layer=active_nodes_per_layer,
        )
