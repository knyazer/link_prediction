from pathlib import Path

from shared import (
    ExitMode,
    HEART_DATASET_DIR,
    HEART_BENCHMARKING_DIR,
    build_sas_model,
)

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from evalutors import evaluate_mrr
from model import BackboneConfig, ExitConfig
from scoring import mlp_score
from utils import Logger, get_logger, init_seed

log_print = get_logger("ours_testrun", "log", str(HEART_BENCHMARKING_DIR / "config"))


def read_data(data_name: str, dir_path: Path | str, filename: str) -> dict:
    dir_path = str(dir_path)
    node_set: set[int] = set()
    train_pos: list[tuple[int, int]] = []
    valid_pos: list[tuple[int, int]] = []
    test_pos: list[tuple[int, int]] = []

    for split in ["train", "test", "valid"]:
        path = f"{dir_path}/{data_name}/{split}_pos.txt"
        for line in open(path):
            sub, obj = line.strip().split("\t")
            sub, obj = int(sub), int(obj)
            node_set.add(sub)
            node_set.add(obj)
            if sub == obj:
                continue
            if split == "train":
                train_pos.append((sub, obj))
            elif split == "valid":
                valid_pos.append((sub, obj))
            elif split == "test":
                test_pos.append((sub, obj))

    num_nodes = len(node_set)
    print(f"the number of nodes in {data_name} is: {num_nodes}")

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge, train_edge[[1, 0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    with open(f"{dir_path}/{data_name}/heart_valid_{filename}", "rb") as f:
        valid_neg = torch.from_numpy(np.load(f))
    with open(f"{dir_path}/{data_name}/heart_test_{filename}", "rb") as f:
        test_neg = torch.from_numpy(np.load(f))

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

    train_pos_tensor = torch.tensor(train_pos)
    valid_pos_t = torch.tensor(valid_pos)
    test_pos_t = torch.tensor(test_pos)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[: valid_pos_t.size(0)]
    train_val = train_pos_tensor[idx]

    feature_embeddings = torch.load(f"{dir_path}/{data_name}/gnn_feature", weights_only=False)
    feature_embeddings = feature_embeddings["entity_embedding"]

    return {
        "adj": adj,
        "train_pos": train_pos_tensor,
        "train_val": train_val,
        "valid_pos": valid_pos_t,
        "valid_neg": valid_neg,
        "test_pos": test_pos_t,
        "test_neg": test_neg,
        "x": feature_embeddings,
    }


def get_metric_score(
    evaluator_mrr: object,
    pos_train_pred: torch.Tensor,
    pos_val_pred: torch.Tensor,
    neg_val_pred: torch.Tensor,
    pos_test_pred: torch.Tensor,
    neg_test_pred: torch.Tensor,
) -> dict:
    result = {}
    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred)
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)

    result["MRR"] = (result_mrr_train["MRR"], result_mrr_val["MRR"], result_mrr_test["MRR"])
    for k in [1, 3, 10, 100]:
        result[f"Hits@{k}"] = (
            result_mrr_train[f"mrr_hit{k}"],
            result_mrr_val[f"mrr_hit{k}"],
            result_mrr_test[f"mrr_hit{k}"],
        )
    return result


def train(
    model: torch.nn.Module,
    score_func: torch.nn.Module,
    train_pos: torch.Tensor,
    x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
) -> float:
    model.train()
    score_func.train()

    total_loss = 0.0
    total_examples = 0

    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        num_nodes = x.size(0)

        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0

        train_edge_mask = train_pos[mask].transpose(1, 0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1, 0]]), dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)

        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(
            train_pos.device
        )

        model_out = model(x, adj)
        h = model_out.node_embeddings if hasattr(model_out, "node_embeddings") else model_out

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

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test_edge(
    score_func: torch.nn.Module,
    input_data: torch.Tensor,
    h: torch.Tensor,
    batch_size: int,
    negative_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    pos_preds: list[torch.Tensor] = []
    neg_preds: list[torch.Tensor] = []

    if negative_data is not None:
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))
            pos_preds.append(score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu())
            neg_preds.append(score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu())
        return torch.cat(pos_preds, dim=0), torch.cat(neg_preds, dim=0)
    else:
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds.append(score_func(h[edge[0]], h[edge[1]]).cpu())
        return torch.cat(pos_preds, dim=0), None


@torch.no_grad()
def test(
    model: torch.nn.Module,
    score_func: torch.nn.Module,
    data: dict,
    x: torch.Tensor,
    evaluator_mrr: object,
    batch_size: int,
) -> dict:
    model.eval()
    score_func.eval()

    model_out = model(x, data["adj"].to(x.device))
    h = model_out.node_embeddings if hasattr(model_out, "node_embeddings") else model_out

    pos_train_pred, _ = test_edge(score_func, data["train_val"], h, batch_size)
    pos_valid_pred, neg_valid_pred = test_edge(score_func, data["valid_pos"], h, batch_size, data["valid_neg"])
    pos_test_pred, neg_test_pred = test_edge(score_func, data["test_pos"], h, batch_size, data["test_neg"])

    pos_train_pred = torch.flatten(pos_train_pred)
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)

    return get_metric_score(
        evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ours - link prediction with early exit")
    parser.add_argument("--data_name", type=str, default="cora")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_layers_predictor", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=9999)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--kill_cnt", type=int, default=10)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--metric", type=str, default="MRR")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input_dir", type=str, default=str(HEART_DATASET_DIR))
    parser.add_argument("--filename", type=str, default="samples.npy")
    parser.add_argument("--eval_mrr_data_name", type=str, default="ogbl-citation2")
    parser.add_argument(
        "--exit_mode",
        type=lambda s: ExitMode(s),
        default=ExitMode.NONE,
        choices=list(ExitMode),
        metavar="{none,node_adaptive,subgraph_adaptive}",
    )
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--confidence_hidden_dim", type=int, default=64)
    parser.add_argument("--hard_gumbel", action="store_true", default=False)
    args = parser.parse_args()

    print(args)
    init_seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    data = read_data(args.data_name, args.input_dir, args.filename)

    x = data["x"].to(device)
    train_pos = data["train_pos"].to(device)
    input_channel = x.size(1)

    from ogb.linkproppred import Evaluator

    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)

    loggers = {key: Logger(args.runs) for key in ["Hits@1", "Hits@3", "Hits@10", "Hits@100", "MRR"]}

    for run in range(args.runs):
        print(f"###### Run {run} ######")
        seed = args.seed if args.runs == 1 else run
        init_seed(seed)

        backbone_config = BackboneConfig(
            in_channels=input_channel,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        exit_config = (
            ExitConfig(
                tau0=args.tau0, confidence_hidden_dim=args.confidence_hidden_dim, hard_gumbel=args.hard_gumbel
            )
            if args.exit_mode != ExitMode.NONE
            else None
        )
        model = build_sas_model(backbone_config, args.exit_mode, exit_config).to(device)

        score_func = mlp_score(
            args.hidden_channels,
            args.hidden_channels,
            1,
            args.num_layers_predictor,
            args.dropout,
        ).to(device)

        model.reset_parameters()
        score_func.reset_parameters()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(score_func.parameters()),
            lr=args.lr,
            weight_decay=args.l2,
        )

        best_valid = 0.0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, score_func, train_pos, x, optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results_rank = test(model, score_func, data, x, evaluator_mrr, args.batch_size)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                for key, result in results_rank.items():
                    train_hits, valid_hits, test_hits = result
                    log_print.info(
                        f"Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, "
                        f"Train: {100 * train_hits:.2f}%, Valid: {100 * valid_hits:.2f}%, Test: {100 * test_hits:.2f}%"
                    )
                print("---")

                best_valid_current = torch.tensor(loggers[args.metric].results[run])[:, 1].max()
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                else:
                    kill_cnt += 1
                    if kill_cnt > args.kill_cnt:
                        print("Early Stopping!!")
                        break

        for key in loggers:
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers:
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
