# Early-Exit Link Prediction

Reference implementation for the link-prediction experiments in `paper.pdf`:
**Early-Exit Graph Neural Networks for Link Prediction**.

Accepted to the **Learning on Graphs 2026 Italian Meetup**.

The core idea is adaptive GNN depth: use a stable SAS-GNN backbone, then let
confidence heads stop message passing once enough evidence has been gathered.

## What Is Implemented

- `WeightSharedSAS`: symmetric/anti-symmetric, weight-shared SAS-GNN backbone.
- `node_adaptive`: each node exits independently with Gumbel-Softmax heads.
- `subgraph_adaptive`: nodes exit after local readiness is propagated.
- Evaluation: MRR, Hits@K, exit distributions, active nodes/edges, and compute cost.

## Setup

```bash
uv sync
```

The code expects a HeaRT checkout at `./HeaRT`:

```text
HeaRT/
  benchmarking/
  dataset/{cora,citeseer,pubmed}/
```

Each dataset needs `train_pos.txt`, `valid_pos.txt`, `test_pos.txt`,
`heart_valid_samples.npy`, `heart_test_samples.npy`, and `gnn_feature`.

## Run

Single run:

```bash
uv run python ours/main.py --data_name cora --num_layers 12 --exit_mode node_adaptive
```

Full SAS-GNN pipeline on Cora and Citeseer:

```bash
uv run python ours/run_all.py
```

PubMed:

```bash
uv run python ours/run_pubmed.py
```

GCN baselines:

```bash
uv run python ours/run_gcn.py
```

## Outputs

- `checkpoints/`: trained backbone and scorer checkpoints.
- `results/*.json`: metrics, exits, and compute traces.
- `results/figures/`: generated plots.

## Paper Connection

Early exits only work when intermediate GNN states stay
useful at many depths. SAS-GNN supplies that stability; EEGNN-style heads turn
the maximum layer count into a compute budget rather than a fixed depth.

## Citation

Until there is a proceedings or arXiv entry, cite the PDF as a preprint:

```bibtex
@misc{knyazhitskiy2026earlyexitlinkprediction,
  title = {Early-Exit Graph Neural Networks for Link Prediction},
  author = {Knyazhitskiy, Roman and Di Francesco, Andrea Giuseppe},
  year = {2026},
  url = {https://github.com/knyazer/link_prediction},
  note = {Preprint. Accepted at the Learning on Graphs 2026 Italian Meetup}
}
```
