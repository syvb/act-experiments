"""Pass 1: mean residual stream at every site, per token, over positions [MIN_POS, MAX_LEN) of FineWeb docs.

Causal attention means each position's activations are exactly those of the doc truncated there, so one forward pass
over a doc gives a "document ending in token t" for every position.
"""
import argparse
import collections

import torch

from common import MIN_POS, hooks, iter_docs, load_model, residual_sites

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
p.add_argument("--data", required=True, help="FineWeb parquet file")
p.add_argument("--tokens", type=float, default=10e6, help="number of positions to average over")
p.add_argument("--min-count", type=int, default=500, help="only keep means of tokens seen at least this often")
p.add_argument("--batch-tokens", type=int, default=16384)
p.add_argument("--device", default="cuda")
p.add_argument("--out", required=True)
args = p.parse_args()

tok, model = load_model(args.model, args.device)
dev = model.device

# Tokenize enough docs, then pick the tokens that occur often enough to get a mean.
docs, n_pos = [], 0
for ids in iter_docs(args.data, tok):
    docs.append(ids)
    n_pos += len(ids) - MIN_POS
    if n_pos >= args.tokens:
        break
counts = collections.Counter(t for ids in docs for t in ids[MIN_POS:])
token_ids = sorted(t for t, c in counts.items() if c >= args.min_count)
slot = torch.full((model.config.vocab_size,), -1, dtype=torch.long, device=dev)  # token id -> row in `sums`
slot[token_ids] = torch.arange(len(token_ids), device=dev)
covered = sum(counts[t] for t in token_ids) / n_pos
print(f"{len(docs)} docs, {n_pos} positions, {len(token_ids)} tokens with >= {args.min_count} occurrences ({covered:.0%} of positions)")

n_sites, d = model.config.num_hidden_layers + 1, model.config.hidden_size
sums = torch.zeros(n_sites, len(token_ids), d, device=dev)
global_sum = torch.zeros(n_sites, d, device=dev)

docs.sort(key=len)  # length-sorted batches waste little on padding
start = 0
while start < len(docs):
    end = start + 1
    while end < len(docs) and (end + 1 - start) * len(docs[end]) <= args.batch_tokens:
        end += 1
    batch = docs[start:end]
    start = end

    L = len(batch[-1])
    ids = torch.zeros(len(batch), L, dtype=torch.long, device=dev)  # right-padded; padding never affects real tokens
    mask = torch.zeros(len(batch), L, dtype=torch.bool, device=dev)
    for i, x in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = True
    keep = mask & (torch.arange(L, device=dev) >= MIN_POS)  # positions that count towards the global mean
    keep_tracked = keep & (slot[ids] >= 0)  # ... and towards a per-token mean
    rows = slot[ids][keep_tracked]

    def accumulate(site, h):
        sums[site].index_add_(0, rows, h[keep_tracked].float())
        global_sum[site] += h[keep].float().sum(0)

    with torch.no_grad(), hooks(residual_sites(model), accumulate):
        model.model(input_ids=ids, attention_mask=mask)
    print(f"\r{start}/{len(docs)} docs", end="", flush=True)
print()

n = torch.tensor([counts[t] for t in token_ids], device=dev)
torch.save(
    dict(
        model=args.model,
        token_ids=torch.tensor(token_ids),
        counts=n.cpu(),
        means=(sums / n[None, :, None]).cpu(),  # [n_sites, n_tokens, d]
        global_mean=(global_sum / n_pos).cpu(),  # [n_sites, d]
    ),
    args.out,
)
