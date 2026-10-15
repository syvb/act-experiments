"""Topic steering vectors: the mean context deviation (residual minus its token's mean) in docs about a topic.

For each topic in common.TOPICS, docs that mention it at least --min-mentions times are taken from the start of the
FineWeb file (steering test samples come from the end). At every site, the vector is the mean of (residual - token
mean) over positions in [MIN_POS, MAX_LEN) after the first mention, whose token has a mean. So it only carries
context, not token identity, and adding it once (scale 1) moves a residual by the average topic shift of real docs.
"""
import argparse
import re

import torch

from common import MIN_POS, TOPICS, hooks, iter_docs, load_model, residual_sites, topic_mentions

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
p.add_argument("--data", required=True, help="FineWeb parquet file")
p.add_argument("--means", required=True, help="output of means.py")
p.add_argument("--docs", type=int, default=200, help="docs per topic")
p.add_argument("--min-mentions", type=int, default=3)
p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
p.add_argument("--device", default="cuda")
p.add_argument("--out", required=True)
args = p.parse_args()

tok, model = load_model(args.model, args.device, getattr(torch, args.dtype))
m = torch.load(args.means)
assert m["model"] == args.model
slot = {t: i for i, t in enumerate(m["token_ids"].tolist())}
n_sites, d = model.config.num_hidden_layers + 1, model.config.hidden_size
topics = list(TOPICS)
sums = torch.zeros(len(topics), n_sites, d, dtype=torch.float64)
sq = torch.zeros(len(topics), n_sites, dtype=torch.float64)  # summed |residual - token mean|^2, for scale
count, docs = [0] * len(topics), [0] * len(topics)


@torch.no_grad()
def all_resids(ids):
    """Residual stream at every position and site: [n_sites, T, d]."""
    out = {}
    with hooks(residual_sites(model), lambda s, h: out.update({s: h[0].float().cpu()})):
        model.model(input_ids=torch.tensor([ids], device=model.device))
    return torch.stack([out[s] for s in range(n_sites)])


for ids, text in iter_docs(args.data, tok, with_text=True):
    starts = {}  # topic index -> first position to use
    for k, (pattern, _) in enumerate(TOPICS.values()):
        if docs[k] < args.docs and re.search(pattern, text, re.I):  # (cheap check before tokenizing with offsets)
            after = topic_mentions(text, ids, tok, pattern)
            if len(after) >= args.min_mentions:
                starts[k] = max(after[0], MIN_POS)
    if not starts:
        continue
    pos = [i for i in range(len(ids)) if ids[i] in slot]
    dev = all_resids(ids)[:, pos] - m["means"][:, [slot[ids[i]] for i in pos]]  # [n_sites, positions, d]
    for k, start in starts.items():
        use = torch.tensor(pos) >= start
        sums[k] += dev[:, use].sum(1).double()
        sq[k] += (dev[:, use] ** 2).sum((1, 2)).double()
        count[k] += int(use.sum())
        docs[k] += 1
    if min(docs) >= args.docs:
        break
    if sum(docs) % 50 == 0:
        print(f"docs per topic: {dict(zip(topics, docs))}", flush=True)

n = torch.tensor(count, dtype=torch.float64)
vectors = (sums / n[:, None, None]).float()  # [topic, n_sites, d]
spread = (sq / n[:, None]).float()  # [topic, n_sites]: mean |residual - token mean|^2 in the topic's docs
for k, t in enumerate(topics):
    share = (vectors[k] ** 2).sum(-1)[1:] / spread[k, 1:]
    print(f"{t}: {docs[k]} docs, {count[k]} positions; |vector|^2 / mean |residual - token mean|^2 at sites 1..: "
          f"min {share.min():.3f}, median {share.median():.3f}, max {share.max():.3f}")
torch.save(dict(model=args.model, topics=topics, vectors=vectors, spread=spread, positions=count, docs=docs,
                min_mentions=args.min_mentions), args.out)
