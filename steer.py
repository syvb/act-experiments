"""Steering: add a topic vector (from steer_vectors.py) to the last position's residual at one site, and measure how
much of the topic is still there at every later site and in the next-token prediction.

Samples are positions in held-out docs, read from the end of the FineWeb file:
  off-topic - docs that mention none of the topics. Each sample is steered with every topic in turn.
  on-topic  - one position per doc, after the first mention, in docs that mention a topic >= --min-mentions times.
              Steered with that topic only.
Conditions, at each scale s: steer_x{s} adds s * the topic vector at the patch site; random_x{s} adds a random
direction of the same size (fixed per sample and topic).
Per row (one sample steered with one topic), with v the topic vector:
  concept [condition, patch site, measured site] = <patched - clean, v at the measured site>. Real topic signals grow
      ~1e4x (squared) with depth, so plot_steer.py compares it with what the steer would carry if later blocks left it
      untouched: s * <v at the patch site, v at the measured site> (vector_dot).
  sq, proj = |patched - clean|^2 and <patched - clean, injected>, as in patch.py
  kl [condition, patch site] = KL(clean || patched); dtopic = change in log P(topic words) (common.TOPICS)
"""
import argparse
import collections
import os
import random
import re

import numpy as np
import torch

from common import MIN_POS, TOPICS, iter_docs, load_model, prefix_cache, run_patched, topic_mentions

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
p.add_argument("--data", required=True, help="FineWeb parquet file")
p.add_argument("--means", required=True, help="output of means.py (for the natural variation at each site)")
p.add_argument("--vectors", required=True, help="output of steer_vectors.py")
p.add_argument("--off-topic", type=int, default=100, help="off-topic samples (each steered with every topic)")
p.add_argument("--on-topic", type=int, default=50, help="on-topic samples per topic")
p.add_argument("--off-topic-docs", type=int, default=10000, help="held-out docs to draw off-topic samples from")
p.add_argument("--min-mentions", type=int, default=3)
p.add_argument("--scales", type=float, nargs="+", default=[1, 2, 4, 8, 16, 32])
p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
args = p.parse_args()
conditions = [f"steer_x{s:g}" for s in args.scales] + [f"random_x{s:g}" for s in args.scales]

tok, model = load_model(args.model, args.device, getattr(torch, args.dtype))
dev = model.device
m, vec = torch.load(args.means), torch.load(args.vectors)
assert m["model"] == vec["model"] == args.model
slot = {t: i for i, t in enumerate(m["token_ids"].tolist())}
topics = vec["topics"]
vectors = vec["vectors"].to(dev)  # [topic, n_sites, d]
words = [[tok(" " + w).input_ids[0] for w in TOPICS[t][1].split()] for t in topics]
n_sites = model.config.num_hidden_layers + 1

# Off-topic candidates: positions in the last --off-topic-docs docs that mention no topic. On-topic: the first
# --on-topic docs (from the end) with enough mentions of a topic. Only positions whose token has a mean are used.
docs, off_candidates, on_docs = [], [], {k: [] for k in range(len(topics))}  # on_docs: (doc, first position)
for ids, text in iter_docs(args.data, tok, reverse=True, with_text=True):
    found = {k: topic_mentions(text, ids, tok, TOPICS[t][0]) for k, t in enumerate(topics)
             if re.search(TOPICS[t][0], text, re.I)}
    if not found and len(docs) < args.off_topic_docs:
        off_candidates += [(len(docs), T) for T in range(MIN_POS, len(ids) - 1) if ids[T] in slot]
    for k, after in found.items():
        if len(after) >= args.min_mentions and len(on_docs[k]) < args.on_topic:
            on_docs[k].append((len(docs), max(after[0], MIN_POS)))
    docs.append(ids)
    if len(docs) >= args.off_topic_docs and all(len(v) >= args.on_topic for v in on_docs.values()):
        break
rng = random.Random(args.seed)
samples = [(doc, T, list(range(len(topics))), False) for doc, T in rng.sample(off_candidates, args.off_topic)]
for k, ds in on_docs.items():
    for doc, start in ds:
        positions = [T for T in range(start, len(docs[doc]) - 1) if docs[doc][T] in slot]
        if positions:
            samples.append((doc, rng.choice(positions), [k], True))

patch_site = torch.arange(-1, n_sites, device=dev)  # row 0 clean, row 1 + s patches site s
results = collections.defaultdict(list)
for i, (doc, T, steer_topics, on_topic) in enumerate(samples):
    ids = docs[doc]
    token = ids[T]
    cache = prefix_cache(model, ids[:T])
    token_mean = m["means"][:, slot[token]].to(dev)
    for k in steer_topics:
        v = vectors[k]
        direction = torch.randn(v.shape, generator=torch.Generator().manual_seed((doc * 1000 + T) * 10 + k)).to(dev)
        random_v = direction / direction.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)
        out = collections.defaultdict(list)  # metric -> one entry per condition
        for base in [v, random_v]:
            for s in args.scales:
                patch_vec = s * torch.cat([base[:1], base])  # row 0's entry is unused
                resid, _, _, logits = run_patched(model, cache, token, patch_site, patch_vec, add=True)
                clean, patched = resid[0], resid[1:]  # [site, d], [patch site, measured site, d]
                diff = patched - clean
                injected = diff[range(n_sites), range(n_sites)]
                logp = logits.log_softmax(-1)
                out["sq"].append((diff**2).sum(-1))
                out["proj"].append(torch.einsum("pmd,pd->pm", diff, injected))
                out["concept"].append(torch.einsum("pmd,md->pm", diff, v))
                out["kl"].append((logp[0].exp() * (logp[0] - logp[1:])).sum(-1))
                topic_logp = logp[:, words[k]].logsumexp(-1)
                out["dtopic"].append(topic_logp[1:] - topic_logp[0])
        results["doc"].append(doc)
        results["T"].append(T)
        results["token"].append(token)
        results["topic"].append(k)
        results["on_topic"].append(on_topic)
        results["clean_topic_logp"].append(topic_logp[0].item())
        results["spread"].append(((clean - token_mean) ** 2).sum(-1).cpu())  # [site]: natural variation
        for key, val in out.items():
            results[key].append(torch.stack(val).cpu())  # [condition, ...]
    if (i + 1) % 25 == 0:
        print(f"{i + 1}/{len(samples)} samples", flush=True)

arrays = {k: torch.stack(v).numpy() if torch.is_tensor(v[0]) else np.array(v) for k, v in results.items()}
np.savez(args.out + ".tmp.npz", model=args.model, dtype=args.dtype, topics=topics, conditions=conditions,
         scales=args.scales, vector_dot=torch.einsum("kld,kmd->klm", vectors, vectors).cpu().numpy(), **arrays)
os.replace(args.out + ".tmp.npz", args.out)  # never leave a truncated file behind
