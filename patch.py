"""Pass 2: replace the last position's residual stream at one site and measure the effect at every later site.

For each test sample (a held-out doc cut at position T), row 0 of the batch is clean and every other row patches one
(ablation, site). Ablations:
  mean     - mean residual over other contexts ending in the same token (token identity kept, context removed)
  resample - the residual of one other real context ending in the same token (the "donor")
  global   - mean residual over all tokens (removes the token identity too)
"""
import argparse
import collections
import random

import numpy as np
import torch

from common import MIN_POS, iter_docs, last_position_resid, load_model, run_patched

ABLATIONS = ["mean", "resample", "global"]

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
p.add_argument("--data", required=True, help="FineWeb parquet file")
p.add_argument("--means", required=True, help="output of means.py")
p.add_argument("--samples", type=int, default=5000)
p.add_argument("--test-docs", type=int, default=10000, help="held-out docs to draw samples from")
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
args = p.parse_args()

tok, model = load_model(args.model, args.device)
dev = model.device
m = torch.load(args.means)
assert m["model"] == args.model
means, global_mean = m["means"].to(dev), m["global_mean"].to(dev)  # [n_sites, n_tokens, d], [n_sites, d]
slot = {t: i for i, t in enumerate(m["token_ids"].tolist())}
n_sites = model.config.num_hidden_layers + 1

# Sample (doc, T) uniformly among held-out positions whose token has a mean; the donor is a random occurrence of the
# same token in a different doc. T+1 must exist so we can measure the loss on the true next token.
docs, where = [], collections.defaultdict(list)
for ids in iter_docs(args.data, tok, reverse=True):
    for T in range(MIN_POS, len(ids) - 1):
        if ids[T] in slot:
            where[ids[T]].append((len(docs), T))
    docs.append(ids)
    if len(docs) >= args.test_docs:
        break
rng = random.Random(args.seed)
candidates = [(doc, T) for occurrences in where.values() for doc, T in occurrences]
samples = []
for doc, T in rng.sample(candidates, min(len(candidates), 2 * args.samples)):
    donor_doc, donor_T = rng.choice(where[docs[doc][T]])
    if donor_doc != doc:  # (skip rather than retry when the draw lands in the same doc)
        samples.append((doc, T, donor_doc, donor_T))
    if len(samples) == args.samples:
        break

# Rows: 0 = clean, then one row per (ablation, site).
patch_site = torch.tensor([-1] + list(range(n_sites)) * len(ABLATIONS), device=dev)
results = collections.defaultdict(list)
for i, (doc, T, donor_doc, donor_T) in enumerate(samples):
    ids = docs[doc]
    token, next_token = ids[T], ids[T + 1]
    donor = last_position_resid(model, docs[donor_doc][: donor_T + 1])  # [n_sites, d]
    token_mean = means[:, slot[token]]  # [n_sites, d]
    patch_vec = torch.cat([torch.zeros(1, model.config.hidden_size, device=dev), token_mean, donor, global_mean])
    resid, attn, mlp, logits = run_patched(model, ids[:T], token, patch_site, patch_vec)
    clean = resid[0]  # [n_sites, d]
    logp = logits.log_softmax(-1)

    per_ablation = collections.defaultdict(list)
    for a in range(len(ABLATIONS)):
        rows = 1 + a * n_sites + torch.arange(n_sites, device=dev)
        diff = resid[rows] - clean  # [patch site, measured site, d]
        injected = diff[range(n_sites), range(n_sites)]  # [patch site, d]: the perturbation we put in
        per_ablation["sq"].append((diff**2).sum(-1))  # [patch, measured]: squared size of the perturbation downstream
        per_ablation["proj"].append(torch.einsum("pmd,pd->pm", diff, injected))  # ... and its overlap with the injected one
        # [patch, block]: how much each block's attention / MLP output pushes against the injected perturbation
        per_ablation["attn_repair"].append(-torch.einsum("pld,pd->pl", attn[rows] - attn[0], injected))
        per_ablation["mlp_repair"].append(-torch.einsum("pld,pd->pl", mlp[rows] - mlp[0], injected))
        per_ablation["kl"].append((logp[0].exp() * (logp[0] - logp[rows])).sum(-1))  # [patch]: KL(clean || patched)
        per_ablation["dloss"].append(logp[0, next_token] - logp[rows, next_token])  # [patch]: increase in next-token loss
        # [patch, measured]: <patched - donor, clean - donor>. Divided by donor_gap, it is the fraction of the way
        # back from the donor's activation to the clean one (only meaningful for resample).
        per_ablation["pull"].append(torch.einsum("pmd,md->pm", resid[rows] - donor, clean - donor))

    results["doc"].append(doc)
    results["T"].append(T)
    results["token"].append(token)
    results["spread"].append(((clean - token_mean) ** 2).sum(-1).cpu())  # [site]: clean's distance from token mean
    results["donor_gap"].append(((clean - donor) ** 2).sum(-1).cpu())  # [site]: clean's distance from the donor
    for k, v in per_ablation.items():
        results[k].append(torch.stack(v).cpu())  # [ablation, ...]
    if i % 10 == 0:
        print(f"\r{i + 1}/{len(samples)} samples", end="", flush=True)
print()

arrays = {k: torch.stack(v).numpy() if torch.is_tensor(v[0]) else np.array(v) for k, v in results.items()}
np.savez(args.out, model=args.model, ablations=ABLATIONS, **arrays)
