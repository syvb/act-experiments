"""Pass 2: replace the last position's residual stream at one site and measure the effect at every later site.

For each test sample (a held-out doc cut at position T) and each ablation, row 0 of a batch is clean and row 1 + s
patches site s. Ablations:
  mean     - mean residual over other contexts ending in the same token (token identity kept, context removed)
  resample - the residual of one other real context ending in the same token (the "donor")
  global   - mean residual over all tokens (removes the token identity too)
Runs in fp32 by default: with bf16, rounding noise is comparable to the effect of small early-layer patches.
"""
import argparse
import collections
import os
import random

import numpy as np
import torch

from common import MIN_POS, iter_docs, last_position_resid, load_model, run_patched

ABLATIONS = ["mean", "resample", "global"]

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
p.add_argument("--data", required=True, help="FineWeb parquet file")
p.add_argument("--means", required=True, help="output of means.py")
p.add_argument("--samples", type=int, default=3000)
p.add_argument("--test-docs", type=int, default=10000, help="held-out docs to draw samples from")
p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--acts", help="also save activations (bf16) here: clean last-position residuals of every sample, and "
                              "all patched residuals of the first --full-acts samples")
p.add_argument("--full-acts", type=int, default=32)
args = p.parse_args()

tok, model = load_model(args.model, args.device, getattr(torch, args.dtype))
dev = model.device
m = torch.load(args.means)
assert m["model"] == args.model
means, global_mean = m["means"], m["global_mean"].to(dev)  # [n_sites, n_tokens, d] (kept on CPU), [n_sites, d]
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

patch_site = torch.arange(-1, n_sites, device=dev)  # row 0 clean, row 1 + s patches site s
results, acts = collections.defaultdict(list), collections.defaultdict(list)
for i, (doc, T, donor_doc, donor_T) in enumerate(samples):
    ids = docs[doc]
    token, next_token = ids[T], ids[T + 1]
    donor = last_position_resid(model, docs[donor_doc][: donor_T + 1])  # [n_sites, d]
    token_mean = means[:, slot[token]].to(dev)  # [n_sites, d]

    out, resids = collections.defaultdict(list), []  # metric -> one entry per ablation
    for patch_vec in [token_mean, donor, global_mean]:
        patch_vec = torch.cat([patch_vec[:1], patch_vec])  # row 0's entry is unused
        resid, attn, mlp, logits = run_patched(model, ids[:T], token, patch_site, patch_vec)
        resids.append(resid)
        clean, patched = resid[0], resid[1:]  # [site, d], [patch site, measured site, d]
        diff = patched - clean
        injected = diff[range(n_sites), range(n_sites)]  # [patch site, d]: the perturbation we put in
        logp = logits.log_softmax(-1)
        out["sq"].append((diff**2).sum(-1))  # [patch, measured]: squared size of the perturbation downstream
        out["proj"].append(torch.einsum("pmd,pd->pm", diff, injected))  # ... and its overlap with the injected one
        # [patch, block]: how much each block's attention / MLP output pushes against the injected perturbation
        out["attn_repair"].append(-torch.einsum("pld,pd->pl", attn[1:] - attn[0], injected))
        out["mlp_repair"].append(-torch.einsum("pld,pd->pl", mlp[1:] - mlp[0], injected))
        out["kl"].append((logp[0].exp() * (logp[0] - logp[1:])).sum(-1))  # [patch]: KL(clean || patched)
        out["dloss"].append(logp[0, next_token] - logp[1:, next_token])  # [patch]: increase in next-token loss
        # [patch, measured]: <patched - donor, clean - donor>. Divided by donor_gap, it is the fraction of the way
        # back from the donor's activation to the clean one (only meaningful for resample).
        out["pull"].append(torch.einsum("pmd,md->pm", patched - donor, clean - donor))

    results["doc"].append(doc)
    results["T"].append(T)
    results["token"].append(token)
    results["donor_doc"].append(donor_doc)
    results["donor_T"].append(donor_T)
    results["spread"].append(((clean - token_mean) ** 2).sum(-1).cpu())  # [site]: clean's distance from token mean
    results["donor_gap"].append(((clean - donor) ** 2).sum(-1).cpu())  # [site]: clean's distance from the donor
    for k, v in out.items():
        results[k].append(torch.stack(v).cpu())  # [ablation, ...]
    if args.acts:
        acts["ids"].append(torch.tensor(ids[: T + 2]))  # context, final token, and the true next token
        acts["clean"].append(clean.bfloat16().cpu())  # [site, d]
        if i < args.full_acts:
            acts["patched"].append(torch.stack(resids).bfloat16().cpu())  # [ablation, 1 + patch site, site, d]
    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{len(samples)} samples", flush=True)

arrays = {k: torch.stack(v).numpy() if torch.is_tensor(v[0]) else np.array(v) for k, v in results.items()}
np.savez(args.out + ".tmp.npz", model=args.model, dtype=args.dtype, ablations=ABLATIONS, **arrays)
os.replace(args.out + ".tmp.npz", args.out)  # never leave a truncated file behind

if args.acts:
    acts = dict(model=args.model, ablations=ABLATIONS, doc=results["doc"], T=results["T"], ids=acts["ids"],
                clean=torch.stack(acts["clean"]), patched=torch.stack(acts["patched"]) if acts["patched"] else None)
    torch.save(acts, args.acts + ".tmp")
    os.replace(args.acts + ".tmp", args.acts)
