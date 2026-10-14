"""Does computing the per-token means in bf16 instead of fp32 matter?

usage: python check_means_precision.py means_bf16.pt means_fp32.pt metrics.npz

Compares |mean_bf16 - mean_fp32|^2 with the natural spread of activations around the token mean (from a patch.py
run), per site. A mean estimated from n occurrences is off by ~spread/n anyway, so a precision gap well below
1/500 of the spread (the minimum count in the real run) is irrelevant. Buckets by count tell random rounding noise
(shrinks like 1/n) from a systematic offset (doesn't shrink).
"""
import sys

import numpy as np
import torch

bf16, fp32 = torch.load(sys.argv[1]), torch.load(sys.argv[2])
assert torch.equal(bf16["token_ids"], fp32["token_ids"]) and torch.equal(bf16["counts"], fp32["counts"])
spread = np.load(sys.argv[3])["spread"].mean(0)  # [site]
counts = bf16["counts"].numpy()
gap = ((bf16["means"] - fp32["means"]) ** 2).sum(-1).numpy()  # [site, token]
weights = counts / counts.sum()  # test samples are drawn in proportion to token frequency

buckets = [(100, 300), (300, 1000), (1000, 10**9)]
print("site   gap/spread (frequency-weighted)   by count: " + "  ".join(f"[{lo}, {hi})" for lo, hi in buckets))
for s in range(1, gap.shape[0]):
    by_count = [gap[s, (counts >= lo) & (counts < hi)].mean() / spread[s] for lo, hi in buckets]
    print(f"{s:4d}   {(gap[s] * weights).sum() / spread[s]:.2e}" + 22 * " " + "  ".join(f"{x:.2e}" for x in by_count))
global_gap = ((bf16["global_mean"] - fp32["global_mean"]) ** 2).sum(-1).numpy()
print(f"global mean: max gap/spread over sites {np.max(global_gap[1:] / spread[1:]):.2e}")
print(f"reference: sampling noise of a mean from n occurrences is ~1/n of the spread (n=500: {1 / 500:.0e})")
