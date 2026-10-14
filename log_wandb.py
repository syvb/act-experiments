"""Log one model's result curves from a patch.py output to Weights & Biases (small metrics only, no activations).

usage: WANDB_API_KEY=... python log_wandb.py metrics.npz [project]
One run per model; the x axis is the patch site, and every metric is measured after the last block.
"""
import sys

import numpy as np
import wandb

from plot import curves, load

r = load(sys.argv[1])
n = r["sq"].shape[-1]
run = wandb.init(project=sys.argv[2] if len(sys.argv) > 2 else "qwen3-self-correction", name=r["name"],
                 config=dict(model=str(r["model"]), dtype=str(r["dtype"]), samples=len(r["T"]), layers=n - 1))
run.define_metric("patch_site")
run.define_metric("*", step_metric="patch_site")
pullback = curves(r, slice(None))["pullback toward clean (resample)"]
for s in range(n):
    row = {"patch_site": s, "relative_depth": s / (n - 1), "resample/pullback": pullback[s]}
    for a, name in enumerate(r["ablations"]):
        row[f"{name}/disruption"] = r["disruption"][a, s, -1]
        row[f"{name}/retained"] = r["retained"][a, s, -1]
        row[f"{name}/kl"] = r["kl"][:, a, s].mean()
        row[f"{name}/dloss"] = r["dloss"][:, a, s].mean()
    run.log({k: float(v) for k, v in row.items()})
half = np.flatnonzero(r["disruption"][r["ablations"].index("mean"), :, -1] >= 0.5)
if len(half):
    run.summary["mean/disruption_reaches_half_at_depth"] = half[0] / (n - 1)
run.finish()
