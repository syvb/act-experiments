"""Plots and a text summary for a patch.py run with --noise-scales (noise added to the clean residual).

usage: python plot_noise.py OUT_DIR metrics_noise.npz

The run must include mean and resample ablation. Metrics are those of plot.py, plus:
  induced[l, l'] = E|part of the perturbation at l' orthogonal to the injected vector|^2 / E|clean - token mean|^2.
      This is what later blocks changed because of the patch. Noise that later blocks ignore stays in the residual
      stream and counts towards disruption, but not towards induced.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot import GRID, INK, INK_2, SERIES, heatmaps, load

LOG_SCALE = "(log scale; 1 = as big as normal context variation)"
SCALE_COLORS = ["#9ec5f4", "#5598e7", "#1e64c2", "#0d356b"]  # light -> dark for larger noise


def induced(r):
    """[condition, patch site, measured site]"""
    injected = np.diagonal(r["sq"], axis1=2, axis2=3)[..., None]  # [sample, condition, patch site, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (r["sq"] - r["proj"] ** 2 / injected).mean(0) / r["spread"].mean(0)
    n = out.shape[-1]
    out[:, np.tril(np.ones((n, n), bool))] = np.nan  # 0 on the diagonal by definition
    out[:, 0] = np.nan  # patches at site 0 are no-ops
    return out


def band(r, x, n_boot=200, seed=0):
    """5th/95th percentiles of x.mean(0) (x: [sample, ...]), resampling whole docs."""
    rng = np.random.default_rng(seed)
    docs = r["by_doc"]
    boots = [x[np.concatenate([docs[k] for k in rng.integers(0, len(docs), len(docs))])].mean(0) for _ in range(n_boot)]
    return np.percentile(boots, [5, 95], axis=0)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.8)


def curves_figure(r, scales, depth_sites, out):
    """Top: KL after the last block vs patch depth, per noise scale. Bottom: KL vs noise scale at a few depths."""
    A, n = r["ablations"], r["sq"].shape[-1]
    x = np.arange(n) / (n - 1)
    kl_mean = r["kl"][:, A.index("mean")].mean(0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for col, kind in enumerate(["context", "random"]):
        ax = axes[0, col]
        for color, s in zip(SCALE_COLORS, scales):
            kl = r["kl"][:, A.index(f"add_{kind}_x{s:g}")]
            ax.fill_between(x[1:], *band(r, kl)[:, 1:], color=color, alpha=0.2, lw=0)
            ax.plot(x[1:], kl.mean(0)[1:], color=color, lw=2, label=f"×{s:g}")
        ax.plot(x[1:], kl_mean[1:], color=INK, lw=1.5, ls="--", label="mean ablation")
        ax.set_yscale("log")
        style(ax, f"add_{kind}: KL after the last block", "patch site (relative depth)", "KL, nats")
        ax.legend(frameon=False, labelcolor=INK_2, title="noise scale", title_fontsize=9)
        ax = axes[1, col]
        for color, s in zip(SERIES, depth_sites):
            kl = [r["kl"][:, A.index(f"add_{kind}_x{sc:g}"), s].mean() for sc in scales]
            ax.plot(scales, kl, color=color, lw=2, marker="o", ms=8, label=f"depth {s / (n - 1):.2f}")
            if s == depth_sites[0]:
                first = kl[0]
        ref = np.array([scales[0], scales[-1]])  # guide through the first point: a smooth response grows as scale^2
        ax.plot(ref, first * (ref / scales[0]) ** 2, color=INK_2, lw=1, ls=":", label="∝ scale²")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style(ax, f"add_{kind}: KL vs noise scale", "noise scale (× |donor − token mean|)", "KL, nats")
        ax.legend(frameon=False, labelcolor=INK_2)
    fig.suptitle(f"{r['name']}: noise added to the last position (n = {len(r['T'])} samples; "
                 "shaded: 90% bootstrap over docs)", color=INK, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def decomposition_figure(r, out):
    """Resample = mean ablation (remove the context) + add_context ×1 (add another one): do the effects add up?"""
    A, n = r["ablations"], r["sq"].shape[-1]
    x = np.arange(1, n) / (n - 1)
    parts = {"mean ablation (remove)": A.index("mean"), "add_context ×1 (add)": A.index("add_context_x1"),
             "resample (remove + add)": A.index("resample")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    metrics = [("KL after the last block, nats", r["kl"].mean(0)),
               ("disruption after the last block", r["sq"][..., -1].mean(0) / r["spread"][:, -1].mean())]
    for ax, (title, M) in zip(axes, metrics):
        for color, (label, a) in zip(SERIES, parts.items()):
            ax.plot(x, M[a, 1:], color=color, lw=2, label=label)
        ax.plot(x, M[parts["mean ablation (remove)"], 1:] + M[parts["add_context ×1 (add)"], 1:], color=INK, lw=1.5,
                ls="--", label="mean + add_context")
        ax.set_yscale("log")
        style(ax, title, "patch site (relative depth)", "")
    axes[0].legend(frameon=False, labelcolor=INK_2)
    fig.suptitle(f"{r['name']}: resample = remove the context + add another one (n = {len(r['T'])})", color=INK,
                 fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(r, ind, scales, depth_sites):
    A, n = r["ablations"], r["sq"].shape[-1]
    lines = [f"== {r['name']}  ({len(r['T'])} samples, {r['dtype']}; values after the last block)",
             "-- decomposition: KL of mean + add_context_x1 vs resample"]
    for s in depth_sites:
        kl = {a: r["kl"][:, A.index(a), s].mean() for a in ["mean", "add_context_x1", "resample"]}
        lines.append(f"site {s:2d} (depth {s / (n - 1):.2f}): {kl['mean']:.3f} + {kl['add_context_x1']:.3f} = "
                     f"{kl['mean'] + kl['add_context_x1']:.3f}  vs  resample {kl['resample']:.3f}")
    for a, name in enumerate(A):
        lines.append(f"-- {name}")
        lines.append("site  depth  disruption  induced  retained     KL    dloss")
        for s in range(1, n):
            lines.append(f"{s:4d}  {s / (n - 1):5.2f}  {r['disruption'][a, s, -1]:10.3f}  {ind[a, s, -1]:7.3f}  "
                         f"{r['retained'][a, s, -1]:8.2f}  {r['kl'][:, a, s].mean():6.3f}  {r['dloss'][:, a, s].mean():6.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir, path = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    r = load(path)
    A, n, name = r["ablations"], r["sq"].shape[-1], r["name"]
    scales = sorted(float(a.split("_x")[1]) for a in A if a.startswith("add_context_x"))
    depth_sites = [round(f * (n - 1)) for f in [0.25, 0.5, 0.75]]
    ind = induced(r)
    shown = [("mean ablation (remove)", "mean"), ("add_context ×1 (add)", "add_context_x1"),
             ("resample (remove + add)", "resample"), ("add_random ×1 (add)", "add_random_x1")]
    heatmaps(r, [(title, r["disruption"][A.index(a)]) for title, a in shown],
             "perturbation size relative to natural context variation", f"disruption {LOG_SCALE}",
             TwoSlopeNorm(vcenter=0, vmin=-6, vmax=6), f"{out_dir}/{name}_noise_disruption.png", log2=True)
    heatmaps(r, [(title, ind[A.index(a)]) for title, a in shown],
             "part of the perturbation orthogonal to the injected vector (what later blocks changed)",
             f"induced {LOG_SCALE}", TwoSlopeNorm(vcenter=0, vmin=-6, vmax=6), f"{out_dir}/{name}_noise_induced.png",
             log2=True)
    heatmaps(r, [(title, r["retained"][A.index(a)]) for title, a in shown],
             "share of the injected direction still present", "retained (1 = all still there, 0 = cancelled, >1 = grown)",
             TwoSlopeNorm(vcenter=1, vmin=0, vmax=2), f"{out_dir}/{name}_noise_retained.png")
    curves_figure(r, scales, depth_sites, f"{out_dir}/{name}_noise_curves.png")
    decomposition_figure(r, f"{out_dir}/{name}_noise_decomposition.png")
    print(summary(r, ind, scales, depth_sites))
