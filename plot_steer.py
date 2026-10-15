"""Plots and a text summary for a steer.py run.

usage: python plot_steer.py OUT_DIR steer.npz

With v_l the topic vector at site l and s the steer scale:
  uptake[l, l'] = E<patched - clean at l', v_l'> / (s <v_l, v_l'>): how much of the topic the perturbation carries at
      l', relative to an untouched steer (1 = as if later blocks ignored it, < 1 = removed, > 1 = the model added more
      of the topic). Real topic signals grow ~1e4x (squared) with depth, so this compares with doing nothing rather than
      with the topic's full size at l'.
  retained[l, l'] = share of the injected vector still present, as in plot.py (for steer and random directions).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot import GRID, INK, INK_2, heatmaps

SCALE_COLORS = ["#c6dcf8", "#9ec5f4", "#6aa6ec", "#3987e5", "#1e64c2", "#0d356b"]  # light -> dark: larger steers


def load(path):
    f = np.load(path)
    r = {k: f[k] for k in f.files}
    r["name"] = str(r["model"]).split("/")[-1]
    r["conditions"], r["topics"] = list(r["conditions"]), list(r["topics"])
    n = r["sq"].shape[-1]
    lower = np.tril(np.ones((n, n), bool), -1)
    steer = [c.startswith("steer") for c in r["conditions"]]
    untouched = r["vector_dot"][r["topic"]][:, None] * np.asarray(r["scales"])[None, :, None, None]  # [row, scale, l, l']
    injected = np.diagonal(r["sq"], axis1=2, axis2=3)  # [row, condition, patch site]
    with np.errstate(invalid="ignore", divide="ignore"):
        r["uptake"] = {on: r["concept"][r["on_topic"] == on][:, steer].mean(0) / untouched[r["on_topic"] == on].mean(0)
                       for on in [False, True]}  # [scale, l, l'] per group
        r["retained"] = {on: r["proj"][r["on_topic"] == on].mean(0) / injected[r["on_topic"] == on].mean(0)[..., None]
                         for on in [False, True]}  # [condition, l, l'] per group
    for M in [*r["uptake"].values(), *r["retained"].values()]:
        M[:, lower] = np.nan
        M[:, 0] = np.nan  # steering vectors are 0 at site 0 (embeddings carry no context)
    return r


def style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=11)
    ax.set_xlabel("patch site (relative depth)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.8)


def curves_figure(r, out):
    """Rows: off-topic, on-topic. Columns: uptake after the last block, change in log P(topic words), KL."""
    C, scales, n = r["conditions"], r["scales"], r["sq"].shape[-1]
    x = np.arange(1, n) / (n - 1)
    on_mask = r["on_topic"]
    natural = r["clean_topic_logp"][on_mask].mean() - r["clean_topic_logp"][~on_mask].mean()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, (label, on) in enumerate([("off-topic", False), ("on-topic", True)]):
        idx = np.flatnonzero(on_mask == on)
        for j, (color, s) in enumerate(zip(SCALE_COLORS, scales)):
            steer, rand = C.index(f"steer_x{s:g}"), C.index(f"random_x{s:g}")
            axes[row, 0].plot(x, r["uptake"][on][j, 1:, -1], color=color, lw=2, label=f"×{s:g}")
            axes[row, 1].plot(x, r["dtopic"][idx, steer, 1:].mean(0), color=color, lw=2, label=f"steer ×{s:g}")
            axes[row, 2].plot(x, r["kl"][idx, steer, 1:].mean(0), color=color, lw=2, label=f"steer ×{s:g}")
            axes[row, 2].plot(x, r["kl"][idx, rand, 1:].mean(0), color=color, lw=1.5, ls="--")
        axes[row, 0].axhline(1, color=INK_2, lw=1, ls=":")
        axes[row, 0].set_yscale("log", base=2)
        axes[row, 1].axhline(natural, color=INK_2, lw=1, ls=":", label="on-topic − off-topic (clean)")
        axes[row, 2].set_yscale("log")
        style(axes[row, 0], f"{label}: topic after the last block", "vs. an untouched steer (1 = same)")
        style(axes[row, 1], f"{label}: change in log P(topic words)", "nats")
        style(axes[row, 2], f"{label}: KL (dashed: random direction)", "KL, nats")
    axes[0, 0].legend(frameon=False, labelcolor=INK_2, title="steer scale", title_fontsize=9)
    axes[0, 1].legend(frameon=False, labelcolor=INK_2, fontsize=8)
    fig.suptitle(f"{r['name']}: topic steering at the last position (off-topic n = {(~on_mask).sum()} sample×topic "
                 f"rows, on-topic n = {on_mask.sum()})", color=INK, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(r):
    C, scales, n = r["conditions"], r["scales"], r["sq"].shape[-1]
    sites = [round(f * (n - 1)) for f in [0.25, 0.5, 0.75]]
    spread = r["spread"].mean(0)
    lines = [f"== {r['name']}  ({r['dtype']}; values after the last block, at depth "
             + ", ".join(f"{s / (n - 1):.2f}" for s in sites) + ")"]
    for k, t in enumerate(r["topics"]):
        vsq = np.diagonal(r["vector_dot"][k])
        mask = r["topic"] == k
        on, off = r["clean_topic_logp"][mask & r["on_topic"]], r["clean_topic_logp"][mask & ~r["on_topic"]]
        lines.append(f"{t}: |v|^2 / E|clean - token mean|^2 = " + "/".join(f"{vsq[s] / spread[s]:.3f}" for s in sites)
                     + f"; |v|^2 grows {vsq[-1] / vsq[1]:.0f}x from site 1 to the last; clean log P(topic words) "
                     f"on-topic {on.mean():.2f} vs off-topic {off.mean():.2f}")
    for label, on in [("off-topic", False), ("on-topic", True)]:
        idx = np.flatnonzero(r["on_topic"] == on)
        lines.append(f"-- {label} ({len(idx)} rows): uptake | retained steer / random | d log P(topic) | KL steer / random")
        for j, s in enumerate(scales):
            steer, rand = C.index(f"steer_x{s:g}"), C.index(f"random_x{s:g}")
            cells = [f"{r['uptake'][on][j, p, -1]:5.2f} | {r['retained'][on][steer, p, -1]:4.2f} / "
                     f"{r['retained'][on][rand, p, -1]:4.2f} | {r['dtopic'][idx, steer, p].mean():+6.2f} | "
                     f"{r['kl'][idx, steer, p].mean():6.3f} / {r['kl'][idx, rand, p].mean():6.3f}" for p in sites]
            lines.append(f"x{s:<4g} " + "  ||  ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir, path = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    r = load(path)
    C, scales, name = r["conditions"], list(r["scales"]), r["name"]
    note = f"off-topic {(~r['on_topic']).sum()} rows, on-topic {r['on_topic'].sum()} rows"
    s = scales[len(scales) // 2]
    heatmaps(r, [(f"{label}, steer ×{sc:g}", r["uptake"][on][scales.index(sc)])
                 for sc in [s, scales[-1]] for label, on in [("off-topic", False), ("on-topic", True)]],
             "topic carried by the perturbation, relative to an untouched steer",
             "uptake (log scale; 1 = as if untouched)",
             TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3), f"{out_dir}/{name}_steer_uptake.png", log2=True, n_note=note)
    heatmaps(r, [(f"{label}, {kind} ×{s:g}", r["retained"][on][C.index(f"{kind}_x{s:g}")])
                 for kind in ["steer", "random"] for label, on in [("off-topic", False), ("on-topic", True)]],
             "share of the injected vector still present", "retained (1 = all still there, 0 = cancelled, >1 = grown)",
             TwoSlopeNorm(vcenter=1, vmin=0, vmax=2), f"{out_dir}/{name}_steer_retained.png", n_note=note)
    curves_figure(r, f"{out_dir}/{name}_steer_curves.png")
    print(summary(r))
