"""Plots and a text summary from one or more patch.py outputs.

usage: python plot.py OUT_DIR metrics_a.npz [metrics_b.npz ...]

Disruption[l, l'] = size of the perturbation at site l', relative to how much this token's activations naturally vary
across contexts at l' (E|clean - token mean|^2), normalized so the diagonal (the injected perturbation) is 1.
  1 = the perturbation keeps its size relative to normal variation, <1 = later layers shrink it, >1 = they grow it.
Raw size ratios are not used: the residual norm grows ~10x over the last blocks, so they mostly show norm growth.

Erasure[l, l'] = component of the perturbation at l' along the injected one, as a share of the injected one.
  1 = all of the injected direction is still there, 0 = it has been cancelled.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#9c2828"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def load(path):
    f = np.load(path)
    r = {k: f[k] for k in f.files}
    sq = r["sq"].mean(0)  # [ablation, patch, measured]
    injected = np.diagonal(sq, axis1=1, axis2=2)  # [ablation, patch]
    with np.errstate(invalid="ignore", divide="ignore"):
        noop = injected < 1e-6 * sq.max()  # e.g. a same-token patch of the embeddings changes nothing
        relative = sq / r["spread"].mean(0)  # perturbation vs natural spread of this token's activations
        r["disruption"] = relative / np.diagonal(relative, axis1=1, axis2=2)[:, :, None]
        r["disruption"][:, 0, :] = np.nan  # embeddings don't vary across contexts, so the spread at site 0 is 0
        r["erasure"] = r["proj"].mean(0) / injected[:, :, None]
    n = sq.shape[-1]
    below_diag = np.tril(np.ones((n, n), bool), -1)
    for k in ["disruption", "erasure"]:
        r[k][:, below_diag] = np.nan
        r[k][noop] = np.nan
    r["name"] = str(r["model"]).split("/")[-1]
    return r


def heatmaps(r, key, title, cbar_label, norm, out, log2=False):
    ablations = list(r["ablations"])
    n = r[key].shape[-1]
    fig, axes = plt.subplots(1, len(ablations), figsize=(4.2 * len(ablations) + 1, 4.8), constrained_layout=True)
    for ax, a, name in zip(axes, range(len(ablations)), ablations):
        values = np.log2(r[key][a]) if log2 else r[key][a]
        im = ax.imshow(values, cmap=DIVERGING, norm=norm, origin="upper", interpolation="nearest")
        ax.set_title(f"{name} ablation", color=INK, fontsize=11)
        ax.set_xlabel("measured at site l'")
        ax.set_ylabel("patched at site l")
        ticks = np.arange(0, n, max(1, n // 7))
        ax.set_xticks(ticks), ax.set_yticks(ticks)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    if log2:
        cb.set_ticks(cb.get_ticks(), labels=[f"{2.0 ** t:g}" for t in cb.get_ticks()])
    cb.set_label(cbar_label, color=INK_2)
    cb.outline.set_visible(False)
    fig.suptitle(f"{r['name']}: {title}  (n = {len(r['T'])} samples; site 0 = embeddings, site i = after block i)",
                 color=INK, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def lines(runs, y_of, title, ylabel, out, log=True):
    """One panel per ablation; x = relative depth of the patch site, one line per model."""
    ablations = list(runs[0]["ablations"])
    fig, axes = plt.subplots(1, len(ablations), figsize=(4.2 * len(ablations), 3.6), sharey=True, constrained_layout=True)
    for ax, a, name in zip(axes, range(len(ablations)), ablations):
        for color, r in zip(SERIES, runs):
            y = y_of(r, a).astype(float)
            if log:
                y[y <= 0] = np.nan  # no-op patches (same-token ablations of the embeddings) have exactly zero effect
            x = np.arange(len(y)) / (len(y) - 1)
            ax.plot(x, y, color=color, lw=2, label=r["name"])
        ax.set_title(f"{name} ablation", color=INK, fontsize=11)
        ax.set_xlabel("patch site (relative depth)")
        ax.grid(True, color=GRID, lw=0.8)
        if log:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    if len(runs) > 1:
        axes[-1].legend(frameon=False, labelcolor=INK_2)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(r):
    """Text table: for each patch site, what's left at the last site and the effect on the output."""
    lines_ = [f"== {r['name']}  ({len(r['T'])} samples)"]
    ablations = list(r["ablations"])
    header = "site " + "".join(f"| {a:>8} disrupt  erasure  KL      dloss " for a in ablations)
    lines_.append(header)
    n = r["sq"].shape[-1]
    for site in range(n):
        row = f"{site:4d} "
        for a in range(len(ablations)):
            row += (f"| {r['disruption'][a, site, -1]:15.2f} {r['erasure'][a, site, -1]:8.2f} "
                    f"{r['kl'][:, a, site].mean():7.3f} {r['dloss'][:, a, site].mean():7.3f} ")
        lines_.append(row)
    return "\n".join(lines_)


if __name__ == "__main__":
    out_dir, paths = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    runs = [load(p) for p in paths]
    for r in runs:
        heatmaps(r, "disruption", "size of the perturbation relative to natural context variation",
                 "disruption (1 = same as injected, <1 shrunk, >1 grown)",
                 Normalize(vmin=-3, vmax=3), f"{out_dir}/{r['name']}_disruption.png", log2=True)
        heatmaps(r, "erasure", "share of the injected direction still present",
                 "erasure (1 = all still there, 0 = cancelled)",
                 TwoSlopeNorm(vcenter=1, vmin=0, vmax=2), f"{out_dir}/{r['name']}_erasure.png")
        print(summary(r))
    lines(runs, lambda r, a: r["kl"][:, a].mean(0), "Effect on next-token prediction",
          "KL(clean || patched), nats", f"{out_dir}/output_kl.png")
    lines(runs, lambda r, a: r["disruption"][a, :, -1], "Disruption left at the last site (after the final block)",
          "disruption at final site", f"{out_dir}/final_disruption.png")
