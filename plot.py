"""Plots and a text summary from one or more patch.py outputs.

usage: python plot.py OUT_DIR metrics_a.npz [metrics_b.npz ...]

Sites: 0 = token embeddings, i = residual stream after i blocks. All metrics are ratios of means over samples.
  disruption[l, l'] = E|patched - clean|^2 / E|clean - token mean|^2, both at l'. The perturbation's size relative to
      how much this token's activations naturally vary across contexts at l'. On the diagonal it is 1 for mean
      ablation, ~2 for resample (two independent contexts) and larger for global.
  retained[l, l'] = share of the injected direction still present at l' (1 = all of it, 0 = cancelled, >1 = grown).
  along = share of the final-site perturbation (squared size) that lies along the injected direction; the rest is new,
      orthogonal differences created by later blocks.
  pullback = (resample only) fraction of the way back from the donor's activation to the clean one at the final site.
Raw ratios E|diff at l'|^2 / E|injected at l|^2 are not plotted: typical activation sizes grow ~10x over the last
blocks, so they aren't comparable across layers. Disruption compares against natural variation at each layer instead.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

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
    r["name"] = str(r["model"]).split("/")[-1]
    r["ablations"] = list(r["ablations"])
    n = r["sq"].shape[-1]
    injected = np.diagonal(r["sq"], axis1=2, axis2=3)  # [sample, ablation, patch site]
    E_injected = injected.mean(0)[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        r["disruption"] = r["sq"].mean(0) / r["spread"].mean(0)
        if "sq_no_massive" in r:  # same, ignoring the few massive-activation dims (see patch.py --massive-dims)
            r["disruption_no_massive"] = r["sq_no_massive"].mean(0) / r["spread_no_massive"].mean(0)
        r["retained"] = r["proj"].mean(0) / E_injected
        # per-block change in the retained share; block j writes into site j + 1
        r["block_attn"], r["block_mlp"] = [np.full_like(r["retained"], np.nan) for _ in range(2)]
        r["block_attn"][:, :, 1:] = -r["attn_repair"].mean(0) / E_injected
        r["block_mlp"][:, :, 1:] = -r["mlp_repair"].mean(0) / E_injected
    for k, include_diag in [("disruption", True), ("disruption_no_massive", True), ("retained", True),
                            ("block_attn", False), ("block_mlp", False)]:
        if k in r:
            r[k][:, np.tril(np.ones((n, n), bool), -1 if include_diag else 0)] = np.nan
    for k in ["disruption", "disruption_no_massive"]:
        if k in r:
            r[k][:, :, 0] = np.nan  # embeddings don't vary across contexts (spread is 0)
            r[k][[a != "global" for a in r["ablations"]], 0] = np.nan  # same-token patches of embeddings: no-ops
    for k in ["retained", "block_attn", "block_mlp"]:
        r[k][:, 0] = np.nan  # ... and global's embedding-scale perturbation isn't comparable to the later ones
    docs = np.unique(r["doc"], return_inverse=True)[1]
    r["by_doc"] = [np.flatnonzero(docs == k) for k in range(docs.max() + 1)]
    return r


def curves(r, idx):
    """Per-patch-site curves after the last block, from the samples in idx."""
    mean, res = r["ablations"].index("mean"), r["ablations"].index("resample")
    return {
        "disruption (mean ablation)": r["sq"][idx, mean, :, -1].mean(0) / r["spread"][idx, -1].mean(),
        "pullback toward clean (resample)": r["pull"][idx, res, :, -1].mean(0) / r["donor_gap"][idx, -1].mean(),
        "KL, nats (mean ablation)": r["kl"][idx, mean].mean(0),
        "KL, nats (resample)": r["kl"][idx, res].mean(0),
    }


def bootstrap(r, n_boot=200, seed=0):
    """5th/95th percentiles of curves(), resampling whole docs (samples from one doc are correlated)."""
    rng = np.random.default_rng(seed)
    boots = [curves(r, np.concatenate([r["by_doc"][k] for k in rng.integers(0, len(r["by_doc"]), len(r["by_doc"]))]))
             for _ in range(n_boot)]
    return {k: np.nanpercentile([b[k] for b in boots], [5, 95], axis=0) for k in boots[0]}


def heatmaps(r, panels, title, cbar_label, norm, out, log2=False, xlabel="measured at site l'", n_note=None):
    """One heatmap per (subtitle, [patch site, measured site] matrix) panel."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels) + 1, 4.8), constrained_layout=True)
    for ax, (subtitle, M) in zip(axes, panels):
        with np.errstate(invalid="ignore", divide="ignore"):
            im = ax.imshow(np.log2(M) if log2 else M, cmap=DIVERGING, norm=norm, interpolation="nearest")
        ax.set_title(subtitle, color=INK, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("patched at site l")
        ticks = np.arange(0, M.shape[0], max(1, M.shape[0] // 7))
        ax.set_xticks(ticks), ax.set_yticks(ticks)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, extend="both")
    if log2:
        cb.set_ticks(cb.get_ticks(), labels=[f"{2**t:g}" if t >= 0 else f"1/{2**-t:g}" for t in cb.get_ticks()])
    cb.set_label(cbar_label, color=INK_2)
    cb.outline.set_visible(False)
    n_note = n_note or f"n = {len(r['T'])} samples"
    fig.suptitle(f"{r['name']}: {title}\n({n_note}, {r['dtype']}; "
                 "site 0 = embeddings, site i = after i blocks)", color=INK, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def cross_size(runs, out):
    """Curves after the last block vs relative depth of the patch site, one line per model, 90% doc-bootstrap bands."""
    keys = list(curves(runs[0], slice(None)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for color, r in zip(SERIES, runs):
        point, band = curves(r, slice(None)), bootstrap(r)
        x = np.arange(len(point[keys[0]])) / (len(point[keys[0]]) - 1)
        for ax, k in zip(axes.flat, keys):
            ax.fill_between(x, *band[k], color=color, alpha=0.2, lw=0)
            ax.plot(x, point[k], color=color, lw=2, label=r["name"])
    for ax, k in zip(axes.flat, keys):
        ax.set_title(k, color=INK, fontsize=11)
        ax.set_xlabel("patch site (relative depth)")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, labelcolor=INK_2)
    fig.suptitle("Effect after the last block, by depth of the patch (shaded: 90% bootstrap over docs)",
                 color=INK, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(r):
    """Text table per ablation: for each patch site, what's left after the last block and the effect on the output."""
    n = r["sq"].shape[-1]
    below_diag = r["sq"][:, :, np.tril(np.ones((n, n), bool), -1)]
    lines = [f"== {r['name']}  ({len(r['T'])} samples, {r['dtype']}; values after the last block)",
             f"max |diff|^2 at sites before the patch site: {below_diag.max():.3g} (must be 0)"]
    disruption = curves(r, slice(None))["disruption (mean ablation)"]
    half = np.flatnonzero(disruption >= 0.5)
    if len(half):
        lines.append(f"mean-ablation disruption first reaches 0.5 at site {half[0]} (relative depth {half[0] / (n - 1):.2f})")
    with np.errstate(invalid="ignore", divide="ignore"):
        per_sample = r["sq"][..., -1] / r["spread"][:, None, None, -1]  # [sample, ablation, patch site]
        injected = np.diagonal(r["sq"], axis1=2, axis2=3)
        along = (r["proj"][..., -1] ** 2 / injected).mean(0) / r["sq"][..., -1].mean(0)
        pullback = r["pull"][..., -1].mean(0) / r["donor_gap"][:, -1].mean()
    for a, name in enumerate(r["ablations"]):
        lines.append(f"-- {name} ablation")
        lines.append("site  disruption (median)  retained  along   KL      dloss" + ("   pullback" if name == "resample" else ""))
        for s in range(n):
            lines.append(f"{s:4d}  {r['disruption'][a, s, -1]:10.3f} {np.median(per_sample[:, a, s]):8.3f}  "
                         f"{r['retained'][a, s, -1]:8.2f}  {along[a, s]:5.2f}  {r['kl'][:, a, s].mean():6.3f} "
                         f"{r['dloss'][:, a, s].mean():7.3f}" + (f"  {pullback[a, s]:8.2f}" if name == "resample" else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir, paths = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    runs = [load(p) for p in paths]
    for r in runs:
        name = r["name"]
        # symmetric log scale (1/64 and 64 equally saturated); older *_disruption.png files used 1/64..8
        heatmaps(r, [(f"{a} ablation", r["disruption"][i]) for i, a in enumerate(r["ablations"])],
                 "perturbation size relative to natural context variation",
                 "disruption (log scale; 1 = as big as normal context variation)",
                 TwoSlopeNorm(vcenter=0, vmin=-6, vmax=6), f"{out_dir}/{name}_disruption_sym.png", log2=True)
        if "disruption_no_massive" in r:
            heatmaps(r, [(f"{a} ablation", r["disruption_no_massive"][i]) for i, a in enumerate(r["ablations"])],
                     f"disruption ignoring the {len(r['massive_dims'])} massive-activation dims",
                     "disruption (log scale; 1 = as big as normal context variation)",
                     TwoSlopeNorm(vcenter=0, vmin=-6, vmax=6), f"{out_dir}/{name}_disruption_no_massive_sym.png",
                     log2=True)
        heatmaps(r, [(f"{a} ablation", r["retained"][i]) for i, a in enumerate(r["ablations"])],
                 "share of the injected direction still present",
                 "retained (1 = all still there, 0 = cancelled, >1 = grown)",
                 TwoSlopeNorm(vcenter=1, vmin=0, vmax=2), f"{out_dir}/{name}_retained.png")
        mean = r["ablations"].index("mean")
        heatmaps(r, [("attention", r["block_attn"][mean]), ("MLP", r["block_mlp"][mean])],
                 "per-block change in the retained share (mean ablation)",
                 "change in retained share (<0 cancels, >0 grows)",
                 TwoSlopeNorm(vcenter=0, vmin=-0.25, vmax=0.25), f"{out_dir}/{name}_blocks.png",
                 xlabel="block writing into site l'")
        print(summary(r))
    cross_size(runs, f"{out_dir}/cross_size.png")
