# act-experiments: how self-correcting are Qwen3 base models?

**Results: see [RESULTS.md](RESULTS.md).** The data is on Hugging Face (private dataset
`syvb/qwen3-last-position-patching`).

Replace the residual stream at the **last position** at one site with an ablated value, then measure how far every
later site's residual stream at that position moves away from the clean run. Repeating this for every site gives a
triangular (patched site × measured site) matrix per model. Sites: 0 = token embeddings, i = after i blocks.

Ablations (all applied only at the final position; earlier positions are untouched because attention is causal):

- **mean**: the mean residual over other contexts ending in the *same token*. This keeps the token's identity but
  removes context-specific information, which later layers could only recover by re-reading earlier positions.
- **resample**: the residual of one other real context ending in the same token (the donor). This is
  on-distribution, and it lets us ask how far the model moves back from the donor towards the true context.
- **global**: the mean over all tokens, which also deletes the token's identity. This is a contrast condition.

## Metrics (see `plot.py` for exact definitions)

- **disruption**[l, l']: E|patched − clean|² / E|clean − token mean|², both at l'. The perturbation's size relative
  to how much this token's activations naturally vary across contexts at that layer. On the diagonal it is 1 for
  mean ablation, ≈2 for resample (two independent contexts), and larger for global.
- **retained**[l, l']: share of the injected direction still present (1 = all, 0 = cancelled, >1 = grown).
  **along**: share of the final perturbation that lies along the injected direction. The rest is new, orthogonal
  differences created by later blocks. Per-block attention/MLP contributions to "retained" are plotted too.
- **pullback** (resample): fraction of the way back from the donor's activation to the clean one.
- **Output**: KL(clean ‖ patched) of the next-token distribution and the increase in loss on the true next token.

Raw size ratios (|diff at l'| / |injected at l|) are not used, because typical activation sizes grow ~10× over the
last blocks. Line plots show 90% bootstrap bands that resample whole docs.

## Caveats

- Only tokens seen ≥ `--min-count` times in the means set get a mean (~2k token types for 10M tokens, covering ~73%
  of positions). Test positions are sampled uniformly over those positions, so frequent function tokens (" the", ",",
  ".") dominate and rare content words are absent.
- Means are computed in bf16; rounding noise averages out over ≥500 occurrences. Patching runs in **fp32**: in bf16,
  per-row rounding noise is comparable to the effect of small early-layer patches.
- Means come from docs at the start of the FineWeb parquet file and test docs from the end (~35k vs. 10k of ~1M
  docs), so the two sets don't overlap.

## Files

- `common.py`: model/data loading, hooks, and `run_patched`. The context is run once and its KV cache is shared by a
  batch of last-token rows: row 0 is clean, and row 1 + s patches site s.
- `means.py`: pass 1, per-token mean residuals over FineWeb positions [64, 512).
- `patch.py`: pass 2, the patching experiment on held-out docs. It makes one `run_patched` call per ablation.
- `plot.py`: per-model heatmaps (disruption, retained, per-block), a cross-size figure, and a text summary.
- `test_patch.py`: checks `run_patched` (residuals, attention/MLP outputs, logits) against a plain full forward pass
  with a hook, in fp32.
- `check_means_precision.py`: checks that bf16 means match fp32 ones. On 0.6B the gap is 70–800× below the
  sampling noise of a mean from 500 occurrences.
- `log_wandb.py`: logs one model's result curves to W&B (metrics only).
- `patch.py --acts` also saves bf16 activations: the clean last-position residuals of every sample, the sample
  token ids, and every patched residual for the first 32 samples.
- `run_robust.sh`: an extra 1,000-sample pass after `run_all.sh` that also records distances without the 5
  massive-activation dims.
- `run_all.sh`: the full GPU run. Launch it with `nohup ./run_all.sh > run.log 2>&1 &`, and copy `out/` off the pod
  before terminating it.

## Running

```
python test_patch.py
python means.py --data data/fineweb_000.parquet --model Qwen/Qwen3-0.6B-Base --out out/means_0.6B.pt
python patch.py --data data/fineweb_000.parquet --model Qwen/Qwen3-0.6B-Base --means out/means_0.6B.pt --out out/metrics_0.6B.npz
python plot.py out/plots out/metrics_0.6B.npz
```
