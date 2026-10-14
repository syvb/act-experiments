# act-experiments: how self-correcting are Qwen3 base models?

Replace the residual stream at the **last position** after some layer with an ablated value, then measure how far
every later layer's residual stream at that position moves away from the clean run. Repeating this for every layer
gives a triangular (patched layer × measured layer) matrix per model.

Ablations (all applied only at the final position; earlier positions are untouched because attention is causal):

- **mean**: the mean residual over other contexts ending in the *same token*. This keeps the token's identity but
  removes context-specific information, which later layers could only recover by re-reading earlier positions.
- **resample**: the residual of one other real context ending in the same token. This is on-distribution, and it
  lets us ask how far the model moves back from the donor towards the true context.
- **global**: the mean over all tokens, which also deletes the token's identity. This is a contrast condition.

Main metric is **persistence**: the RMS size of the perturbation at layer l' divided by what was injected at layer
l. 1 means it simply rides along the residual stream, <1 means later layers repair it, >1 means they amplify it.
Raw relative L2 (|diff| / |activation|) would show fake "repair", because the residual norm grows with depth.
Also recorded:

- erasure: how much of the injected direction survives
- disrupted share: perturbation size vs. the natural spread of that token's activations
- per-block attention/MLP push-back
- KL divergence and change in next-token loss at the output
- for resample, the fraction of the way back from the donor to the clean activation

## Files

- `common.py`: model/data loading, hooks, and `run_patched`. The context is run once and its KV cache is shared by a
  batch of last-token rows: row 0 is clean, and each other row patches one (ablation, site).
- `means.py`: pass 1, per-token mean residuals over FineWeb positions [64, 512).
- `patch.py`: pass 2, the patching experiment on held-out docs (taken from the other end of the same parquet file).
- `plot.py`: heatmaps, cross-model line plots, and a text summary.
- `test_patch.py`: checks `run_patched` against a plain full forward pass with a hook (fp32).
- `run_all.sh`: the full GPU run.

## Running

```
python test_patch.py
python means.py --data data/fineweb_000.parquet --model Qwen/Qwen3-0.6B-Base --out out/means_0.6B.pt
python patch.py --data data/fineweb_000.parquet --model Qwen/Qwen3-0.6B-Base --means out/means_0.6B.pt --out out/metrics_0.6B.npz
python plot.py out/plots out/metrics_0.6B.npz
```
