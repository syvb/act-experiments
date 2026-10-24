# Looking for a Tower Bridge feature in Qwen3.5-27B with Qwen-Scope

Model: [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) (bf16, transformers 5.20.2).
SAEs: [`Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100`](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100):
TopK (k=100), 81,920 features, one per layer on the residual stream after each of the 64 layers.
Everything runs on Modal (`tower_bridge.py`); raw outputs are in `results/`.

## Result

Two features reliably and specifically **detect** Tower Bridge. Neither **steers** the model
toward talking about it the way Golden Gate Claude did.

| Feature | What it is | Evidence |
|---|---|---|
| **L25 / F241** | Cleanest Tower Bridge detector | Fires on the token that completes the name: " Bridge", 桥, ッジ, мост, 브리지 (2.2–5.6). Covers 100% of English mentions and 7/8 other-language ones. On 297k tokens of WikiText it fired on only 4 tokens, max 2.3 (wrestling tag-team names like "The New **Day**"). Steering up to 0.66× the residual norm: the model still answers "Golden Gate Bridge". |
| **L23 / F39366** | Tower Bridge detector that stays active for a few tokens | Fires on 100% of English and other-language mentions, up to 5.1. Also fires on "draw**bridge**" in a description that never names the bridge. WikiText: 22 of 297k tokens, max 1.6. Steering at about 0.5× the norm: "*a tourist visiting **London**… I would choose the **Tower of London***" (seen in two runs). Stronger steering drifts to income inequality. |
| L6 / F67082, L8 / F15311 | Second word of a compound name | Strongest on "Tower **Bridge**", also "storm **surge**", "power **pop**". Steering just produces token glitches ("thethe"). |
| L19 / F45178, L20 / F30896, L21 / F22269, L24 / F20858, L38 / F31408 | "Collapsible / retractable / movable" | Fire on every Tower Bridge text but harder on a Dutch drawbridge and the Thames Barrier. WikiText top activations: "**collaps**ible", "**retract**able". Logit lens: retract, unfold, 升降 (raise and lower). |
| L23 / F78006 | Deep time / antiquity | The best-looking feature on the small contrastive corpus. WikiText top activations: "**Carbon**iferous", "330 million **years** ago" (up to 17.5, versus 3.7 on Tower Bridge). Steering pushes toward "ancient, mist-shrouded temples". |

Both detectors fire on "Tower Bridge *in Sacramento*", because at the " Bridge" token the model can't
yet know which bridge is meant. Neither fires on "*Sacramento's* Tower Bridge", so they look tied to
London's bridge rather than to the string. That rests on a single example.

## Method

1. **Contrastive scan** (`run`). 96 short texts (`corpus.py`):
   - Positives: Tower Bridge named in English, named in 7 other languages, and described without the name.
   - Hard negatives: London Bridge, the Tower of London, other London landmarks, other bridges, other towers, generic text (including cell towers and contract bridge).

   For every layer and feature, score = recall on positives − false-positive rate on negatives.
   Token 0 is dropped (attention-sink outlier).
2. **Specificity** (`specific`). For every feature, find its max activation over the negatives
   *and* over ~100k tokens of WikiText-103. Rank by the fraction of Tower Bridge texts that beat
   that max. Step 1 alone was misleading: the feature that "won" there turned out to mean
   Carboniferous. A feature means what its *strongest* activations mean, and a small corpus can't
   show you those.
3. **Validation** (`validate`, `density`). Top activations over 297k WikiText tokens, then
   steering: add `s · W_dec[:, f]` to the residual stream after the layer, at every position,
   with greedy decoding. Strengths are set as a fraction of the layer's median residual norm.
   Multiples of the feature's own max activation transfer badly between features. Outputs break
   down at around 0.75–1.0× the norm.

Why detection without steering? With 81,920 features, about 400× fewer than the 34M-feature SAE
behind Golden Gate Claude, the dictionary seems to hold "the name Tower Bridge just occurred"
features rather than a "think about Tower Bridge" concept. Untested next steps:
- Steer with several features or layers at once.
- Clamp the feature instead of adding to it.
- Try the wider-sparsity `L0_50` SAE.

## Cost (actual Modal bill for this experiment: $2.37)

| Step | Hardware | Wall time | Billed |
|---|---|---|---|
| `download` (55.6 GB model + 64 × 3.36 GB SAEs → Volume) | CPU only, 16 parallel containers | 2.8 min | $0.32 (incl. $0.19 egress, covered by allowance) |
| `run`: model load 44 s, scan of all 64 layers 22 min, density, steering | A100-80GB | 27 min | $1.33 |
| `validate` | A100-80GB | 3.4 min | $0.17 |
| `specific` (holds ~65 GB of WikiText activations in RAM) | A100-80GB, 96 GiB RAM | 7 min | $0.40 |
| `validate --tag final` | H100 (fallback) | 2.2 min | $0.16 |

Knowing the method up front, a clean rerun is about **$0.80**: download, then `specific` (5 min on
an A100), then one steering run. The 22-minute scan was slow because of SAE I/O: each layer is
3.36 GB of fp32, read via mmap from the Volume. `specificity` prefetches `W_enc` in background
threads and covers all 64 layers in about 3 minutes.

Storage: the `qwen-scope-cache` Volume holds about 270 GB, which is free within Modal's 1 TiB
allowance and $0.09/GiB-month beyond it. Remove it with `modal volume delete qwen-scope-cache`.

## Serving the model as a steering API (estimate)

Steering costs nothing measurable at inference time. It is one vector add at one layer: 5,120
numbers, against about 54 GFLOP/token for the model. The SAE isn't needed at serve time.
Reading feature activations per request would need one layer's encoder: 0.84 GB in bf16 and
0.84 GFLOP/token, about 1.6% overhead. **So the cost is the cost of serving Qwen3.5-27B.**

The engine matters most. This repo's HF transformers + hook harness uses reference PyTorch kernels
for the Gated DeltaNet layers and has no continuous batching. It measured about 10–15 tok/s per
sequence on an A100/H100, which is fine for experiments but not for an API.

Production would mean vLLM or SGLang with a small patch. Neither has a steering API I could
confirm; [EasySteer](https://arxiv.org/abs/2509.25175) is a vLLM-based steering framework. A
fixed vector is a constant add that survives CUDA graphs. Per-request vectors need per-sequence
indexing. Prefix caching must be off, or keyed by steering config, if the prompt is steered.

Modal list prices: H100 $3.95/h, A100-80GB $2.50/h, L40S $1.95/h, plus about $0.45/h for
4 CPU cores and 32 GiB RAM. Throughput below is a bandwidth-based estimate, not a vLLM measurement.

| Setup | Fits? | Single-stream decode (est.) |
|---|---|---|
| H100, bf16 (54 GB) | yes, about 20 GB left for cache | ~50 tok/s |
| H100, [FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) (31 GB) | yes | ~80–95 tok/s |
| A100-80GB, bf16 | yes | ~30 tok/s |
| L40S, FP8 | yes, about 14 GB left | ~25 tok/s |

Monthly cost depends on usage:
- **Personal / scale-to-zero** (H100 bf16, about 2–3 min cold start, 5 min idle window,
  10 sessions/week of about 6 min): about 10 GPU-hours/month, **≈ $40–45/month**. One warm
  request (500 in, 300 out) takes about 6 s, roughly 0.7¢.
- **Always on, one GPU:** ≈ $3,200/mo (H100 bf16), $2,150/mo (A100-80GB), $1,750/mo (L40S FP8).
- **At saturation** with continuous batching: very roughly $0.5–1.2 per million output tokens.
  Alibaba's hosted Qwen3.5-27B is about $0.30 in / $2.40 out per million, but can't steer.

FP8 steering vectors come from an SAE trained on bf16 activations. They should transfer, but
that is untested.

## Reproduce

```sh
pip install 'modal[api-proxy-support]'
modal run tower_bridge.py::download
modal run tower_bridge.py::run         # scan of all 64 layers (results/scan.json is needed below)
modal run tower_bridge.py::specific    # specificity ranking + density + steering
modal run tower_bridge.py::validate --candidates "25:241,23:39366" --norm-fractions "0.35,0.42,0.5,0.58,0.66" --tag final
```

`results/steer.json` and `results/density.json` come from the first `run`, which ranked by
recall − FPR only and so picked the "collapsible" features. `pick_candidates` now also uses
separation. The two steering prompts at the end of `STEER_PROMPTS` were added before the final
run only.
