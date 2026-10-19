# Attention-head outputs through the layer-20 NLA

Feeds the outputs of individual attention heads of Qwen2.5-7B-Instruct into the activation verbalizer (AV) of the
[layer-20 Natural Language Autoencoder](https://transformer-circuits.pub/2026/nla/index.html)
([`kitft/nla-qwen2.5-7b-L20-av`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av)), and scores each explanation by
round-tripping it through the activation reconstructor (AR,
[`kitft/nla-qwen2.5-7b-L20-ar`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar)).

A head's output is its 128-dim slice of the attention output times its 128 columns of `o_proj`, so the 28 heads of a
layer sum to that layer's attention output. Each vector is rescaled to norm 150 and written over the `㈎` token
embedding in the AV prompt, then decoded greedily.

Two injection variants are decoded for every head:

- **raw**: the head output itself.
- **mean+**: the mean unit residual μ plus the head's direction with its μ component removed, scaled to the median
  off-mean size of a real unit residual. Real residuals share a large common direction (cosine ≈ 0.53 with μ) that
  head outputs lack.

Controls: the full layer-20 residual at the same token (positive), random Gaussian vectors (negative), and μ alone.

## Layout

| Path | What |
|---|---|
| `modal/nla_heads.py` | Modal app: extract head outputs, decode with the AV, score with the AR (one H100 run, ~1–2 min) |
| `modal/config_4texts.json` | 4 texts × 10 random heads (`results/results_v2.json`) |
| `modal/config_200.json` | 20 texts × 10 random heads = 200 heads (`results/results_200.json`) |
| `results/results_v1.json` | First run on the 4 texts: raw variant only, plus one T=1 sample per vector |
| `judge/` | Haiku judge (`judge.mjs`) and its analysis (`analyze.mjs`); not run yet |
| `report/` | Page and markdown report built from the 4-text run |

All texts were written for this experiment, so they can't be in the NLA's training data. Positions and heads are
picked by a seeded RNG: 8 heads from layers 0–20 (which write into the vector the NLA reads) and 2 from layers 21–27
per text.

## Running

```sh
# Modal (needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; in a proxied sandbox install modal[api-proxy-support])
cd modal
modal run nla_heads.py::download                       # weights -> Modal volume nla-hf-cache
modal run nla_heads.py::main --config config_200.json --out ../results/results_200.json
modal volume delete nla-hf-cache                       # the ~42 GB volume is billed for storage while it exists

# Haiku judge (needs ANTHROPIC_API_KEY; ~1,380 calls to claude-haiku-5-5, estimated $0.25–0.40)
cd judge
npm ci
node judge.mjs ../results/results_200.json --dry-run   # prints sample prompts, no API calls
node judge.mjs ../results/results_200.json             # appends to judgments.jsonl; rerun to retry failures
node analyze.mjs ../results/results_200.json judgments.jsonl   # writes summary.md / summary.json
```

`judge/mock_server.mjs` is a local stand-in for the Messages API, used to test `judge.mjs` without a key:
start it, then run the judge with `ANTHROPIC_BASE_URL=http://127.0.0.1:18765 ANTHROPIC_API_KEY=test JUDGE_OUT=mock.jsonl`.

The judge makes three calls per decode:

1. A 0–3 rubric score against the passage the vector came from.
2. The same rubric against a different passage, which measures how lenient the judge is.
3. A forced choice between the true passage and a distractor, presented in random order.

The judge sees only the passage up to the decoded token and the explanation. It never sees head metadata.

## Results so far (AR round-trip, 200 heads)

| | Mean-removed cosine, AR(explanation) vs. source | AR picks the right head out of 10 |
|---|---|---|
| Layer-20 residual | 0.833 | – |
| Heads, raw | 0.043 | 92/200 |
| Heads, mean+ | 0.062 | 130/200 |
| Random vectors, raw / mean+ | 0.037 / 0.046 | – |

Chance for the last column is 20/200. 32 of the 200 raw head decodes contain substantial Chinese text, against none of
the mean+ and residual decodes.
