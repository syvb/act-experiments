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
| `judge/` | Haiku judge (`judge.mjs`), its 1,380 judgments (`judgments.jsonl`) and analysis (`analyze.mjs` → `summary.md`, `summary.json`) |
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

# Haiku judge (needs ANTHROPIC_API_KEY; 1,380 calls to claude-haiku-5-5, $0.16 for the 200-head run)
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

## Haiku judge results (200 heads)

Full tables, including splits by layer, attention pattern, AR match rank and head norm, are in `judge/summary.md`.

| | Rubric 0–3, true passage | Rubric, wrong passage | Any matching element, true / wrong passage | Forced choice | Balanced forced choice |
|---|---|---|---|---|---|
| Layer-20 residual | 1.65 | 0.00 | 20/20 / 0/20 | 20/20 | 100% |
| Heads, raw | 0.04 | 0.01 | 10/200 / 5/200 | 115/200 (57%) | 59% |
| Heads, mean+ | 0.07 | 0.01 | 20/200 / 6/200 | 115/200 (57%) | 56% |
| Random, raw | 0.05 | 0.00 | 1/20 / 0/20 | 10/20 | 46% |
| Random, mean+ | 0.00 | 0.00 | 0/20 / 0/20 | 9/20 | 40% |

The judge agrees with the AR: head decodes carry almost no information about their passage. No head decode scored
above 2, and only one (L13H16 on the airlock passage, mean+) reached 2. The forced choice is the most sensitive
measure and is a little above chance: over all 400 head decodes, Fisher's exact test of the judge's choice against the
true position gives p = 0.001. Raw heads in layers 7–13 do best (44/61, 74% balanced, p = 0.0002). Whether a head's
raw and mean+ decodes are judged right is correlated (both right for 74 heads, both wrong for 44, split for 82), so
some heads carry a little information. What gets through is broad topic words, such as "wildfire", "airlock",
"sink" or "Python".

Two judge caveats:

- Haiku picks passage A 63% (raw) and 72% (mean+) of the time on head decodes. Balanced accuracy averages the hit
  rates when the true passage is A and when it is B, so this bias can't move it off 50% by itself.
- The rubric score runs low. Residual decodes that name the marked final token (" sun", " faire") or the dish
  ("lentil soup") got 1, though the rubric gives 3 for the final token and 2 for a specific element. The listed
  matching elements and the true-vs-wrong-passage difference are more reliable than the absolute score.
