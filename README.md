# act-experiments

## Forward Pass: every activation of a transformer, as squares

`viewer/index.html` is a static page that shows **every intermediate value GPT-2 small
computes** on a fixed prompt, from token embeddings to the final probabilities, as grids of
colored squares. Everything is precomputed; the page itself does no model arithmetic (it only
maps stored numbers to colors). The included trace runs the classic indirect-object-identification
prompt, *"When Mary and John went to the store, John gave a drink to"*, for which GPT-2 predicts
" Mary" (67.7%).

**Open it:** double-click `viewer/index.html` (it works from `file://`), or serve the folder with
`python3 -m http.server -d viewer` and visit <http://localhost:8000>.

### Reading the page

- Red squares are positive, blue negative, gray near zero. Probabilities and attention weights run
  from pale to dark orange. Each grid's legend gives its color limit. By default that limit is the
  99th percentile of |value|, so GPT-2's huge outlier activations (the first token's residual stream
  reaches about 3000) don't wash everything else out. Switch to **Full range** in the header to see
  them unclipped.
- Hover any square for its exact value and indices. Values are stored as float16, so expect 3–4
  significant digits.
- The outline walks the computation in order: tokens and embeddings, then for each of the 12
  layers LayerNorm → attention → LayerNorm → MLP, then the final LayerNorm, logits and predictions.
  **← / →** step between pages.
- There's one shared selection: a **query token**, a **key token** and an **attention head**.
  Clicking a square selects the token, head, or query/key pair it belongs to. You can also click the
  tokens in the header (shift-click, or the *Key* toggle, picks the key) or use the head buttons.
  Outlined squares show the selection, and the close-ups follow it.
- Weight matrices aren't shipped (GPT-2 small has 124M parameters); they appear as dashed
  placeholders with their shapes. Biases and LayerNorm parameters are shown.

### What the attention page shows

It opens with the story of one token, then lays out the full computation:

1. All 12 heads' attention patterns, so you can pick a head.
2. **What token i reads from the KV cache.** This is a table with one row per cached token
   j ≤ i. Each row shows that token's key, the query's score against it, the attention weight, its
   value, and the weighted value. The weighted values sum to the head's output z_i. The last column
   shows each token's *share of what's moved*, ‖A_ij · v_j · W_O‖ as a fraction of all rows: a token
   can get lots of attention yet hand over almost nothing because its value is tiny, which is
   GPT-2's usual way of parking attention on the first token. Use ◀ ▶ or *Play* to step through the
   prompt and watch the cache grow.
3. Every head's attention weights next to its shares of what's moved, for the same token, and what
   the selected head adds to that token's residual stream.
4. In full: q, k and v for all heads (`x·W + b`); the head's `q·kᵀ`, the `÷ √64` scaling, the causal
   mask (hatched squares are −∞) and softmax; close-ups of one score (`q_i ⊙ k_j` and its sum), one
   softmax row, and the value mixing; then `z`, the head's contribution through `W_O`, the sum over
   heads plus `b_O`, and the residual add.

The page computes all positions at once, as GPT-2 does when reading a prompt.
`tests/test_kv_cache.py` checks that this gives the same keys, values, patterns and logits as
running each prefix on its own, which is what generating with a KV cache computes.

## Exporting other prompts

```sh
pip install -r requirements.txt
python -m actviz --prompt "The Eiffel Tower is in the city of" --id gpt2-eiffel
python -m actviz --model gpt2-medium --prompt "..."     # also gpt2-large, gpt2-xl
```

Each export goes to `viewer/data/<id>/` and is added to `viewer/data/traces.json`. When there's
more than one, the viewer shows a picker. Useful flags: `--query-pos`, `--key-pos` and `--head`
set what's selected when the page opens (the bundled trace uses `--key-pos 2 --head 9`: head 9.9 is
a "name mover" that attends from "to" to " Mary"), and `--no-bos` drops the leading
`<|endoftext|>`.

The bundled 15-token trace is 31 MB, split into chunks of up to about 4 MB that load only when a page
needs them. The attention close-ups grow with tokens² (each query–key pair stores its 64 products)
and everything else grows with tokens, so keep prompts short.

## How it's built

```
actviz/                     exporter (Python: numpy, tokenizers, huggingface_hub)
  models/gpt2.py            GPT-2 forward pass in numpy, recording every intermediate
  models/components.py      reusable steps: LayerNorm, causal attention, residual add, top-k
  models/base.py            the ModelAdapter interface
  trace.py                  Trace: tensors + sections → pages → blocks; term helpers t/op/weight/row/grid
  writer.py                 serializes a Trace to viewer/data/
  hub.py                    HF Hub download + a small safetensors reader (handles bf16)
viewer/                     the static page: no build step, no dependencies besides Google Fonts
  js/core.js                helpers, OKLab color scales, data loading, float16 decoding
  js/heatmap.js             one tensor (or slice) drawn as squares; hover, click, selection outline
  js/blocks.js              renderers for each block type
  js/app.js                 outline, header, routing, shared selection
tests/
  test_gpt2_matches_hf.py   compares against Hugging Face's PyTorch GPT-2 (needs torch + transformers)
  test_kv_cache.py          checks all-at-once matches one-token-at-a-time (prefix) runs
  test_writer.py            round-trips the data format
```

The viewer knows nothing about GPT-2. The exporter decides what to show and how to arrange it,
and the page renders that description. To support a new model you write an exporter adapter and
leave the viewer alone.

### Data format

Every data file is a script that calls `ACTVIZ.receive(kind, key, payload)` instead of a JSON
file read with `fetch`. Browsers refuse `fetch` on `file://` URLs but will load a `<script>`, so
the page opens straight from disk.

- `manifest.js` holds the model config, tokens, axis labelling, the page layout, and per-tensor
  metadata: shape, named axes, color kind, stats for the color scale, and the chunk and byte offset
  where its data lives. A tensor can also be an alias for a slice of another (`of` + `at`), which is
  how each residual-stream snapshot points into one `[stage, pos, d_model]` stack without storing
  the data twice.
- `<chunk>.js` holds base64 of little-endian float16 data (float32 for tensors of 4096 elements or
  fewer), with each tensor aligned to 8 bytes.
- `labels-<name>.js` holds big label lists, such as the 50,257 vocabulary strings used in logit
  tooltips.

### Adding a model

1. Write an adapter in `actviz/models/<name>.py` that subclasses `ModelAdapter`, with `load`,
   `tokenize` and `trace` methods. `hub.Checkpoint` reads single-file or sharded safetensors,
   including bf16.
2. In `trace`, run the forward pass in numpy. Record each intermediate with
   `tr.add(id, array, axes, chunk=...)`, and describe the walk-through with
   `tr.section(...).page(...)` plus `page.equation(title, *lines)`. Reuse
   `components.layernorm`, `components.attention` and `components.residual_add` where they fit.
3. Register it in `actviz/models/__init__.py`, then add a test against a reference implementation,
   as `tests/test_gpt2_matches_hf.py` does.

A Llama-style model would need a few new components written the same way: RMSNorm; RoPE (an
equation showing q and k before and after rotation); grouped-query attention (repeat each k/v
head for the query heads that share it before calling `attention`); and SwiGLU (two up-projections,
SiLU and an elementwise product). The existing `fnplot` block can plot SiLU exactly as it plots GELU.

**Equation terms** (from `actviz.trace`):

| helper | renders as |
|---|---|
| `t(tensor, label, rows=, cols=, fix=, ...)` | a tensor as squares. `rows`/`cols` choose the axes (several axes flatten, with gaps between groups); `fix` pins an axis to an index or to a selection variable `"$pos"`, `"$kpos"` or `"$head"`; `select` overrides which variable a click sets; `values=True` prints numbers beside a column; `scale="name"` shares a color scale between terms; `norm="row"`/`"range"` colors each row on its own scale; `cell`, `wrap` and `labels` tune the layout |
| `op("÷ 8")` | an operator or connective |
| `weight("W_Q", [768, 768])` | a placeholder for a matrix that isn't shipped |
| `row(...)`, `grid([...], [...])` | terms side by side, or in aligned columns |

Labels and titles can include `{pos}`, `{kpos}` and `{head}`, which update with the selection.

**Blocks** (`page.block(type, ...)`): `text` (markdown-ish prose), `equation`, `tokens`,
`multiples` (one tensor split into small multiples along an axis, used for the head grids),
`cache` (a table with one row per token j up to the selected one; each column is a term using
`"$j"`, shown as squares, a `number` or a `bar`, with optional `query`/`total` rows),
`topk` and `predictions` (read `trace.extra["topk"]`), and `fnplot` (a 1-D function with the actual
inputs and outputs plotted on it).

**Color kinds** (`kind=` in `tr.add`): `signed` (diverging around 0), `positive` (0 to a robust
max) and `prob` (fixed 0–1).

## Tests

```sh
pip install pytest torch transformers   # torch/transformers only for the HF comparison
python -m pytest tests/
```

The numpy forward pass matches Hugging Face's `GPT2LMHeadModel` to float32 rounding (largest
logit difference about 6e-5). The test also checks that the decompositions shown on the page
add back up to what the model used (Σ q⊙k = q·k, Σ_j A_ij·v_j = z, Σ_h head outputs + b_O = attention
output).
