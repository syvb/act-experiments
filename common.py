"""Shared helpers: model/data loading, residual-stream hooks, and the patched forward pass."""
import contextlib
import copy

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MIN_POS = 64   # skip early positions (Qwen3 has no BOS; the first tokens act as attention sinks)
MAX_LEN = 512  # docs are truncated to this; means and test positions both come from [MIN_POS, MAX_LEN)


def load_model(name, device, dtype=torch.bfloat16):
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 should mean fp32 (TF32 would add rounding noise)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return tok, model


def iter_docs(parquet_path, tok, reverse=False):
    """Yield FineWeb docs as token-id lists truncated to MAX_LEN.
    The means pass reads from the start of the file, test samples from the end (reverse=True). The file has ~1M docs;
    the means pass uses ~35k for 10M tokens and the test set 10k, so they never overlap."""
    f = pq.ParquetFile(parquet_path)
    groups = range(f.num_row_groups)
    for g in reversed(groups) if reverse else groups:
        texts = f.read_row_group(g, columns=["text"]).column("text").to_pylist()
        for text in reversed(texts) if reverse else texts:
            ids = tok(text[: MAX_LEN * 16]).input_ids[:MAX_LEN]  # pre-cut very long docs before tokenizing
            if len(ids) >= MIN_POS + 2:
                yield ids


def residual_sites(model):
    """Modules whose outputs are the residual stream. Site 0 = token embeddings, site i = after i blocks."""
    return [model.model.embed_tokens, *model.model.layers]


@contextlib.contextmanager
def hooks(modules, fn):
    """Call fn(i, output) after modules[i] runs; if fn returns a tensor, it replaces the output.
    Attention modules return (output, weights): fn sees only the output and must not replace it."""

    def make(i):
        def hook(module, args, out):
            return fn(i, out[0] if isinstance(out, tuple) else out)

        return hook

    handles = [m.register_forward_hook(make(i)) for i, m in enumerate(modules)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def last_position_resid(model, ids):
    """Clean residual stream at the last position of `ids`, at every site: [n_sites, d]."""
    out = {}

    def record(site, h):
        out[site] = h[0, -1].float()

    with hooks(residual_sites(model), record):
        model.model(input_ids=torch.tensor([ids], device=model.device))
    return torch.stack([out[s] for s in range(len(out))])


@torch.no_grad()
def prefix_cache(model, prefix):
    """KV cache of `prefix` (one row), to share between run_patched calls on the same context."""
    return model.model(input_ids=torch.tensor([prefix], device=model.device), use_cache=True).past_key_values


@torch.no_grad()
def run_patched(model, cache, token, patch_site, patch_vec, add=False):
    """Run `token` after the prefix in `cache` (from prefix_cache), once per row of a batch. Row r has its residual
    stream at site patch_site[r] (at this last position only) replaced by patch_vec[r], or with add=True has
    patch_vec[r] added to it; patch_site = -1 leaves the row clean.

    The prefix is unaffected by the patch (causal attention), so its KV cache is computed once and shared by all rows.
    Returns residuals [B, n_sites, d], attention and MLP outputs [B, n_layers, d], and logits [B, vocab].
    """
    B = len(patch_site)
    layers = model.model.layers
    cache = copy.deepcopy(cache)  # the forward pass below extends the cache, so leave the caller's copy untouched
    cache.batch_repeat_interleave(B)
    resid, attn, mlp = {}, {}, {}

    def patch(site, h):
        h = h.clone()
        new = h[:, -1] + patch_vec.to(h.dtype) if add else patch_vec.to(h.dtype)
        h[:, -1] = torch.where((patch_site == site)[:, None], new, h[:, -1])
        resid[site] = h[:, -1].float()
        return h

    def record_attn(i, h):
        attn[i] = h[:, -1].float()

    def record_mlp(i, h):
        mlp[i] = h[:, -1].float()

    with (
        hooks(residual_sites(model), patch),
        hooks([l.self_attn for l in layers], record_attn),
        hooks([l.mlp for l in layers], record_mlp),
    ):
        tokens = torch.full((B, 1), token, device=model.device)
        logits = model(input_ids=tokens, past_key_values=cache).logits[:, -1].float()
    stack = lambda d: torch.stack([d[i] for i in range(len(d))], dim=1)
    return stack(resid), stack(attn), stack(mlp), logits
