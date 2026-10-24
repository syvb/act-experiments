"""The page computes every position at once. Check that this gives the same numbers as
generating one token at a time with a KV cache, as the attention page says: the keys and
values cached for earlier tokens, and every later result, don't depend on what comes after."""

import numpy as np

from actviz.__main__ import DEFAULT_PROMPT
from actviz.models import GPT2


def test_prefix_matches_full_run():
    model = GPT2.load("openai-community/gpt2")
    ids = model.tokenize(DEFAULT_PROMPT)
    full = model.trace(ids, DEFAULT_PROMPT).tensors
    for n in (1, 5, len(ids) - 1):
        part = model.trace(ids[:n], DEFAULT_PROMPT).tensors
        for L in (0, 6, 11):
            for name in ("k", "v", "q", "z"):
                a, b = part[f"L{L}.attn.{name}"].data, full[f"L{L}.attn.{name}"].data[:n]
                np.testing.assert_allclose(a, b, atol=2e-4, err_msg=f"L{L} {name}, prefix {n}")
            a, b = part[f"L{L}.attn.pattern"].data, full[f"L{L}.attn.pattern"].data[:, :n, :n]
            np.testing.assert_allclose(a, b, atol=1e-5, err_msg=f"L{L} pattern, prefix {n}")
        np.testing.assert_allclose(part["logits"].data, full["logits"].data[:n], atol=2e-3)
