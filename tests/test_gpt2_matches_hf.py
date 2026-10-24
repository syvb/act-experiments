"""Check the numpy GPT-2 against Hugging Face's PyTorch implementation.

Needs torch and transformers (not required by the exporter itself):
    pip install torch transformers
    python -m pytest tests/   # or: python tests/test_gpt2_matches_hf.py
"""

import numpy as np

from actviz.__main__ import DEFAULT_PROMPT
from actviz.models import GPT2


def test_gpt2_matches_hf():
    import torch
    from transformers import GPT2LMHeadModel

    ours = GPT2.load("openai-community/gpt2")
    ids = ours.tokenize(DEFAULT_PROMPT)
    tr = ours.trace(ids, DEFAULT_PROMPT)
    T = tr.tensors

    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager").eval()
    with torch.no_grad():
        res = hf(torch.tensor([ids]), output_hidden_states=True, output_attentions=True)

    def close(name, a, b, tol):
        err = float(np.abs(a - b).max())
        print(f"{name:28s} max |diff| = {err:.2e}")
        assert err < tol, name

    n_layers = hf.config.n_layer
    for L in range(n_layers):
        close(f"L{L} resid_pre", T["resid"].data[2 * L], res.hidden_states[L][0].numpy(), 1e-3)
        close(f"L{L} pattern", T[f"L{L}.attn.pattern"].data, res.attentions[L][0].numpy(), 1e-5)
    close("ln_f out", T["ln_f.out"].data, res.hidden_states[n_layers][0].numpy(), 1e-3)
    close("logits", T["logits"].data, res.logits[0].numpy(), 2e-3)

    # The decompositions shown on the page must add back up to what the model used.
    for L in range(n_layers):
        p = f"L{L}.attn"
        close(f"L{L} Σ q⊙k = q·k", T[f"{p}.qk"].data.sum(-1), T[f"{p}.scores_raw"].data, 1e-3)
        close(f"L{L} Σ A·v = z", T[f"{p}.weighted"].data.sum(2).transpose(1, 0, 2), T[f"{p}.z"].data, 1e-4)
        close(f"L{L} Σ heads + b = out", T[f"{p}.head_out"].data.sum(0) + T[f"{p}.b_O"].data, T[f"{p}.out"].data, 1e-3)
    assert tr.extra["topk"][-1]["top"][0]["text"] == " Mary"


if __name__ == "__main__":
    test_gpt2_matches_hf()
    print("ok")
