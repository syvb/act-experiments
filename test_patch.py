"""Check that run_patched (shared KV cache, batched rows) matches a plain full forward pass with a hook, in fp32."""
import sys

import torch

from common import hooks, load_model, residual_sites, run_patched

model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B-Base"
tok, model = load_model(model_name, "cpu", dtype=torch.float32)
ids = tok("The quick brown fox jumps over the lazy dog. " * 8).input_ids[:70]
prefix, token = ids[:-1], ids[-1]
n_sites, d = model.config.num_hidden_layers + 1, model.config.hidden_size
layers = model.model.layers

torch.manual_seed(0)
patch_site = torch.tensor([-1, 0, 1, n_sites // 2, n_sites - 1])
patch_vec = torch.randn(len(patch_site), d)
batched = run_patched(model, prefix, token, patch_site, patch_vec)  # resid, attn, mlp, logits


@torch.no_grad()
def reference(site, vec):
    """Full forward pass over prefix + token, patching the last position at `site`."""
    resid, attn, mlp = {}, {}, {}

    def patch(s, h):
        if s == site:
            h = h.clone()
            h[0, -1] = vec
        resid[s] = h[0, -1]
        return h

    with (
        hooks(residual_sites(model), patch),
        hooks([l.self_attn for l in layers], lambda i, h: attn.update({i: h[0, -1]})),
        hooks([l.mlp for l in layers], lambda i, h: mlp.update({i: h[0, -1]})),
    ):
        logits = model(input_ids=torch.tensor([prefix + [token]])).logits[0, -1]
    stack = lambda x: torch.stack([x[i] for i in range(len(x))])
    return stack(resid), stack(attn), stack(mlp), logits


worst = 0.0
for row, site in enumerate(patch_site.tolist()):
    errors = []
    for name, got, want in zip(["resid", "attn", "mlp", "logits"], [x[row] for x in batched], reference(site, patch_vec[row])):
        errors.append(((got - want).norm(dim=-1) / want.norm(dim=-1)).max().item())
        print(f"patch site {site:3d}: max relative error {name:6s} {errors[-1]:.1e}")
    worst = max(worst, *errors)
assert worst < 1e-4, worst
print("OK")
