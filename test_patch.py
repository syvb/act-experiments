"""Check that run_patched (shared KV cache, batched rows) matches a plain full forward pass with a hook, in fp32."""
import sys

import torch

from common import hooks, load_model, residual_sites, run_patched

model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B-Base"
tok, model = load_model(model_name, "cpu", dtype=torch.float32)
ids = tok("The quick brown fox jumps over the lazy dog. " * 8).input_ids[:70]
prefix, token = ids[:-1], ids[-1]
n_sites, d = model.config.num_hidden_layers + 1, model.config.hidden_size

torch.manual_seed(0)
patch_site = torch.tensor([-1, 0, 1, n_sites // 2, n_sites - 1])
patch_vec = torch.randn(len(patch_site), d)
resid, _, _, logits = run_patched(model, prefix, token, patch_site, patch_vec)


@torch.no_grad()
def reference(site, vec):
    """Full forward pass over prefix + token, patching the last position at `site`."""
    out = {}

    def patch(s, h):
        if s == site:
            h = h.clone()
            h[0, -1] = vec
        out[s] = h[0, -1]
        return h

    with hooks(residual_sites(model), patch):
        logits = model(input_ids=torch.tensor([prefix + [token]])).logits[0, -1]
    return torch.stack([out[s] for s in range(n_sites)]), logits


worst = 0.0
for row, site in enumerate(patch_site.tolist()):
    ref_resid, ref_logits = reference(site, patch_vec[row])
    err = ((resid[row] - ref_resid).norm(dim=-1) / ref_resid.norm(dim=-1)).max().item()
    err_logits = ((logits[row] - ref_logits).norm() / ref_logits.norm()).item()
    print(f"patch site {site:3d}: max relative error resid {err:.1e}, logits {err_logits:.1e}")
    worst = max(worst, err, err_logits)
assert worst < 1e-4, worst
print("OK")
