"""Find a Tower Bridge feature in Qwen3.5-27B using the Qwen-Scope SAEs, on Modal.

    modal run tower_bridge.py::download   # cache model + all 64 SAE layers in a Volume (CPU only)
    modal run tower_bridge.py::run        # scan -> density check -> steering (one GPU container)
    modal run tower_bridge.py::steer --layer 30 --feature 1234 --strengths 50,100,200

Results are written to ./results/.
"""

import json
import os
import time
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.5-27B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100"
N_LAYERS = 64
D_SAE = 81920
TOP_K = 100

CACHE = "/cache"
MODEL_DIR = f"{CACHE}/models/Qwen3.5-27B"
SAE_DIR = f"{CACHE}/saes/{SAE_REPO.split('/')[1]}"

# Same no-thinking ChatML template the Qwen-Scope demo app uses.
NOTHINK_TEMPLATE = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

GPU = ["A100-80GB", "H100"]  # cheapest first; Modal falls back to the next if unavailable

app = modal.App("qwen-scope-tower-bridge")
volume = modal.Volume.from_name("qwen-scope-cache", create_if_missing=True, version=2)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cu126")
    .uv_pip_install(
        "transformers==5.20.2",
        "accelerate>=1.1.0",
        "huggingface_hub[hf_xet]>=1.31.0",
        "safetensors>=0.8.0",
        "pyarrow",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


# ── Downloads (CPU containers) ──────────────────────────────────────────────


@app.function(image=image, volumes={CACHE: volume}, cpu=4, memory=8192, timeout=3600)
def download_model():
    from huggingface_hub import snapshot_download

    t0 = time.time()
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*"],
        max_workers=16,
    )
    volume.commit()
    return time.time() - t0


@app.function(image=image, volumes={CACHE: volume}, cpu=1, memory=4096, timeout=3600, max_containers=16)
def download_sae_layer(layer: int):
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt", local_dir=SAE_DIR)
    volume.commit()
    return time.time() - t0


# ── GPU work ────────────────────────────────────────────────────────────────


class _Stop(Exception):
    pass


@app.cls(image=image, gpu=GPU, volumes={CACHE: volume}, cpu=4, memory=32768, timeout=3600, scaledown_window=60)
class Scope:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model, info = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, device_map="cuda", output_loading_info=True
        )
        self.model.eval().requires_grad_(False)  # attribution needs grads wrt activations only
        if info["missing_keys"]:
            raise RuntimeError(f"weights missing after load: {sorted(info['missing_keys'])[:10]}")
        self.layers = self.model.model.layers
        assert len(self.layers) == N_LAYERS
        self.load_secs = time.time() - t0
        self.gpu_name = torch.cuda.get_device_name()

    # -- helpers --

    def _capture(self, ids, layers):
        """Residual stream after each layer in `layers` -> {layer: [batch, seq, d]}.

        Stops the forward pass after the deepest requested layer.
        """
        import torch

        buf, hooks, last = {}, [], max(layers)
        for l in layers:

            def hook(mod, inp, out, l=l):
                buf[l] = (out[0] if isinstance(out, tuple) else out).detach()
                if l == last:
                    raise _Stop

            hooks.append(self.layers[l].register_forward_hook(hook))
        try:
            with torch.no_grad():
                self.model.model(input_ids=ids, use_cache=False)
        except _Stop:
            pass
        finally:
            for h in hooks:
                h.remove()
        return buf

    def _sae(self, layer, keys=("W_enc", "b_enc")):
        import torch

        # mmap so only the tensors we touch are read from the volume
        sae = torch.load(f"{SAE_DIR}/layer{layer}.sae.pt", map_location="cpu", mmap=True, weights_only=True)
        return {k: sae[k].to("cuda", torch.float32) for k in keys}

    @staticmethod
    def _encode(x, sae):
        """TopK-ReLU SAE encoder, as in the Qwen-Scope demo app. x: [n, d] -> [n, d_sae]."""
        import torch

        pre = torch.relu(x.float() @ sae["W_enc"].T + sae["b_enc"])
        vals, idx = pre.topk(TOP_K, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, vals)

    def _decoder_logit_lens(self, vec, n=12):
        import torch

        with torch.no_grad():
            h = self.model.model.norm(vec.to(torch.bfloat16)[None, None])
            logits = self.model.lm_head(h)[0, 0].float()
        top = logits.topk(n).indices.tolist()
        return [self.tok.decode([t]) for t in top]

    # -- methods --

    @modal.method()
    def sanity(self):
        import torch

        ids = self.tok("The bridge with two towers next to the Tower of London is called", return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=8, do_sample=False)
        return {
            "gpu": self.gpu_name,
            "load_secs": self.load_secs,
            "completion": self.tok.decode(out[0, ids.shape[1]:]),
        }

    @modal.method()
    def scan(self, items, positive, neutral, layers=None, top_n=15, n_token_detail=3):
        """Contrastive scan: which features fire on Tower Bridge texts and nowhere else?"""
        import torch

        t0 = time.time()
        layers = list(range(N_LAYERS)) if layers is None else layers
        cats = [c for c, _ in items]
        pos = torch.tensor([c in positive for c in cats], device="cuda")
        neg = torch.tensor([c not in positive and c not in neutral for c in cats], device="cuda")
        cat_names = list(dict.fromkeys(cats))
        cat_idx = {c: torch.tensor([i for i, cc in enumerate(cats) if cc == c], device="cuda") for c in cat_names}

        # 1. One forward pass per text, capturing every layer. Drop token 0: its
        #    residual is an attention-sink outlier and fires junk features.
        per_layer = {l: [] for l in layers}
        tokens, offsets = [], [0]
        for _, text in items:
            ids = self.tok(text, return_tensors="pt").input_ids.cuda()
            buf = self._capture(ids, layers)
            for l in layers:
                per_layer[l].append(buf[l][0, 1:])
            tokens.append([self.tok.decode([t]) for t in ids[0, 1:].tolist()])
            offsets.append(offsets[-1] + ids.shape[1] - 1)
        t_fwd = time.time() - t0

        results = []
        for l in layers:
            x = torch.cat(per_layer[l])  # [T, d]
            del per_layer[l]
            sae = self._sae(l, keys=("W_enc", "b_enc", "W_dec"))
            acts = self._encode(x, sae)  # [T, d_sae]
            # per-text max activation of every feature -> [n_texts, d_sae]
            M = torch.stack([acts[a:b].max(0).values for a, b in zip(offsets[:-1], offsets[1:])])
            fires = (M > 0).float()
            recall = fires[pos].mean(0)
            fpr = fires[neg].mean(0)
            max_neg = M[neg].max(0).values
            # fraction of positives whose activation beats every negative: threshold-free separation
            sep = (M[pos] > max_neg).float().mean(0)
            score = recall - fpr
            # rank by score, break ties by separation then mean positive activation
            mean_pos = M[pos].mean(0)
            key = score * 1e6 + sep * 1e3 + mean_pos / (mean_pos.max() + 1e-9)
            top = key.topk(top_n).indices.tolist()

            resid_norm = x.float().norm(dim=-1).median().item()
            feats = []
            for rank, f in enumerate(top):
                d = sae["W_dec"][:, f]
                entry = {
                    "layer": l,
                    "feature": f,
                    "score": round(score[f].item(), 4),
                    "recall": round(recall[f].item(), 4),
                    "fpr": round(fpr[f].item(), 4),
                    "sep": round(sep[f].item(), 4),
                    "mean_pos": round(mean_pos[f].item(), 3),
                    "max_pos": round(M[pos, f].max().item(), 3),
                    "max_neg": round(max_neg[f].item(), 3),
                    "dec_norm": round(d.norm().item(), 4),
                    "per_category_fire_rate": {c: round(fires[cat_idx[c], f].mean().item(), 3) for c in cat_names},
                    "per_text_max": [round(v, 2) for v in M[:, f].tolist()],
                    "logit_lens": self._decoder_logit_lens(d),
                }
                if rank < n_token_detail:
                    entry["token_acts"] = [
                        [round(v, 2) for v in acts[a:b, f].tolist()] for a, b in zip(offsets[:-1], offsets[1:])
                    ]
                feats.append(entry)
            results.append({"layer": l, "resid_median_norm": round(resid_norm, 2), "top": feats})
            del sae, acts, M, x
            torch.cuda.empty_cache()

        return {
            "gpu": self.gpu_name,
            "load_secs": self.load_secs,
            "forward_secs": t_fwd,
            "scan_secs": time.time() - t0,
            "texts": [{"category": c, "text": t, "tokens": tok} for (c, t), tok in zip(items, tokens)],
            "layers": results,
        }

    def _wikitext_ids(self, max_tokens, seq_len):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"
        )
        text = "".join(pq.read_table(path).column("text").to_pylist())
        ids = self.tok(text, return_tensors="pt").input_ids[0][:max_tokens]
        n_seq = len(ids) // seq_len
        return ids[: n_seq * seq_len].view(n_seq, seq_len)

    @modal.method()
    def specificity(self, items, positive, neutral, layers=None, generic_tokens=100_000, seq_len=1024, batch=4, top_n=10):
        """Rank features by how often a Tower Bridge text beats the feature's activation on
        everything else: the contrastive negatives *and* ~100k tokens of WikiText.

        A feature whose strongest activations are on Tower Bridge scores 1; a feature that
        fires on Tower Bridge but fires harder on something else (e.g. "Carboniferous")
        scores 0, however selective it looked on the small contrastive corpus.
        """
        from concurrent.futures import ThreadPoolExecutor

        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        t0 = time.time()
        layers = list(range(N_LAYERS)) if layers is None else layers
        cats = [c for c, _ in items]
        pos = torch.tensor([c in positive for c in cats], device="cuda")
        neg = torch.tensor([c not in positive and c not in neutral for c in cats], device="cuda")
        cat_names = list(dict.fromkeys(cats))
        cat_idx = {c: torch.tensor([i for i, cc in enumerate(cats) if cc == c], device="cuda") for c in cat_names}

        # contrastive corpus -> GPU (small)
        per_layer = {l: [] for l in layers}
        tokens, offsets = [], [0]
        for _, text in items:
            ids = self.tok(text, return_tensors="pt").input_ids.cuda()
            buf = self._capture(ids, layers)
            for l in layers:
                per_layer[l].append(buf[l][0, 1:])
            tokens.append([self.tok.decode([t]) for t in ids[0, 1:].tolist()])
            offsets.append(offsets[-1] + ids.shape[1] - 1)

        # generic corpus -> CPU, bf16 (~1 GB per layer per 100k tokens)
        gids = self._wikitext_ids(generic_tokens, seq_len)
        gen = {l: torch.empty((gids.shape[0], seq_len - 1, self.model.config.hidden_size), dtype=torch.bfloat16) for l in layers}
        for s in range(0, gids.shape[0], batch):
            buf = self._capture(gids[s : s + batch].cuda(), layers)
            for l in layers:
                gen[l][s : s + batch] = buf[l][:, 1:].cpu()
        n_generic = gids.shape[0] * (seq_len - 1)
        t_fwd = time.time() - t0

        def load_cpu(l):  # read W_enc/b_enc off the volume in a background thread
            sae = torch.load(f"{SAE_DIR}/layer{l}.sae.pt", map_location="cpu", mmap=True, weights_only=True)
            return {"W_enc": sae["W_enc"].clone(), "b_enc": sae["b_enc"].clone()}

        results = []
        ahead = 4  # bounded prefetch: at most ~5 layers of W_enc held in RAM
        with ThreadPoolExecutor(ahead) as pool:
            futures = {l: pool.submit(load_cpu, l) for l in layers[:ahead]}
            for i, l in enumerate(layers):
                if i + ahead < len(layers):
                    futures[layers[i + ahead]] = pool.submit(load_cpu, layers[i + ahead])
                sae = {k: v.to("cuda", torch.float32) for k, v in futures.pop(l).result().items()}
                acts = self._encode(torch.cat(per_layer.pop(l)), sae)
                M = torch.stack([acts[a:b].max(0).values for a, b in zip(offsets[:-1], offsets[1:])])
                G = torch.zeros(D_SAE, device="cuda")
                fires = torch.zeros(D_SAE, device="cuda")
                g = gen.pop(l).view(-1, self.model.config.hidden_size)
                for s in range(0, g.shape[0], 4096):
                    a = self._encode(g[s : s + 4096].cuda(), sae)
                    G = torch.maximum(G, a.max(0).values)
                    fires += (a > 0).sum(0)
                    del a
                del g
                bar = torch.maximum(G, M[neg].max(0).values)  # strongest activation anywhere else
                spec = (M[pos] > bar).float().mean(0)
                recall = (M[pos] > 0).float().mean(0)
                mean_pos = M[pos].mean(0)
                key = spec * 1e3 + recall + mean_pos / (mean_pos.max() + 1e-9) * 1e-3
                top = key.topk(top_n).indices.tolist()
                feats = []
                for rank, f in enumerate(top):
                    entry = {
                        "layer": l,
                        "feature": f,
                        "spec": round(spec[f].item(), 4),
                        "recall": round(recall[f].item(), 4),
                        "fpr": round((M[neg, f] > 0).float().mean().item(), 4),
                        "mean_pos": round(mean_pos[f].item(), 3),
                        "max_pos": round(M[pos, f].max().item(), 3),
                        "max_neg": round(M[neg, f].max().item(), 3),
                        "generic_max": round(G[f].item(), 3),
                        "generic_density": fires[f].item() / n_generic,
                        "per_category_fire_rate": {c: round((M[cat_idx[c], f] > 0).float().mean().item(), 3) for c in cat_names},
                        "per_text_max": [round(v, 2) for v in M[:, f].tolist()],
                    }
                    if rank < 3:
                        entry["token_acts"] = [
                            [round(v, 2) for v in acts[a:b, f].tolist()] for a, b in zip(offsets[:-1], offsets[1:])
                        ]
                    feats.append(entry)
                results.append({"layer": l, "top": feats})
                del sae, acts, M
                torch.cuda.empty_cache()

        return {
            "gpu": self.gpu_name,
            "load_secs": self.load_secs,
            "forward_secs": t_fwd,
            "secs": time.time() - t0,
            "generic_tokens": n_generic,
            "texts": [{"category": c, "text": t, "tokens": tok} for (c, t), tok in zip(items, tokens)],
            "layers": results,
        }

    def _split_ids(self, context, target):
        """Token ids of context + target, and how many trailing tokens are the target."""
        ctx = self.tok(context).input_ids
        full = self.tok(context + target).input_ids
        assert full[: len(ctx)] == ctx, "target merged with the context's last token"
        return full, len(full) - len(ctx)

    def _row_logps(self, rows, seqs):
        """Summed log-prob of each sequence's last n tokens, once per steering row.

        rows: list of (layer, vector) or None (unsteered); every row sees the same sequence,
        so a batch needs no padding. seqs: list of (ids, n). Returns [len(rows), len(seqs)].
        """
        import torch

        d = self.model.config.hidden_size
        per_layer = {}
        for i, r in enumerate(rows):
            if r is not None:
                l, v = r
                per_layer.setdefault(l, torch.zeros(len(rows), d, device="cuda", dtype=torch.bfloat16))[i] = v

        def make_hook(V):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h + V[:, None, :]
                return (h, *out[1:]) if isinstance(out, tuple) else h

            return hook

        hooks = [self.layers[l].register_forward_hook(make_hook(V)) for l, V in per_layer.items()]
        res = torch.zeros(len(rows), len(seqs))
        try:
            for j, (ids, n) in enumerate(seqs):
                x = torch.tensor([ids], device="cuda").repeat(len(rows), 1)
                with torch.no_grad():
                    logits = self.model(input_ids=x, use_cache=False, logits_to_keep=n + 1).logits[:, :-1].float()
                res[:, j] = logits.log_softmax(-1).gather(-1, x[:, -n:, None])[..., 0].sum(-1).cpu()
        finally:
            for h in hooks:
                h.remove()
        return res

    @modal.method()
    def output_features(
        self,
        attr_prompts,
        target,
        control_prompts,
        norms,
        negatives=(),
        per_layer=5,
        fractions=(0.3, 0.5),
        batch=16,
        max_control_drop=0.5,
        watch=((25, 241), (23, 39366)),
    ):
        """Find SAE features that make the model *say* `target` (output features).

        Objective J = log P(target) - mean log P(negatives), averaged over attribution prompts
        (with no negatives, J = log P(target)).
        1. Attribution: dJ / d resid, summed over positions, for every layer in one backward
           pass per (prompt, continuation). Adding s * W_dec[:, f] at every position changes J
           by about s * (grad . W_dec[:, f]), so grad @ W_dec scores all features at once.
        2. Real effect: steer with each layer's top `per_layer` features at `fractions` of the
           layer's median residual norm; measure the gain in J and in log P(target), and the
           per-token log-prob drop on the model's own answers to unrelated control prompts.
        """
        from concurrent.futures import ThreadPoolExecutor

        import torch

        t0 = time.time()
        layers = list(range(N_LAYERS))
        d = self.model.config.hidden_size
        conts = [target, *negatives]
        weights = [1.0] + [-1.0 / len(negatives) for _ in negatives]

        # -- 1. attribution --
        grads = {l: torch.zeros(d, device="cuda") for l in layers}
        attr_seqs, attr_w = [], []  # prompt-major, then continuation (target first)
        for user, prefix in attr_prompts:
            for cont, w in zip(conts, weights):
                attr_seqs.append(self._split_ids(NOTHINK_TEMPLATE.format(content=user) + prefix, cont))
                attr_w.append(w)
        for (ids, n), w in zip(attr_seqs, attr_w):
            x = torch.tensor([ids], device="cuda")
            outs = {}
            hooks = [
                self.layers[l].register_forward_hook(
                    lambda m, i, o, l=l: outs.__setitem__(l, o[0] if isinstance(o, tuple) else o)
                )
                for l in layers
            ]
            try:
                with torch.enable_grad():
                    emb = self.model.model.embed_tokens(x).detach().requires_grad_(True)
                    logits = self.model(inputs_embeds=emb, use_cache=False, logits_to_keep=n + 1).logits[0, :-1]
                    lp = logits.float().log_softmax(-1).gather(-1, x[0, -n:, None]).sum()
                    gs = torch.autograd.grad(lp, [outs[l] for l in layers])
            finally:
                for h in hooks:
                    h.remove()
            for l, g in zip(layers, gs):
                grads[l] += w * g[0].float().sum(0)
        t_attr = time.time() - t0

        def load_dec(l):
            sae = torch.load(f"{SAE_DIR}/layer{l}.sae.pt", map_location="cpu", mmap=True, weights_only=True)
            return sae["W_dec"].clone()

        attribution, vecs = [], {}
        ahead = 4
        with ThreadPoolExecutor(ahead) as pool:
            futures = {l: pool.submit(load_dec, l) for l in layers[:ahead]}
            for i, l in enumerate(layers):
                if i + ahead < len(layers):
                    futures[layers[i + ahead]] = pool.submit(load_dec, layers[i + ahead])
                W_dec = futures.pop(l).result().to("cuda", torch.float32)  # [d, d_sae]
                score = grads[l] @ W_dec  # first-order d log P per unit of feature f added everywhere
                top = score.topk(20)
                rank = score.argsort(descending=True).argsort()  # rank of every feature
                for f in top.indices[:per_layer].tolist():
                    vecs[(l, f)] = W_dec[:, f].to(torch.bfloat16)
                attribution.append(
                    {
                        "layer": l,
                        "grad_norm": round(grads[l].norm().item(), 4),
                        "top": [
                            {
                                "feature": f,
                                "attr": round(s, 4),
                                "logit_lens": self._decoder_logit_lens(W_dec[:, f]) if j < per_layer else None,
                            }
                            for j, (f, s) in enumerate(zip(top.indices.tolist(), top.values.tolist()))
                        ],
                        "watched": {f"{wl}:{wf}": {"attr": round(score[wf].item(), 4), "rank": int(rank[wf])} for wl, wf in watch if wl == l},
                    }
                )
                del W_dec
        t_dec = time.time() - t0

        # -- 2. real effect of steering --
        ctl_seqs = []
        for p in control_prompts:
            ids = self.tok(NOTHINK_TEMPLATE.format(content=p), return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=40, do_sample=False)
            gen = out[0, ids.shape[1] :].tolist()
            if self.tok.eos_token_id in gen:
                gen = gen[: gen.index(self.tok.eos_token_id)]
            ctl_seqs.append((ids[0].tolist() + gen, len(gen)))
        ctl_len = torch.tensor([n for _, n in ctl_seqs], dtype=torch.float32)
        S = len(attr_seqs)
        W = torch.tensor(attr_w)
        is_target = torch.tensor([i % len(conts) == 0 for i in range(S)])
        base = self._row_logps([None], attr_seqs + ctl_seqs)[0]
        base_attr, base_ctl = base[:S], base[S:]

        rows = [(l, f, frac) for (l, f) in vecs for frac in fractions]
        effects = []
        for s in range(0, len(rows), batch):
            chunk = rows[s : s + batch]
            lp = self._row_logps([(l, vecs[(l, f)] * (frac * norms[l])) for l, f, frac in chunk], attr_seqs + ctl_seqs)
            delta = lp[:, :S] - base_attr
            obj = (delta * W).sum(1) / len(attr_prompts)
            gain = delta[:, is_target].mean(1)
            drop = ((base_ctl - lp[:, S:]) / ctl_len).mean(1)
            for (l, f, frac), o, g, dr in zip(chunk, obj.tolist(), gain.tolist(), drop.tolist()):
                effects.append(
                    {
                        "layer": l,
                        "feature": f,
                        "fraction": frac,
                        "objective_gain": round(o, 3),
                        "target_gain": round(g, 3),
                        "control_drop": round(dr, 3),
                    }
                )
        effects.sort(key=lambda e: e["objective_gain"] if e["control_drop"] < max_control_drop else -1e9, reverse=True)

        return {
            "gpu": self.gpu_name,
            "secs": time.time() - t0,
            "attr_secs": t_attr,
            "decoder_secs": t_dec - t_attr,
            "target": target,
            "negatives": list(negatives),
            "base_target_logp": [round(v, 3) for v in base_attr[is_target].tolist()],
            "attribution": attribution,
            "effects": effects,
        }

    @modal.method()
    def density(self, candidates, max_tokens=300_000, seq_len=1024, batch=4, n_examples=12, ctx=12):
        """Run candidates over WikiText-103 (test split): how often do they fire, and on what?"""
        import heapq

        import torch

        t0 = time.time()
        ids = self._wikitext_ids(max_tokens, seq_len)
        n_seq = ids.shape[0]

        layers = sorted({c["layer"] for c in candidates})
        saes = {l: self._sae(l) for l in layers}
        by_layer = {l: [c["feature"] for c in candidates if c["layer"] == l] for l in layers}
        counts = {(c["layer"], c["feature"]): 0 for c in candidates}
        heaps = {k: [] for k in counts}  # min-heaps of (act, seq, pos)
        n_tok = 0
        for s in range(0, n_seq, batch):
            chunk = ids[s : s + batch].cuda()
            buf = self._capture(chunk, layers)
            for l in layers:
                h = buf[l][:, 1:]  # skip each sequence's first token, as in the scan
                acts = self._encode(h.reshape(-1, h.shape[-1]), saes[l])[:, by_layer[l]]
                acts = acts.view(h.shape[0], h.shape[1], -1)
                for j, f in enumerate(by_layer[l]):
                    a = acts[..., j]
                    counts[(l, f)] += int((a > 0).sum())
                    vals, flat = a.flatten().topk(n_examples)
                    for v, fi in zip(vals.tolist(), flat.tolist()):
                        if v <= 0:
                            break
                        item = (v, s + fi // a.shape[1], fi % a.shape[1] + 1)
                        heap = heaps[(l, f)]
                        heapq.heappush(heap, item) if len(heap) < n_examples else heapq.heappushpop(heap, item)
            n_tok += chunk.numel() - chunk.shape[0]

        def context(seq, pos):
            row = ids[seq].tolist()
            left = self.tok.decode(row[max(0, pos - ctx) : pos])
            tok = self.tok.decode([row[pos]])
            right = self.tok.decode(row[pos + 1 : pos + 1 + ctx // 2])
            return f"{left}[[{tok}]]{right}"

        out = []
        for c in candidates:
            k = (c["layer"], c["feature"])
            out.append(
                {
                    "layer": c["layer"],
                    "feature": c["feature"],
                    "fires": counts[k],
                    "density": counts[k] / n_tok,
                    "top_examples": [
                        {"act": round(v, 2), "context": context(sq, p)} for v, sq, p in sorted(heaps[k], reverse=True)
                    ],
                }
            )
        return {"tokens": n_tok, "secs": time.time() - t0, "candidates": out}

    @modal.method()
    def steer(self, layer, feature, prompts, strengths, max_new_tokens=60):
        """Greedy generations with strength * W_dec[:, feature] added to the residual stream
        after `layer` at every position. Row 0 of each batch is the unsteered baseline."""
        import torch

        t0 = time.time()
        d = self._sae(layer, keys=("W_dec",))["W_dec"][:, feature].to(torch.bfloat16)
        s = torch.tensor([0.0, *strengths], device="cuda", dtype=torch.bfloat16)[:, None, None]

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + s[: h.shape[0]] * d
            return (h, *out[1:]) if isinstance(out, tuple) else h

        handle = self.layers[layer].register_forward_hook(hook)
        results = []
        try:
            for p in prompts:
                ids = self.tok(NOTHINK_TEMPLATE.format(content=p), return_tensors="pt").input_ids.cuda()
                ids = ids.repeat(len(strengths) + 1, 1)  # same prompt per row -> no padding needed
                with torch.no_grad():
                    out = self.model.generate(
                        ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False
                    )
                gens = self.tok.batch_decode(out[:, ids.shape[1] :], skip_special_tokens=True)
                results.append({"prompt": p, "outputs": dict(zip(["baseline", *map(str, strengths)], gens))})
        finally:
            handle.remove()
        return {"layer": layer, "feature": feature, "secs": time.time() - t0, "results": results}


# ── Local entrypoints ───────────────────────────────────────────────────────

RESULTS = Path(__file__).parent / "results"


def _save(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


@app.local_entrypoint()
def download():
    t0 = time.time()
    model_call = download_model.spawn()
    sae_secs = list(download_sae_layer.map(range(N_LAYERS)))
    model_secs = model_call.get()
    print(f"model: {model_secs:.0f}s; SAE layers: {sum(sae_secs):.0f} container-s total; wall {time.time() - t0:.0f}s")


def pick_candidates(scan, n=4, per_layer=2):
    # Rank by recall - fpr plus separation. (The first run ranked by recall - fpr
    # alone, which picked "collapsible/retractable" features that fire on every
    # Tower Bridge text but also, more strongly, on drawbridges.)
    flat = [f for L in scan["layers"] for f in L["top"]]
    flat.sort(key=lambda f: (f["score"] + f["sep"], f["mean_pos"]), reverse=True)
    picked, seen = [], {}
    for f in flat:
        if seen.get(f["layer"], 0) < per_layer:
            picked.append(f)
            seen[f["layer"]] = seen.get(f["layer"], 0) + 1
        if len(picked) == n:
            break
    return picked


@app.local_entrypoint()
def run(n_candidates: int = 4, steer_multipliers: str = "2,5,10"):
    from corpus import NEUTRAL, POSITIVE, STEER_PROMPTS, flat

    t0 = time.time()
    scope = Scope()
    print("sanity:", scope.sanity.remote())

    scan = scope.scan.remote(flat(), POSITIVE, NEUTRAL)
    _save("scan.json", scan)
    print(f"scan done in {scan['scan_secs']:.0f}s (model load {scan['load_secs']:.0f}s on {scan['gpu']})")
    print("best feature per layer (layer: feature score recall fpr sep):")
    for L in scan["layers"]:
        f = L["top"][0]
        print(f"  {L['layer']:2d}: {f['feature']:5d}  {f['score']:.2f} {f['recall']:.2f} {f['fpr']:.2f} {f['sep']:.2f}")

    cands = pick_candidates(scan, n=n_candidates)
    print("candidates:", [(c["layer"], c["feature"], c["score"]) for c in cands])

    dens = scope.density.remote([{"layer": c["layer"], "feature": c["feature"]} for c in cands])
    _save("density.json", dens)
    print(f"density done in {dens['secs']:.0f}s over {dens['tokens']} tokens")

    mults = [float(m) for m in steer_multipliers.split(",")]
    steers = []
    for c in cands:
        strengths = [round(m * c["max_pos"], 1) for m in mults]
        steers.append(scope.steer.remote(c["layer"], c["feature"], STEER_PROMPTS, strengths))
        print(f"steered L{c['layer']} F{c['feature']} in {steers[-1]['secs']:.0f}s")
    _save("steer.json", steers)
    print(f"total wall time {time.time() - t0:.0f}s")


@app.local_entrypoint()
def validate(candidates: str = "23:78006,8:15311", norm_fractions: str = "0.3,0.45,0.6,0.75,0.9", tag: str = "validate"):
    """Density check + steering for hand-picked "layer:feature" pairs from results/scan.json.

    Steering strengths are fractions of the layer's median residual-stream norm, which
    transfers across features better than multiples of the feature's own activation.
    """
    from corpus import STEER_PROMPTS

    t0 = time.time()
    scan = json.loads((RESULTS / "scan.json").read_text())
    norms = {L["layer"]: L["resid_median_norm"] for L in scan["layers"]}
    cands = [{"layer": int(l), "feature": int(f)} for l, f in (c.split(":") for c in candidates.split(","))]
    scope = Scope()
    dens = scope.density.remote(cands)
    _save(f"density_{tag}.json", dens)
    print(f"density done in {dens['secs']:.0f}s over {dens['tokens']} tokens")
    steers = []
    for c in cands:
        strengths = [round(float(x) * norms[c["layer"]], 1) for x in norm_fractions.split(",")]
        steers.append(scope.steer.remote(c["layer"], c["feature"], STEER_PROMPTS, strengths))
        print(f"steered L{c['layer']} F{c['feature']} in {steers[-1]['secs']:.0f}s")
    _save(f"steer_{tag}.json", steers)
    print(f"total wall time {time.time() - t0:.0f}s")


@app.local_entrypoint()
def specific(n_candidates: int = 3, norm_fractions: str = "0.3,0.45,0.6,0.75,0.9"):
    """Specificity ranking against WikiText -> density check -> steering for the winners."""
    from corpus import NEUTRAL, POSITIVE, STEER_PROMPTS, flat

    t0 = time.time()
    norms = {L["layer"]: L["resid_median_norm"] for L in json.loads((RESULTS / "scan.json").read_text())["layers"]}
    scope = Scope.with_options(memory=98304)()  # holds ~65 GB of WikiText residuals in RAM
    spec = scope.specificity.remote(flat(), POSITIVE, NEUTRAL)
    _save("specificity.json", spec)
    print(f"specificity done in {spec['secs']:.0f}s (forward {spec['forward_secs']:.0f}s) on {spec['gpu']}")
    feats = sorted(
        (f for L in spec["layers"] for f in L["top"]),
        key=lambda f: (f["spec"], f["recall"], f["mean_pos"]),
        reverse=True,
    )
    print("layer feature spec recall fpr mean_pos generic_max generic_density")
    for f in feats[:20]:
        print(
            f"  {f['layer']:2d} {f['feature']:5d} {f['spec']:.2f} {f['recall']:.2f} {f['fpr']:.2f}"
            f" {f['mean_pos']:6.2f} {f['generic_max']:6.2f} {f['generic_density']:.1e}"
        )
    cands, seen = [], set()
    for f in feats:
        if f["layer"] not in seen:
            cands.append({"layer": f["layer"], "feature": f["feature"]})
            seen.add(f["layer"])
        if len(cands) == n_candidates:
            break
    dens = scope.density.remote(cands)
    _save("density_specific.json", dens)
    steers = []
    for c in cands:
        strengths = [round(float(x) * norms[c["layer"]], 1) for x in norm_fractions.split(",")]
        steers.append(scope.steer.remote(c["layer"], c["feature"], STEER_PROMPTS, strengths))
    _save("steer_specific.json", steers)
    print(f"total wall time {time.time() - t0:.0f}s")


@app.local_entrypoint()
def outputs(n_generate: int = 5, gen_fractions: str = "0.2,0.3,0.4,0.5,0.6", contrastive: bool = True):
    """Option 2: pick features by their effect on the output, then test them by steering.

    --no-contrastive scores plain log P(" Tower Bridge"), which mostly finds "say 'tower'"
    features (results/*_outputs.json); the default subtracts log P of the negatives.
    """
    import re

    from corpus import ATTRIBUTION_NEGATIVES, ATTRIBUTION_PROMPTS, ATTRIBUTION_TARGET, CONTROL_PROMPTS, STEER_PROMPTS

    t0 = time.time()
    tag = "outputs_contrastive" if contrastive else "outputs"
    norms = {L["layer"]: L["resid_median_norm"] for L in json.loads((RESULTS / "scan.json").read_text())["layers"]}
    scope = Scope()
    res = scope.output_features.remote(
        ATTRIBUTION_PROMPTS,
        ATTRIBUTION_TARGET,
        CONTROL_PROMPTS,
        norms,
        negatives=ATTRIBUTION_NEGATIVES if contrastive else (),
        batch=32,
    )
    _save("output_features_contrastive.json" if contrastive else "output_features.json", res)
    print(f"attribution {res['attr_secs']:.0f}s, decoders {res['decoder_secs']:.0f}s, total {res['secs']:.0f}s on {res['gpu']}")
    print("baseline log P(target):", res["base_target_logp"])
    for L in res["attribution"]:
        if L["watched"]:
            print(f"  detector features at layer {L['layer']}: {L['watched']}")
    print("top real effects (layer feature fraction objective_gain target_gain control_drop):")
    for e in res["effects"][:15]:
        print(
            f"  {e['layer']:2d} {e['feature']:5d} {e['fraction']} {e['objective_gain']:+.2f}"
            f" {e['target_gain']:+.2f} {e['control_drop']:.2f}"
        )

    picked, per_layer = [], {}
    for e in res["effects"]:
        key = (e["layer"], e["feature"])
        if key in [(p["layer"], p["feature"]) for p in picked] or per_layer.get(e["layer"], 0) >= 2:
            continue
        picked.append(e)
        per_layer[e["layer"]] = per_layer.get(e["layer"], 0) + 1
        if len(picked) == n_generate:
            break

    steers = []
    tb = re.compile(r"tower\s+bridge", re.I)
    other_tower = re.compile(r"tower(?!\s+bridge)", re.I)  # "tower" anywhere else: the failure mode
    for p in picked:
        strengths = [round(float(x) * norms[p["layer"]], 1) for x in gen_fractions.split(",")]
        s = scope.steer.remote(p["layer"], p["feature"], STEER_PROMPTS, strengths)
        for name, rx in (("mention_rate", tb), ("other_tower_rate", other_tower)):
            s[name] = {
                k: round(sum(bool(rx.search(r["outputs"][k])) for r in s["results"]) / len(s["results"]), 2)
                for k in s["results"][0]["outputs"]
            }
        steers.append(s)
        print(f"steered L{p['layer']} F{p['feature']}: Tower Bridge {s['mention_rate']}  other tower {s['other_tower_rate']}")
    _save(f"steer_{tag}.json", steers)

    dens = scope.density.remote([{"layer": p["layer"], "feature": p["feature"]} for p in picked[:3]])
    _save(f"density_{tag}.json", dens)
    print(f"total wall time {time.time() - t0:.0f}s")


@app.local_entrypoint()
def steer(layer: int, feature: int, strengths: str, max_new_tokens: int = 60):
    from corpus import STEER_PROMPTS

    res = Scope().steer.remote(layer, feature, STEER_PROMPTS, [float(s) for s in strengths.split(",")], max_new_tokens)
    _save(f"steer_L{layer}_F{feature}.json", res)
    for r in res["results"]:
        print("##", r["prompt"])
        for k, v in r["outputs"].items():
            print(f"  [{k}] {v!r}")
