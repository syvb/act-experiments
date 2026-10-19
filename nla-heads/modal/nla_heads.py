"""Feed individual attention-head outputs of Qwen2.5-7B-Instruct into the layer-20 NLA verbalizer.

modal run nla_heads.py::download      # one-time: fetch weights into a Volume (CPU only)
modal run nla_heads.py::main          # GPU run, writes results.json locally
"""

import json
import pathlib

import modal

app = modal.App("nla-heads")
vol = modal.Volume.from_name("nla-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "safetensors",
        "huggingface_hub[hf_xet]<1.0",
        "pyyaml",
        "numpy<2.3",
    )
    .env({"HF_HOME": "/cache/hf"})
)

BASE = "Qwen/Qwen2.5-7B-Instruct"
AV = "kitft/nla-qwen2.5-7b-L20-av"
AR = "kitft/nla-qwen2.5-7b-L20-ar"
NLA_LAYER = 20  # NLA reads the output of block 20 (HF hidden_states[21])


@app.function(image=image, volumes={"/cache": vol}, timeout=3600, cpu=4, memory=8192)
def download():
    from huggingface_hub import snapshot_download

    for repo in (BASE, AV, AR):
        p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.yaml", "*.jinja"])
        print(repo, "->", p)
    vol.commit()


@app.function(image=image, volumes={"/cache": vol}, gpu="H100", timeout=1800, memory=65536)
def run(cfg: dict) -> dict:
    import random
    import time

    import torch
    import yaml
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dev = "cuda"
    torch.set_grad_enabled(False)

    # ---------- target model: per-head outputs ----------
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=dev, attn_implementation="eager"
    ).eval()
    bc = base.config
    n_layers, n_heads = bc.num_hidden_layers, bc.num_attention_heads
    hd = bc.hidden_size // n_heads
    print(f"base loaded {time.time()-t0:.0f}s")

    def run_base(ids):
        cap = {}
        hooks = []
        for li, layer in enumerate(base.model.layers):
            attn = layer.self_attn

            def pre(mod, args, li=li):
                cap[("z", li)] = args[0][0].float()  # [T, H*hd], concat of per-head outputs pre-W_O

            def post_attn(mod, args, out, li=li):
                cap[("pattern", li)] = out[1][0].float()  # [H, T, T]

            def post_oproj(mod, args, out, li=li):
                cap[("attn_out", li)] = out[0].float()

            hooks.append(attn.o_proj.register_forward_pre_hook(pre))
            hooks.append(attn.o_proj.register_forward_hook(post_oproj))
            hooks.append(attn.register_forward_hook(post_attn))

        def post_layer(mod, args, out):
            cap["resid"] = (out[0] if isinstance(out, tuple) else out)[0].float()

        hooks.append(base.model.layers[NLA_LAYER].register_forward_hook(post_layer))
        out = base(torch.tensor([ids], device=dev), output_hidden_states=True)
        for h in hooks:
            h.remove()
        # sanity: our hook == HF hidden_states[21]
        hs = out.hidden_states[NLA_LAYER + 1][0].float()
        cap["hs_check"] = ((hs - cap["resid"]).norm() / cap["resid"].norm()).item()
        print("hidden_states[21] vs block-20 hook rel diff:", cap["hs_check"])
        return cap

    def head_vec(cap, li, h, pos):
        W = base.model.layers[li].self_attn.o_proj.weight[:, h * hd:(h + 1) * hd].float()  # [d, hd]
        return W @ cap[("z", li)][pos, h * hd:(h + 1) * hd]

    W_U = base.lm_head.weight.float()
    g_final = base.model.norm.weight.float()

    def logit_lens(v, k=8):
        # direct-logit readout of a single vector (ignores the final norm's scale; uses its gain)
        logits = W_U @ (g_final * v)
        top = torch.topk(logits, k).indices.tolist()
        bot = torch.topk(-logits, k).indices.tolist()
        return [tok.decode([t]) for t in top], [tok.decode([t]) for t in bot]

    def tokstr(t):
        return tok.decode([t])

    def encode(ex):
        if ex["kind"] == "chat":
            return tok.apply_chat_template([{"role": "user", "content": ex["text"]}], tokenize=True, add_generation_prompt=True)
        return tok(ex["text"], add_special_tokens=False)["input_ids"]

    # pool of unit-normalized layer-20 residuals, to estimate the mean direction every residual shares
    pool = []

    def add_to_pool(cap):
        r = cap["resid"]
        n = r.norm(dim=-1)
        keep = (torch.arange(len(r), device=dev) >= 4) & (n < 300)  # drop attention-sink / massive-norm positions
        pool.append(r[keep] / n[keep, None])

    for ex in cfg.get("mean_texts", []):
        add_to_pool(run_base(encode(ex)))

    items = []  # each: dict(meta..., vec)
    examples_out = []
    for ei, ex in enumerate(cfg["examples"]):
        ids = encode(ex)
        T = len(ids)
        rng = random.Random(cfg["seed"] * 1000 + ei)
        lo = ex.get("min_pos", 16)
        pos = ex.get("pos", rng.randrange(lo, T))
        cap = run_base(ids)
        add_to_pool(cap)

        # verify per-head decomposition sums to the o_proj output at the sampled position
        for li in (0, NLA_LAYER, n_layers - 1):
            s = sum(head_vec(cap, li, h, pos) for h in range(n_heads))
            ref = cap[("attn_out", li)][pos]
            rel = ((s - ref).norm() / ref.norm()).item()
            assert rel < 2e-2, f"head decomposition mismatch at L{li}: {rel}"

        resid = cap["resid"][pos]
        heads = set()
        while len(heads) < cfg["n_heads_early"]:
            heads.add((rng.randrange(0, NLA_LAYER + 1), rng.randrange(n_heads)))
        late = set()
        while len(late) < cfg["n_heads_late"]:
            late.add((rng.randrange(NLA_LAYER + 1, n_layers), rng.randrange(n_heads)))
        chosen = sorted(heads) + sorted(late)

        # norms of every head at this position, for context
        all_norms = torch.stack([
            torch.stack([head_vec(cap, li, h, pos).norm() for h in range(n_heads)]) for li in range(n_layers)
        ])  # [L, H]

        ex_rec = {
            "idx": ei,
            "name": ex["name"],
            "kind": ex["kind"],
            "n_tokens": T,
            "pos": pos,
            "token": tokstr(ids[pos]),
            "context": tok.decode(ids[: pos + 1]),
            "after": tok.decode(ids[pos + 1:]),
            "resid_norm": resid.norm().item(),
            "head_norm_median": all_norms.median().item(),
            "hs_check": cap["hs_check"],
        }
        examples_out.append(ex_rec)

        top_t, bot_t = logit_lens(resid)
        items.append({"ex": ei, "kind": "resid", "label": f"layer-{NLA_LAYER} residual (baseline)", "vec": resid,
                      "raw_norm": resid.norm().item(), "lens_top": top_t, "lens_bot": bot_t})
        for li, h in chosen:
            v = head_vec(cap, li, h, pos)
            patt = cap[("pattern", li)][h, pos, : pos + 1]
            k = min(4, pos + 1)
            w, src = torch.topk(patt, k)
            top_t, bot_t = logit_lens(v)
            rank = int((all_norms[li] > v.norm()).sum().item()) + 1
            items.append({
                "ex": ei, "kind": "head", "layer": li, "head": h, "label": f"L{li}H{h}",
                "vec": v, "raw_norm": v.norm().item(), "norm_rank_in_layer": rank,
                "cos_with_resid": torch.nn.functional.cosine_similarity(v, resid, dim=0).item(),
                "attn_top": [{"src_pos": int(s), "tok": tokstr(ids[int(s)]), "w": float(ww)} for ww, s in zip(w.tolist(), src.tolist())],
                "lens_top": top_t, "lens_bot": bot_t,
            })

    # controls: random gaussian directions
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    for i in range(cfg["n_random"]):
        v = torch.randn(bc.hidden_size, generator=gen).to(dev)
        items.append({"ex": None, "kind": "random", "label": f"random gaussian #{i}", "vec": v, "raw_norm": v.norm().item()})

    # mean direction of unit residuals, and how big the per-token deviation from it typically is
    U = torch.cat(pool)
    mu = U.mean(0)
    mu_hat = mu / mu.norm()

    def perp(v):
        v = v / v.norm()
        return v - (v @ mu_hat) * mu_hat

    dev_norm = torch.stack([perp(u).norm() for u in U]).median().item()
    sub = U[torch.randperm(len(U), generator=torch.Generator().manual_seed(0))[:300].to(dev)]
    C = sub @ sub.T
    pool_stats = {
        "n_pool": len(U),
        "mu_norm_unit": mu.norm().item(),
        "mean_cos_resid_mu": (U @ mu_hat).mean().item(),
        "mean_pairwise_cos_resid": ((C.sum() - C.diag().sum()) / (len(sub) * (len(sub) - 1))).item(),
        "dev_norm_unit": dev_norm,
    }
    print("pool stats", pool_stats)

    # "mean+": keep only the part of the vector orthogonal to the shared mean direction, and put it on top
    # of the mean at the typical size of a real residual's deviation
    def mean_plus(v):
        p = perp(v)
        return mu + dev_norm * p / p.norm()

    items.append({"ex": None, "kind": "mean", "label": "mean residual direction (control)", "vec": mu.clone(), "raw_norm": mu.norm().item()})
    for it in items:
        it["cos_with_mu"] = torch.nn.functional.cosine_similarity(it["vec"], mu, dim=0).item()
    for it in list(items):
        if it["kind"] == "mean":
            continue
        items.append({**{k: v for k, v in it.items() if k not in ("vec",)}, "variant": "mean+", "vec": mean_plus(it["vec"]), "src_vec": it["vec"]})
        it["variant"] = "raw"
        it["src_vec"] = it["vec"]
    items[[i for i, it in enumerate(items) if it["kind"] == "mean"][0]].update(variant="raw", src_vec=mu.clone())

    del base
    torch.cuda.empty_cache()
    print(f"{len(items)} vectors extracted {time.time()-t0:.0f}s")

    # ---------- AV: verbalize ----------
    av_dir = snapshot_download(AV)
    meta = yaml.safe_load(open(f"{av_dir}/nla_meta.yaml"))
    av_tok = AutoTokenizer.from_pretrained(av_dir)
    av = AutoModelForCausalLM.from_pretrained(av_dir, torch_dtype=torch.bfloat16, device_map=dev).eval()
    tk = meta["tokens"]
    assert av_tok.convert_tokens_to_ids(av_tok.tokenize(tk["injection_char"])) == [tk["injection_token_id"]]
    content = meta["prompt_templates"]["av"].format(injection_char=tk["injection_char"])
    p_ids = av_tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
    inj = [i for i, t in enumerate(p_ids) if t == tk["injection_token_id"]
           and p_ids[i - 1] == tk["injection_left_neighbor_id"] and p_ids[i + 1] == tk["injection_right_neighbor_id"]]
    assert len(inj) == 1, inj
    inj = inj[0]
    scale = meta["extraction"]["injection_scale"]
    base_emb = av.get_input_embeddings()(torch.tensor([p_ids], device=dev))  # [1, P, d] bf16
    print(f"AV loaded {time.time()-t0:.0f}s; prompt len {len(p_ids)}, inject at {inj}, scale {scale}")

    def verbalize(vecs, do_sample, temperature=1.0, seed=0):
        outs = []
        bs = cfg["batch_size"]
        for i in range(0, len(vecs), bs):
            chunk = torch.stack(vecs[i:i + bs])
            chunk = chunk / chunk.norm(dim=-1, keepdim=True) * scale
            emb = base_emb.repeat(len(chunk), 1, 1).clone()
            emb[:, inj] = chunk.to(emb.dtype)
            torch.manual_seed(seed)
            gen_ids = av.generate(
                inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=dev),
                max_new_tokens=cfg["max_new_tokens"], do_sample=do_sample,
                **({"temperature": temperature, "top_p": 1.0, "top_k": 0} if do_sample else {}),
                pad_token_id=av_tok.pad_token_id,
            )
            for row in gen_ids:
                outs.append(av_tok.decode(row, skip_special_tokens=True))
        return outs

    vecs = [it["vec"] for it in items]
    greedy = verbalize(vecs, do_sample=False)
    print(f"greedy done {time.time()-t0:.0f}s")
    sampled = verbalize(vecs, do_sample=True, temperature=1.0, seed=cfg["seed"]) if cfg["also_sample"] else [None] * len(vecs)
    print(f"sampled done {time.time()-t0:.0f}s")
    for it, g, s in zip(items, greedy, sampled):
        it["greedy"] = g
        it["sampled_t1"] = s

    del av
    torch.cuda.empty_cache()

    # ---------- AR: reconstruct, score ----------
    ar_dir = snapshot_download(AR)
    ar_meta = yaml.safe_load(open(f"{ar_dir}/nla_meta.yaml"))
    ar_tok = AutoTokenizer.from_pretrained(ar_dir)
    ar = AutoModelForCausalLM.from_pretrained(ar_dir, torch_dtype=torch.bfloat16, device_map=dev).eval()
    ar.model.norm = torch.nn.Identity()
    vh = torch.nn.Linear(bc.hidden_size, bc.hidden_size, bias=False, dtype=torch.bfloat16, device=dev)
    vh.load_state_dict(load_file(f"{ar_dir}/value_head.safetensors"))
    import re
    ex_re = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.S)

    def reconstruct(text):
        m = ex_re.search(text)
        expl = m.group(1) if m else text
        prompt = ar_meta["prompt_templates"]["ar"].format(explanation=expl)
        ids = ar_tok(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
        h = ar.model(ids, use_cache=False).last_hidden_state[0, -1]
        return vh(h).float()

    cos = torch.nn.functional.cosine_similarity

    def cos_perp(a, b):
        # similarity of the parts that are not the shared mean direction
        return cos(perp(a), perp(b), dim=0).item()

    for it in items:
        rec = reconstruct(it["greedy"])
        it["recon"] = rec
        it["ar_cos_input"] = cos(rec, it["vec"], dim=0).item()
        if it["kind"] != "mean":
            it["ar_cos_perp_src"] = cos_perp(rec, it["src_vec"])
        if it["ex"] is not None:
            resid = next(x for x in items if x["ex"] == it["ex"] and x["kind"] == "resid")["src_vec"]
            it["ar_cos_resid"] = cos(rec, resid, dim=0).item()
            it["ar_cos_perp_resid"] = cos_perp(rec, resid)
        if it["sampled_t1"] is not None:
            it["ar_cos_input_sampled"] = cos(reconstruct(it["sampled_t1"]), it["vec"], dim=0).item()

    # matching test within each example: does AR(expl_i), minus the mean direction, point at head i more than
    # at the other heads decoded at the same position?
    for ei in range(len(examples_out)):
        for variant in ("raw", "mean+"):
            hs = [x for x in items if x["ex"] == ei and x["kind"] == "head" and x["variant"] == variant]
            P = torch.stack([perp(x["src_vec"]) / perp(x["src_vec"]).norm() for x in hs])
            for i, x in enumerate(hs):
                r = perp(x["recon"])
                sims = P @ (r / r.norm())
                mine = sims[i].item()
                x["match_rank"] = 1 + sum(s > mine + 1e-9 for s in sims.tolist())
                x["match_of"] = len(hs)

    for it in items:
        del it["vec"], it["recon"], it["src_vec"]
    print(f"done {time.time()-t0:.0f}s")
    return {"examples": examples_out, "items": items, "cfg": cfg, "pool_stats": pool_stats, "av_prompt": av_tok.decode(p_ids)}


@app.local_entrypoint()
def main(config: str = "config.json", out: str = "results.json"):
    cfg = json.loads(pathlib.Path(config).read_text())
    res = run.remote(cfg)
    pathlib.Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("wrote", out)
