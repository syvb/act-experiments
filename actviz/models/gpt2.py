"""GPT-2 (any size), implemented in numpy so that every intermediate can be recorded.

Matches Hugging Face's GPT2LMHeadModel (see tests/test_gpt2_matches_hf.py).
Weights use HF's Conv1D layout, i.e. y = x @ W + b with W of shape [in, out].
"""

from __future__ import annotations

import json

import numpy as np

from ..hub import Checkpoint, download
from ..trace import Trace, op, row, t, weight
from . import components as C
from .base import ModelAdapter, token_infos, token_text, vocab_labels

DISPLAY_NAMES = {
    "openai-community/gpt2": "GPT-2 small",
    "openai-community/gpt2-medium": "GPT-2 medium",
    "openai-community/gpt2-large": "GPT-2 large",
    "openai-community/gpt2-xl": "GPT-2 XL",
}


class GPT2(ModelAdapter):
    def __init__(self, repo: str, config: dict, ckpt: Checkpoint, tokenizer):
        self.repo = repo
        self.cfg = config
        self.w = ckpt
        self.tok = tokenizer

    @classmethod
    def load(cls, repo: str = "openai-community/gpt2", revision: str | None = None) -> "GPT2":
        from tokenizers import Tokenizer

        config = json.loads(download(repo, "config.json", revision).read_text())
        tok = Tokenizer.from_file(str(download(repo, "tokenizer.json", revision)))
        return cls(repo, config, Checkpoint.from_hub(repo, revision), tok)

    def tokenize(self, prompt: str, bos: bool = True) -> list[int]:
        ids = self.tok.encode(prompt, add_special_tokens=False).ids
        return ([self.cfg["bos_token_id"]] if bos else []) + ids

    def _p(self, name: str) -> np.ndarray:
        # Checkpoints saved from GPT2LMHeadModel prefix everything with "transformer."
        return self.w[name] if name in self.w else self.w["transformer." + name]

    def trace(self, ids: list[int], prompt: str) -> Trace:
        cfg = self.cfg
        n_layers, n_heads, d_model = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
        d_head, d_mlp, eps = d_model // n_heads, cfg.get("n_inner") or 4 * d_model, cfg["layer_norm_epsilon"]
        n_vocab, n_pos = cfg["vocab_size"], len(ids)
        if n_pos > cfg["n_positions"]:
            raise ValueError(f"prompt is {n_pos} tokens; the model's context is {cfg['n_positions']}")
        name = DISPLAY_NAMES.get(self.repo, self.repo)

        stages = ["embed"] + [f"L{L} {part}" for L in range(n_layers) for part in ("+attn", "+mlp")]
        tr = Trace(
            model={
                "name": "gpt2",
                "repo": self.repo,
                "display_name": name,
                "config": {
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "d_model": d_model,
                    "d_head": d_head,
                    "d_mlp": d_mlp,
                    "n_vocab": n_vocab,
                },
            },
            prompt=prompt,
            tokens=token_infos(self.tok, ids),
            axes={
                "pos": {"labels": "tokens"},
                "q_pos": {"labels": "tokens", "title": "query"},
                "k_pos": {"labels": "tokens", "title": "key"},
                "head": {"prefix": "H"},
                "vocab": {"labels": "vocab"},
                "stage": {"labels": stages},
                "sample": {},
            },
        )
        tr.add_labels("vocab", vocab_labels(self.tok, n_vocab))

        resid_stages: list[np.ndarray] = []

        # ---- overview (filled in after the forward pass) and embedding --------------
        model_sec = tr.section("model", name)
        overview = model_sec.page("overview", "Overview", "Overview")
        emb = model_sec.page("embed", "Tokens and embeddings", "Embed")

        W_E, W_P = self._p("wte.weight"), self._p("wpe.weight")
        tok_emb, pos_emb = W_E[ids], W_P[:n_pos]
        resid = tok_emb + pos_emb
        resid_stages.append(resid)
        tr.add("embed.tok", tok_emb, ["pos", "d_model"], chunk="embed", label="token embedding")
        tr.add("embed.pos", pos_emb, ["pos", "d_model"], chunk="embed", label="position embedding")

        emb.text(
            f"""The prompt is split into **{n_pos} tokens** (GPT-2's byte-level BPE vocabulary has {n_vocab:,} entries).
The first token is `<|endoftext|>`, which GPT-2 saw between documents during training, so it acts as a
"beginning of text" marker. Each token id selects a row of the embedding matrix W_E, and each position
selects a row of the position-embedding matrix W_P. Their sum is the **residual stream**: a {d_model}-dimensional
vector per token that every later layer reads from and adds to."""
        )
        emb.block("tokens")
        emb.equation(
            "Look up each token's embedding",
            row(op("W_E [token id]"), weight("W_E", [n_vocab, d_model])),
            op("="),
            t("embed.tok", "token embedding", scale="resid"),
            formula="row i is W_E[tokenᵢ]: the same token always gets the same vector, wherever it appears",
        )
        emb.equation(
            "Look up each position's embedding",
            row(op("W_P [position]"), weight("W_P", [cfg["n_positions"], d_model])),
            op("="),
            t("embed.pos", "position embedding", scale="resid"),
            formula="row i is W_P[i]: this is the only way the model knows token order",
        )
        emb.equation(
            "Sum them to start the residual stream",
            t("embed.tok", "token embedding", scale="resid"),
            op("+"),
            t("embed.pos", "position embedding", scale="resid"),
            op("="),
            t("L0.resid_pre", "residual stream", scale="resid"),
        )

        # ---- transformer blocks ---------------------------------------------------
        resid_id = "L0.resid_pre"
        for L in range(n_layers):
            sec = tr.section(f"L{L}", f"Layer {L}", short=f"L{L}")
            pre = f"h.{L}."
            ch, ch_attn = f"L{L}", f"L{L}a"

            # LayerNorm 1
            p_ln1 = sec.page(f"L{L}.ln1", f"Layer {L} · LayerNorm before attention", "LN₁")
            p_ln1.text(
                f"""Before attention reads the residual stream, LayerNorm rescales each token's vector to have mean 0
and variance 1, then applies a learned per-dimension scale γ and shift β. Attention never sees the raw stream:
only this normalized copy."""
            )
            ln1_id, ln1 = C.layernorm(
                tr, p_ln1, prefix=f"L{L}.ln1", x_id=resid_id, x=resid,
                gamma=self._p(pre + "ln_1.weight"), beta=self._p(pre + "ln_1.bias"), eps=eps, chunk=ch,
                x_label=f"x: residual stream entering layer {L}",
            )

            # Attention
            p_attn = sec.page(f"L{L}.attn", f"Layer {L} · Attention", "Attn")
            p_attn.text(
                f"""Attention is how information moves *between* tokens. Each of the {n_heads} heads projects every
token into a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a **value** ("what do I pass on
if attended to?"). A query's dot product with every earlier key gives a score; softmax turns each row of scores into
weights; and each token's output is the weighted average of the values. The selected head, query and key (set them by
clicking squares, the tokens at the top, or the head buttons) drive the close-ups on this page."""
            )
            w_qkv, b_qkv = self._p(pre + "attn.c_attn.weight"), self._p(pre + "attn.c_attn.bias")
            qkv = ln1 @ w_qkv + b_qkv
            hd = ["pos", "head", "d_head"]
            q, k, v = (a.reshape(n_pos, n_heads, d_head) for a in np.split(qkv, 3, axis=-1))
            b_q, b_k, b_v = (b.reshape(n_heads, d_head) for b in np.split(b_qkv, 3))
            q_id = tr.add(f"L{L}.attn.q", q, hd, chunk=ch, label="q (queries)")
            k_id = tr.add(f"L{L}.attn.k", k, hd, chunk=ch, label="k (keys)")
            v_id = tr.add(f"L{L}.attn.v", v, hd, chunk=ch, label="v (values)")
            lines = [t(ln1_id, "x: LN₁ output")]
            for nm, tid, b in (("Q", q_id, b_q), ("K", k_id, b_k), ("V", v_id, b_v)):
                bid = tr.add(f"L{L}.attn.b_{nm}", b, ["head", "d_head"], chunk=ch, label=f"b_{nm}")
                lines += [
                    row(op("x ·"), weight(f"W_{nm}", [d_model, d_model]), op("+")),
                    t(bid, f"b_{nm}", rows=[], cols=["head", "d_head"]),
                    op("="),
                    t(tid, nm.lower()),
                ]
            p_attn.equation(
                "Project every token into queries, keys and values",
                *lines,
                formula="q = x·W_Q + b_Q,  k = x·W_K + b_K,  v = x·W_V + b_V",
                note=f"Columns are grouped by head: each block of {d_head} is one head's share. "
                "The outlined block is the selected head.",
            )
            w_o = self._p(pre + "attn.c_proj.weight").reshape(n_heads, d_head, d_model)
            attn_id, attn = C.attention(
                tr, p_attn, prefix=f"L{L}.attn", q_id=q_id, k_id=k_id, v_id=v_id, q=q, k=k, v=v,
                w_o=w_o, b_o=self._p(pre + "attn.c_proj.bias"),
                chunk=ch, detail_chunk=ch_attn, pattern_chunk="patterns",
            )
            resid_mid = resid + attn
            resid_stages.append(resid_mid)
            C.residual_add(
                p_attn, title="Add to the residual stream", resid_id=resid_id, delta_id=attn_id,
                out_id=f"L{L}.resid_mid", delta_label="attention output",
                formula="resid_mid = resid_pre + attn_out: attention only ever adds to the stream",
            )

            # LayerNorm 2
            p_ln2 = sec.page(f"L{L}.ln2", f"Layer {L} · LayerNorm before the MLP", "LN₂")
            p_ln2.text("The MLP also reads a freshly normalized copy of the residual stream.")
            ln2_id, ln2 = C.layernorm(
                tr, p_ln2, prefix=f"L{L}.ln2", x_id=f"L{L}.resid_mid", x=resid_mid,
                gamma=self._p(pre + "ln_2.weight"), beta=self._p(pre + "ln_2.bias"), eps=eps, chunk=ch,
                x_label=f"x: residual stream after layer {L}'s attention",
            )

            # MLP
            p_mlp = sec.page(f"L{L}.mlp", f"Layer {L} · MLP", "MLP")
            p_mlp.text(
                f"""The MLP processes each token **independently** (no information moves between tokens here).
It expands each {d_model}-dim vector to {d_mlp} "neurons", applies the GELU nonlinearity, and projects back down
to {d_model} to add to the residual stream."""
            )
            w_in, b_in = self._p(pre + "mlp.c_fc.weight"), self._p(pre + "mlp.c_fc.bias")
            w_out, b_out = self._p(pre + "mlp.c_proj.weight"), self._p(pre + "mlp.c_proj.bias")
            h_pre = ln2 @ w_in + b_in
            h_post = C.gelu_tanh(h_pre)
            mlp_out = h_post @ w_out + b_out
            tr.add(f"L{L}.mlp.b_in", b_in, ["d_mlp"], chunk=ch, label="b_in")
            tr.add(f"L{L}.mlp.pre", h_pre, ["pos", "d_mlp"], chunk=ch, label="pre-activation")
            tr.add(f"L{L}.mlp.post", h_post, ["pos", "d_mlp"], chunk=ch, label="neuron activation")
            tr.add(f"L{L}.mlp.b_out", b_out, ["d_model"], chunk=ch, label="b_out")
            tr.add(f"L{L}.mlp.out", mlp_out, ["pos", "d_model"], chunk=ch, label="MLP output")
            p_mlp.equation(
                f"Expand to {d_mlp} neurons",
                t(ln2_id, "x: LN₂ output"),
                row(op("x ·"), weight("W_in", [d_model, d_mlp]), op("+")),
                t(f"L{L}.mlp.b_in", "b_in"),
                op("="),
                t(f"L{L}.mlp.pre", "pre-activation h"),
                formula="h = x·W_in + b_in",
            )
            p_mlp.equation(
                "Apply GELU to every neuron",
                op("GELU(h) ="),
                t(f"L{L}.mlp.post", "activation g"),
                formula="GELU(h) ≈ ½h(1 + tanh(√(2/π)(h + 0.044715h³))): roughly h for h > 0, roughly 0 for h < 0",
                note="Most pre-activations are negative, so most neurons end up near zero; "
                "the few positive ones carry the MLP's output.",
            )
            p_mlp.block(
                "fnplot",
                title="Where token {pos}'s neurons land on the GELU curve",
                curve={"x": "gelu.x", "y": "gelu.y"},
                points={"x": f"L{L}.mlp.pre", "y": f"L{L}.mlp.post", "axis": "pos"},
                xlabel="pre-activation h",
                ylabel="GELU(h)",
            )
            p_mlp.equation(
                f"Project back to {d_model} dimensions",
                t(f"L{L}.mlp.post", "activation g"),
                row(op("g ·"), weight("W_out", [d_mlp, d_model]), op("+")),
                t(f"L{L}.mlp.b_out", "b_out"),
                op("="),
                t(f"L{L}.mlp.out", "MLP output"),
                formula="mlp_out = g·W_out + b_out",
            )
            resid_post = resid_mid + mlp_out
            resid_stages.append(resid_post)
            C.residual_add(
                p_mlp, title="Add to the residual stream", resid_id=f"L{L}.resid_mid", delta_id=f"L{L}.mlp.out",
                out_id=f"L{L}.resid_post", delta_label="MLP output",
                formula="resid_post = resid_mid + mlp_out",
            )
            resid = resid_post
            resid_id = f"L{L}.resid_post"

        # ---- unembedding ----------------------------------------------------------
        out_sec = tr.section("output", "Output")
        p_lnf = out_sec.page("ln_f", "Final LayerNorm", "LN_f")
        p_lnf.text("One last LayerNorm before the residual stream is read out as next-token predictions.")
        lnf_id, lnf = C.layernorm(
            tr, p_lnf, prefix="ln_f", x_id=resid_id, x=resid,
            gamma=self._p("ln_f.weight"), beta=self._p("ln_f.bias"), eps=eps, chunk="unembed",
            x_label="x: residual stream after the last layer",
        )
        logits = lnf @ W_E.T
        probs = C.softmax(logits)
        tr.add("logits", logits, ["pos", "vocab"], chunk="unembed", label="logits")
        tr.add("probs", probs, ["pos", "vocab"], chunk="unembed", kind="positive", label="probability",
               scale=float(np.percentile(probs, 99.99)), percent=True)
        p_out = out_sec.page("unembed", "Unembedding and predictions", "Logits")
        p_out.text(
            f"""Multiplying by the transpose of the token-embedding matrix (GPT-2 ties the two) gives one **logit**
per vocabulary entry: {n_vocab:,} squares per token, in token-id order. Softmax turns them into a probability
distribution over the next token. Every position predicts the token that follows it; the last position's
prediction is the model's continuation of the prompt."""
        )
        p_out.equation(
            "Score every vocabulary entry",
            t(lnf_id, "x: final LayerNorm output", fix={"pos": "$pos"}),
            row(op("x ·"), weight("W_Eᵀ", [d_model, n_vocab], note="tied to the token embedding")),
            op("="),
            t("logits", "logits for token {pos}", fix={"pos": "$pos"}, cell=3, norm="range"),
            formula="logits = x · W_Eᵀ (no bias)",
            note="Logits are all negative here, and only their differences matter to softmax, so they are colored "
            "relative to this token's own range. Hover a square to see which token it scores.",
        )
        p_out.equation(
            "Softmax into probabilities",
            t("probs", "P(next token) after {pos}", fix={"pos": "$pos"}, cell=3),
            formula="P(v) = exp(logit_v) / Σ_u exp(logit_u)",
            note="Nearly all of the probability sits on a handful of squares; hover to find them, or see the table.",
        )
        topk = C.topk_table(lambda i: token_text(self.tok, i), logits, probs, ids)
        tr.extra["topk"] = topk
        p_out.block("topk", title="Most likely next tokens after {pos}")
        p_out.block("predictions", title="What the model predicts at every position")

        # ---- residual stream stack, GELU curve and overview ------------------------
        stack = np.stack(resid_stages)
        tr.add("resid", stack, ["stage", "pos", "d_model"], chunk="resid", label="residual stream")
        tr.alias("L0.resid_pre", "resid", {"stage": 0}, label="residual stream (embeddings)")
        for L in range(n_layers):
            tr.alias(f"L{L}.resid_mid", "resid", {"stage": 2 * L + 1}, label=f"residual stream after L{L} attention")
            tr.alias(f"L{L}.resid_post", "resid", {"stage": 2 * L + 2}, label=f"residual stream after layer {L}")
        norms = np.linalg.norm(stack, axis=-1)  # [stage, pos]
        tr.add("resid.norm", norms, ["stage", "pos"], chunk="overview", kind="positive", label="‖residual stream‖",
               scale=float(norms[:, 1:].max()) if n_pos > 1 else None)
        gx = np.linspace(-6, 6, 241, dtype=np.float32)
        tr.add("gelu.x", gx, ["sample"], chunk="overview", label="h")
        tr.add("gelu.y", C.gelu_tanh(gx), ["sample"], chunk="overview", label="GELU(h)")

        last = topk[-1]["top"][0]
        overview.text(
            f"""This page lays out **every intermediate value** {name} computes on the prompt below, as grids of
colored squares: red is positive, blue is negative, gray is near zero; probabilities run from pale to deep orange.
Hover any square for its exact value. Everything was computed ahead of time; the page itself runs no model.

The model has {n_layers} layers, each with {n_heads} attention heads and an MLP of {d_mlp} neurons, all reading
from and writing to a {d_model}-dimensional residual stream. Its top prediction for the next token is
**{last['text']!r}** ({last['prob']:.1%}). Use the outline to walk through the computation in order, or
**←/→** to step between pages."""
        )
        overview.block("topk", title="Most likely next tokens after {pos}")
        overview.block(
            "multiples",
            title="Every attention pattern in the model (click one to open it)",
            rows=[{"tensor": f"L{L}.attn.pattern", "label": f"L{L}", "page": f"L{L}.attn"} for L in range(n_layers)],
            split="head",
            cell=3,
        )
        overview.equation(
            "Size of the residual stream as it flows through the model",
            t("resid.norm", "‖residual stream‖", rows=["pos"], cols=["stage"], cell=12),
            note="Columns are the points between steps (after the embeddings, after each attention, after each MLP). "
            "The first token's stream grows far larger than the others'; its color is clipped so the rest stay visible.",
        )
        overview.equation(
            "Token {pos}'s residual stream at every stage",
            t("resid", "residual stream of {pos}", fix={"pos": "$pos"}, rows=["stage"], cols=["d_model"], norm="row"),
            note="Each row is colored on its own scale (the stream grows with depth). Pick another token at the top.",
        )
        return tr
