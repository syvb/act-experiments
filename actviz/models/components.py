"""Reusable pieces of transformer forward passes, each of which both computes
its outputs and records them (with equations) into a Trace page.

A new model adapter composes these and adds whatever is specific to it. For
example, a Llama adapter would need an RMSNorm and a RoPE step, and would pass
grouped-query k/v (repeated to one per query head) into `attention`.
"""

from __future__ import annotations

import numpy as np

from ..trace import Page, Trace, grid, op, row, t, weight


def layernorm(
    tr: Trace,
    page: Page,
    *,
    prefix: str,
    x_id: str,
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    chunk: str,
    x_label: str = "x",
) -> tuple[str, np.ndarray]:
    """LayerNorm over the last axis of x [pos, d_model], every step recorded."""
    d = x.shape[-1]
    mean = x.mean(-1)
    xc = x - mean[:, None]
    var = (xc**2).mean(-1)
    rstd = 1.0 / np.sqrt(var + eps)
    xh = xc * rstd[:, None]
    y = xh * gamma + beta

    ids = {
        "mean": tr.add(f"{prefix}.mean", mean, ["pos"], chunk=chunk, label="μ (mean)"),
        "xc": tr.add(f"{prefix}.centered", xc, ["pos", "d_model"], chunk=chunk, label="x − μ"),
        "var": tr.add(f"{prefix}.var", var, ["pos"], chunk=chunk, kind="positive", label="σ² (variance)"),
        "rstd": tr.add(f"{prefix}.rstd", rstd, ["pos"], chunk=chunk, kind="positive", label="1/√(σ²+ε)"),
        "xh": tr.add(f"{prefix}.normalized", xh, ["pos", "d_model"], chunk=chunk, label="x̂ (normalized)"),
        "g": tr.add(f"{prefix}.w", gamma, ["d_model"], chunk=chunk, label="γ (learned scale)"),
        "b": tr.add(f"{prefix}.b", beta, ["d_model"], chunk=chunk, label="β (learned shift)"),
        "y": tr.add(f"{prefix}.out", y, ["pos", "d_model"], chunk=chunk, label="LayerNorm output"),
    }
    page.equation(
        "Mean of each token's vector",
        row(t(x_id, x_label), op("mean →"), t(ids["mean"], "μ", rows=["pos"], values=True)),
        formula=f"μᵢ = (1/{d}) Σⱼ xᵢⱼ",
    )
    page.equation(
        "Subtract the mean",
        row(t(x_id, x_label), op("−"), t(ids["mean"], "μ", rows=["pos"])),
        op("="),
        t(ids["xc"], "x − μ"),
        formula="every token's vector now averages to 0",
    )
    page.equation(
        "Variance, and the factor that rescales it to 1",
        row(
            t(ids["xc"], "x − μ"),
            op("mean of squares →"),
            t(ids["var"], "σ²", rows=["pos"], values=True),
            op(f"1/√(σ² + {eps:g}) →"),
            t(ids["rstd"], "1/σ", rows=["pos"], values=True),
        ),
        formula=f"σᵢ² = (1/{d}) Σⱼ (xᵢⱼ − μᵢ)²",
    )
    page.equation(
        "Normalize",
        row(t(ids["xc"], "x − μ"), op("×"), t(ids["rstd"], "1/σ", rows=["pos"])),
        op("="),
        t(ids["xh"], "x̂"),
        formula="x̂ = (x − μ) / √(σ² + ε): each row now has mean 0 and variance 1",
    )
    page.equation(
        "Learned per-dimension scale and shift",
        t(ids["xh"], "x̂"),
        op("⊙"),
        t(ids["g"], "γ"),
        op("+"),
        t(ids["b"], "β"),
        op("="),
        t(ids["y"], "y"),
        formula="y = x̂ ⊙ γ + β  (γ and β are the same for every token)",
    )
    return ids["y"], y


def attention(
    tr: Trace,
    page: Page,
    *,
    prefix: str,
    q_id: str,
    k_id: str,
    v_id: str,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    w_o: np.ndarray,
    b_o: np.ndarray | None,
    chunk: str,
    detail_chunk: str,
    pattern_chunk: str,
) -> tuple[str, np.ndarray]:
    """Causal multi-head attention from already-projected q, k, v [pos, head, d_head].

    w_o is [head, d_head, d_model]. Returns the id and value of the attention
    output [pos, d_model] (before it is added to the residual stream).
    """
    n_pos, n_heads, d_head = q.shape
    d_model = w_o.shape[-1]
    qh, kh, vh = (a.transpose(1, 0, 2) for a in (q, k, v))  # [head, pos, d_head]

    qk = qh[:, :, None, :] * kh[:, None, :, :]  # [head, q_pos, k_pos, d_head]
    raw = qh @ kh.transpose(0, 2, 1)
    scale = float(np.sqrt(d_head))
    scores = raw / scale
    causal = np.tril(np.ones((n_pos, n_pos), dtype=bool))
    masked = np.where(causal, scores, -np.inf)
    rowmax = masked.max(-1)
    shifted = masked - rowmax[..., None]
    ex = np.exp(shifted)
    exsum = ex.sum(-1)
    pattern = ex / exsum[..., None]
    weighted = pattern[..., None] * vh[:, None, :, :]  # [head, q_pos, k_pos, d_head]
    z = pattern @ vh  # [head, pos, d_head]
    z_flat = z.transpose(1, 0, 2).reshape(n_pos, n_heads * d_head)
    out = z_flat @ w_o.reshape(n_heads * d_head, d_model)
    if b_o is not None:
        out = out + b_o
    head_out = np.einsum("hpd,hdm->hpm", z, w_o)
    head_norm = np.linalg.norm(head_out, axis=-1)

    P, hqk, hqkd = prefix, ["head", "q_pos", "k_pos"], ["head", "q_pos", "k_pos", "d_head"]
    ids = {
        "qk": tr.add(f"{P}.qk", qk, hqkd, chunk=detail_chunk, label="q ⊙ k (elementwise)"),
        "raw": tr.add(f"{P}.scores_raw", raw, hqk, chunk=detail_chunk, label="q · k"),
        "scores": tr.add(f"{P}.scores", scores, hqk, chunk=detail_chunk, label=f"q · k / √{d_head}"),
        "masked": tr.add(f"{P}.masked", masked, hqk, chunk=detail_chunk, label="masked scores"),
        "rowmax": tr.add(f"{P}.rowmax", rowmax, ["head", "q_pos"], chunk=detail_chunk, label="row max"),
        "shifted": tr.add(f"{P}.shifted", shifted, hqk, chunk=detail_chunk, label="score − row max"),
        "exp": tr.add(f"{P}.exp", ex, hqk, chunk=detail_chunk, kind="prob", label="exp(score − max)"),
        "exsum": tr.add(f"{P}.expsum", exsum, ["head", "q_pos"], chunk=detail_chunk, kind="positive", label="Σ exp"),
        "pattern": tr.add(f"{P}.pattern", pattern, hqk, chunk=pattern_chunk, kind="prob", label="attention pattern"),
        "weighted": tr.add(f"{P}.weighted", weighted, hqkd, chunk=detail_chunk, label="A · v (per key)"),
        "z": tr.add(f"{P}.z", z.transpose(1, 0, 2), ["pos", "head", "d_head"], chunk=chunk, label="z (mixed values)"),
        "head_out": tr.add(f"{P}.head_out", head_out, ["head", "pos", "d_model"], chunk=detail_chunk, label="head output"),
        "head_norm": tr.add(
            f"{P}.head_norm", head_norm, ["head", "pos"], chunk=pattern_chunk, kind="positive", label="‖head output‖"
        ),
        "out": tr.add(f"{P}.out", out, ["pos", "d_model"], chunk=chunk, label="attention output"),
    }
    if b_o is not None:
        ids["b_o"] = tr.add(f"{P}.b_O", b_o, ["d_model"], chunk=chunk, label="b_O")

    hfix = {"head": "$head"}
    page.block(
        "multiples",
        title="Attention patterns of every head (click one to follow it below)",
        rows=[{"tensor": ids["pattern"]}],
        split="head",
    )
    page.equation(
        "Head {head}: its slice of the queries and keys",
        row(
            t(q_id, "q for head {head}", fix=hfix, cell=7),
            t(k_id, "k for head {head}", fix=hfix, select={"pos": "kpos"}, cell=7),
        ),
        formula=f"each head reads its own {d_head} of the {n_heads * d_head} dimensions",
    )
    page.equation(
        "Score every query against every key",
        row(
            t(ids["raw"], "q · kᵀ", fix=hfix),
            op(f"÷ √{d_head} = ÷ {scale:g}"),
            t(ids["scores"], "scores", fix=hfix),
            op("causal mask →"),
            t(ids["masked"], "masked", fix=hfix),
            op("softmax each row →"),
            t(ids["pattern"], "pattern A", fix=hfix),
        ),
        formula=f"Aᵢⱼ = softmaxⱼ(qᵢ · kⱼ / √{d_head}), with Sᵢⱼ = −∞ for j > i so no token sees the future",
        note="Rows are queries (the token doing the looking), columns are keys (the tokens being looked at). "
        "Click a cell to pick a query and key for the close-ups below.",
    )
    fix1 = {"head": "$head", "q_pos": "$pos", "k_pos": "$kpos"}
    page.equation(
        "Close-up: one score, query {pos} → key {kpos}",
        t(q_id, "q at {pos}", fix={"pos": "$pos", "head": "$head"}, cell=10),
        op("⊙"),
        t(k_id, "k at {kpos}", fix={"pos": "$kpos", "head": "$head"}, select={"pos": "kpos"}, cell=10),
        op("="),
        t(ids["qk"], "q ⊙ k", fix=fix1, cell=10),
        row(
            op("sum ="),
            t(ids["raw"], "q · k", fix=fix1, values=True),
            op(f"÷ {scale:g} ="),
            t(ids["scores"], "score", fix=fix1, values=True),
        ),
        formula="a score is a dot product: multiply matching dimensions, then add them up",
    )
    rowfix = {"head": "$head", "q_pos": "$pos"}
    page.equation(
        "Close-up: softmax across the keys for query {pos}",
        row(
            t(ids["masked"], "score", fix=rowfix, rows=["k_pos"], values=True),
            op("−"),
            t(ids["rowmax"], "max", fix=rowfix, values=True),
            op("="),
            t(ids["shifted"], "shifted", fix=rowfix, rows=["k_pos"], values=True),
            op("exp →"),
            t(ids["exp"], "exp", fix=rowfix, rows=["k_pos"], values=True),
            op("÷"),
            t(ids["exsum"], "Σ exp", fix=rowfix, values=True),
            op("="),
            t(ids["pattern"], "A", fix=rowfix, rows=["k_pos"], values=True),
        ),
        formula="Aᵢⱼ = exp(Sᵢⱼ − maxₖ Sᵢₖ) / Σₖ exp(Sᵢₖ − maxₖ Sᵢₖ); subtracting the max changes nothing but avoids overflow",
    )
    page.equation(
        "Close-up: mix the values for query {pos}",
        grid(
            [
                t(ids["pattern"], "A", fix=rowfix, rows=["k_pos"], cell=8),
                op("×"),
                t(v_id, "v (head {head})", fix=hfix, select={"pos": "kpos"}, scale="v", cell=8, labels=False),
            ],
            [
                None,
                op("="),
                t(ids["weighted"], "Aᵢⱼ · vⱼ", fix=rowfix, scale="v", cell=8, labels=False),
            ],
        ),
        op("sum over keys ↓"),
        t(ids["z"], "z at {pos}", fix={"pos": "$pos", "head": "$head"}, cell=10),
        formula="zᵢ = Σⱼ Aᵢⱼ vⱼ: a weighted average of the value vectors, weighted by the pattern row",
        note="Each row of v is scaled by how much query {pos} attends to that key; "
        "rows with near-zero weight fade out. v and Aᵢⱼ·vⱼ share a color scale.",
    )
    page.equation(
        "Every query at once",
        row(
            t(ids["pattern"], "A", fix=hfix, cell=8),
            op("·"),
            t(v_id, "v (head {head})", fix=hfix, cell=8, labels=False),
        ),
        op("="),
        t(ids["z"], "z (head {head})", fix=hfix, cell=8),
        formula="z = A · v: row i of z is the close-up above, done for query i",
    )
    page.equation(
        "Concatenate the heads",
        t(ids["z"], "z (all heads)"),
        formula=f"{n_heads} heads × {d_head} dims = {n_heads * d_head}",
    )
    page.equation(
        "Head {head} writes back to the residual stream",
        row(t(ids["z"], "z (head {head})", fix=hfix, cell=6), op("·"), weight("W_O[{head}]", [d_head, d_model])),
        op("="),
        t(ids["head_out"], "head {head} output", fix=hfix),
        formula=f"each head has its own {d_head}×{d_model} slice of the output matrix",
    )
    page.equation(
        "How strongly each head writes at each position",
        t(ids["head_norm"], "‖head output‖", rows=["head"], cols=["pos"], values=False, cell=14),
        note="Click a square to select that head and token.",
    )
    lines = [op("Σ over heads of head output")]
    if b_o is not None:
        lines += [op("+"), t(ids["b_o"], "b_O")]
    page.equation(
        "Add up the heads",
        *lines,
        op("="),
        t(ids["out"], "attention output"),
        formula="attn_out = Σₕ zₕ · W_O[h] + b_O  (the same as concat(z) · W_O + b_O)",
    )
    return ids["out"], out


def residual_add(page: Page, *, title: str, resid_id: str, delta_id: str, out_id: str, delta_label: str, formula: str):
    page.equation(
        title,
        t(resid_id, "residual stream in", scale="resid"),
        op("+"),
        t(delta_id, delta_label, scale="resid"),
        op("="),
        t(out_id, "residual stream out", scale="resid"),
        formula=formula,
        note="Shown on a shared color scale, so you can compare how large the update is next to the stream.",
    )


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def topk_table(tokenizer_text, logits: np.ndarray, probs: np.ndarray, ids: list[int], k: int = 10) -> list[dict]:
    """Per position: the k most likely next tokens, and how the real next token fared."""
    out = []
    for p in range(logits.shape[0]):
        order = np.argsort(-logits[p])
        entry = {
            "top": [
                {"id": int(i), "text": tokenizer_text(int(i)), "logit": float(logits[p, i]), "prob": float(probs[p, i])}
                for i in order[:k]
            ]
        }
        if p + 1 < len(ids):
            nxt = ids[p + 1]
            entry["next"] = {
                "id": int(nxt),
                "text": tokenizer_text(int(nxt)),
                "logit": float(logits[p, nxt]),
                "prob": float(probs[p, nxt]),
                "rank": int(np.where(order == nxt)[0][0]) + 1,
            }
        out.append(entry)
    return out
