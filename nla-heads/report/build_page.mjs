// node build_page.mjs results_v2.json results_v1.json > ../page/index.html
import fs from "node:fs";

const res = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const v1 = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const sampledRaw = new Map(v1.items.map((x) => [`${x.ex}|${x.label}`, x.sampled_t1]));

const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/\uFFFD/g, "&#xFFFD;");
const f = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(d));
const tokChip = (t) => `<code class="tok">${esc(t.replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/\r/g, "␍"))}</code>`;
const expl = (s) =>
  String(s ?? "")
    .replace(/<\/?explanation>/g, "")
    .trim()
    .split(/\n\s*\n/)
    .map((para) => `<p>${esc(para.trim())}</p>`)
    .join("");

const items = res.items;
const heads = (variant) => items.filter((x) => x.kind === "head" && x.variant === variant);
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const stats = {
  rawPerp: mean(heads("raw").map((x) => x.ar_cos_perp_src)),
  mpPerp: mean(heads("mean+").map((x) => x.ar_cos_perp_src)),
  rawTop1: heads("raw").filter((x) => x.match_rank === 1).length,
  mpTop1: heads("mean+").filter((x) => x.match_rank === 1).length,
  n: heads("raw").length,
  residPerp: items.filter((x) => x.kind === "resid" && x.variant === "raw").map((x) => x.ar_cos_perp_src),
  randPerp: items.filter((x) => x.kind === "random").map((x) => x.ar_cos_perp_src),
  headNorms: heads("raw").map((x) => x.raw_norm).sort((a, b) => a - b),
};
const median = (xs) => xs[Math.floor(xs.length / 2)];

// my qualitative read of which decodes track the head's attended content (see summary)
const hits = {
  "0|L13H16": "starter → yeast / bacteria culture",
  "0|L2H2": "copies the word “all”",
  "0|L22H12": "copies the word “all”",
  "3|L10H20": "letters, a girl, her mother",
  "3|L22H15": "lighthouse, island, night",
  "3|L8H12": "counting letters (mean+ only)",
};

const P = [];
const p = (s) => P.push(s);

p(`<title>NLA Head Decodes</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: a single reading column for prose, widening to two decode columns (raw | mean+) per head card. */
:root {
  --bg: #fbfbf7;
  --panel: #f2f1ea;
  --fg: #1c1c19;
  --muted: #66655d;
  --rule: #dfddd2;
  --ink: #7a5d00;
  --mark: #ffe14d;
  --mark-fg: #1c1c19;
  --good: #2f6b3a;
  --good-bg: #e3f0e2;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #131310; --panel: #1d1d19; --fg: #ebe9e0; --muted: #a29f93; --rule: #33322b;
    --ink: #f4cd38; --mark: #f4cd38; --mark-fg: #131310; --good: #9fd6a5; --good-bg: #1f3322;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #131310; --panel: #1d1d19; --fg: #ebe9e0; --muted: #a29f93; --rule: #33322b;
  --ink: #f4cd38; --mark: #f4cd38; --mark-fg: #131310; --good: #9fd6a5; --good-bg: #1f3322;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg);
  font-family: var(--body); font-size: 17px; line-height: 1.55;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 40px 80px;
}
main { max-width: 1120px; margin: 0 auto; display: flex; flex-direction: column; gap: 56px; }
.prose { max-width: 68ch; display: flex; flex-direction: column; gap: 14px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; letter-spacing: -0.01em; }
h1 { font-size: clamp(2rem, 5vw, 3.1rem); font-weight: 800; line-height: 1.05; }
h2 { font-size: 1.6rem; font-weight: 800; }
h3 { font-size: 1.12rem; font-weight: 700; }
p { margin: 0; }
a { color: var(--ink); }
code, .mono { font-family: var(--mono); font-size: 0.82em; }
.eyebrow { font-family: var(--display); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.lede { font-size: 1.15rem; }
mark { background: var(--mark); color: var(--mark-fg); padding: 0 0.15em; border-radius: 2px; }
ul.findings { margin: 0; padding-left: 1.2em; display: flex; flex-direction: column; gap: 10px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.stat { border-top: 3px solid var(--fg); padding-top: 8px; display: flex; flex-direction: column; gap: 4px; }
.stat b { font-family: var(--display); font-size: 1.7rem; font-variant-numeric: tabular-nums; }
.stat span { color: var(--muted); font-size: 0.9rem; line-height: 1.35; }
.method { font-size: 0.95rem; color: var(--fg); }
.method dt { font-family: var(--display); font-weight: 700; margin-top: 10px; }
.method dd { margin: 2px 0 0 0; }
.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; vertical-align: top; }
th { font-family: var(--display); font-weight: 700; font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); background: var(--panel); }
td.num { text-align: right; font-family: var(--mono); }
tr:last-child td { border-bottom: 0; }
.chip { display: inline-block; font-family: var(--display); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; padding: 1px 7px; border-radius: 999px; background: var(--good-bg); color: var(--good); white-space: nowrap; }
.toc { display: flex; flex-wrap: wrap; gap: 8px 18px; font-family: var(--display); font-size: 0.95rem; }
section.example { display: flex; flex-direction: column; gap: 22px; border-top: 1px solid var(--rule); padding-top: 28px; }
.context { background: var(--panel); border-radius: 4px; padding: 14px 16px; font-family: var(--mono); font-size: 0.82rem; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }
.context .after { color: var(--muted); opacity: 0.55; }
.meta { color: var(--muted); font-size: 0.9rem; }
.card { display: flex; flex-direction: column; gap: 12px; padding-top: 18px; border-top: 1px dashed var(--rule); }
.card-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; }
.facts { display: flex; flex-direction: column; gap: 4px; font-size: 0.88rem; color: var(--muted); }
.facts .row { display: flex; flex-wrap: wrap; gap: 4px 6px; align-items: baseline; }
code.tok { background: var(--panel); border: 1px solid var(--rule); border-radius: 3px; padding: 0 4px; color: var(--fg); white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-all; }
.decodes { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.decode { min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.decode .label { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px 10px; font-family: var(--display); font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
.decode .label .score { font-family: var(--mono); font-weight: 400; text-transform: none; letter-spacing: 0; color: var(--muted); }
.text { border-left: 3px solid var(--rule); padding-left: 12px; display: flex; flex-direction: column; gap: 8px; font-size: 0.95rem; overflow-wrap: anywhere; }
.baseline .text { border-left-color: var(--mark); }
details { font-size: 0.9rem; }
details summary { cursor: pointer; color: var(--ink); font-family: var(--display); font-weight: 700; font-size: 0.8rem; letter-spacing: 0.03em; }
details[open] summary { margin-bottom: 8px; }
summary:focus-visible, a:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
.lens { font-size: 0.82rem; }
footer { color: var(--muted); font-size: 0.88rem; }
@media (max-width: 520px) { body { font-size: 16px; padding-block: 24px 56px; } }
</style>`);

p(`<main>`);

// ---------- header ----------
p(`<header class="prose">
<div class="eyebrow">Qwen2.5-7B-Instruct · NLA at layer 20 · ${stats.n} attention heads across 4 texts</div>
<h1>What the layer-20 NLA says about single attention heads</h1>
<p class="lede">I took the output of individual attention heads (each head's slice of the attention output, projected through its block of W<sub>O</sub> into the 3584-dim residual space) and fed each one to <a href="https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av">kitft/nla-qwen2.5-7b-L20-av</a> in place of a residual-stream activation. Every decode is shown in full below, next to the head's attention pattern and an AR round-trip score.</p>
</header>`);

// ---------- findings ----------
p(`<section class="prose" aria-labelledby="findings">
<h2 id="findings">What came out</h2>
<ul class="findings">
<li><b>Most head decodes are confabulation.</b> They are fluent and correctly formatted, but read like the decodes of random Gaussian vectors ("Structured … format with …", invented quotes, nonsense final tokens). The activation reconstructor recovers almost none of the head direction from them: the mean-removed cosine averages <b>${f(stats.rawPerp, 3)}</b>. For the real layer-20 residual at the same tokens it is <b>${stats.residPerp.map((x) => f(x)).join(", ")}</b>.</li>
<li><b>A minority of heads decode to what they attend to</b>, and they do so under both injection styles. L13H16 reads <code class="tok">ough</code>+<code class="tok"> starter</code>. Its raw decode is about homemade yogurt and bacteria cultures, and its mean+ decode about a homemade yeast culture and “my own sourdough starter”. L2H2 and L22H12 attend to <code class="tok"> all</code>/<code class="tok">-purpose</code> and both decodes keep returning the word “all” (L22H12 once in Chinese, 所有/全部). L10H20 decodes to a girl's letters to her mother, and L22H15 (whose direct logit lens is <code class="tok"> coastal</code> <code class="tok"> nights</code>) to an island, a lighthouse and night tides.</li>
<li><b>Heads parked on the first token (the attention sink) give pure noise.</b> That is expected. Their output is close to a fixed vector that carries no information about the context.</li>
<li><b>Head outputs are out of distribution.</b> The median head-output norm is ${f(median(stats.headNorms))}, against roughly 100–135 for the residual, so the AV has to scale them up 15–300×. They also lack the direction that every residual shares: unit residuals have cosine ${f(res.pool_stats.mean_cos_resid_mu)} with their mean, while heads sit near 0. The AV never saw vectors like this during training.</li>
<li><b>Adding the mean back helps a little.</b> I injected <i>mean residual + the head's direction, scaled to a typical residual's deviation</i> (“mean+”). Decodes become more fluent, and the AR's reconstruction points at the correct head more often: rank 1 of 10 for <b>${stats.mpTop1}/${stats.n}</b> heads, against ${stats.rawTop1}/${stats.n} raw and ${stats.n / 10}/${stats.n} by chance. The cosines stay small (mean ${f(stats.mpPerp, 3)}), though, and the mean direction's own content leaks in. My mean came from only ${res.pool_stats.n_pool} tokens, some of them about baking, and it decodes as a baking blog. That is why so many mean+ decodes mention dough, months and buckets.</li>
</ul>
</section>`);

// ---------- stats ----------
p(`<section class="stats" aria-label="Key numbers">
<div class="stat"><b>${stats.residPerp.map((x) => f(x)).join(" / ")}</b><span>AR round-trip, mean-removed cosine, for the real layer-20 residual at each of the 4 tokens</span></div>
<div class="stat"><b>${f(stats.rawPerp, 3)}</b><span>Same score averaged over ${stats.n} raw head outputs</span></div>
<div class="stat"><b>${f(stats.mpPerp, 3)}</b><span>Same, with mean+ injection</span></div>
<div class="stat"><b>${stats.randPerp.map((x) => f(x, 3)).join(" … ")}</b><span>Same score for random Gaussian control vectors (raw and mean+)</span></div>
</section>`);

// ---------- method ----------
p(`<section class="prose" aria-labelledby="method">
<h2 id="method">Setup</h2>
<dl class="method">
<dt>Target model and read point</dt><dd>Qwen/Qwen2.5-7B-Instruct in bf16. The NLA reads the output of block 20 (HF <code>hidden_states[21]</code>); I checked my hook against it (exact match).</dd>
<dt>Head output</dt><dd>For layer <i>l</i>, head <i>h</i> at token <i>t</i>: <code>W_O[:, 128h:128(h+1)] @ z[t, 128h:128(h+1)]</code>, where <code>z</code> is the input to <code>o_proj</code>. Qwen2's <code>o_proj</code> has no bias. I checked that the 28 heads sum to the attention block's output. Heads in layers 0–20 write directly into the vector the NLA was trained on. Two heads per text come from layers 21–27, for comparison.</dd>
<dt>Texts</dt><dd>Four short texts I wrote for this run, so the NLA can't have trained on them: a chat-formatted question, a local-news paragraph, a Rust blog snippet and a piece of fiction. A seeded RNG picked one token per text, then 8 heads from layers 0–20 and 2 from layers 21–27.</dd>
<dt>Verbalizing</dt><dd>Each vector was rescaled to L2 norm 150 (the sidecar's <code>injection_scale</code>) and written over the <code>㈎</code> token embedding in the sidecar's AV prompt. Decoding was greedy, up to 400 new tokens, with HF <code>generate(inputs_embeds=…)</code>. Run 1 also drew one sample at T=1 for each raw vector. Greedy decodes were byte-identical across the two runs.</dd>
<dt>Mean+ variant</dt><dd>Take the head's unit direction, remove its component along the mean unit residual μ (estimated from ${res.pool_stats.n_pool} positions of 9 texts), rescale it to the median off-mean size of a real unit residual (${f(res.pool_stats.dev_norm_unit)}), and add μ back.</dd>
<dt>Scores</dt><dd>The explanation goes through <a href="https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar">the AR</a>, giving a reconstruction r. The <i>mean-removed cosine</i> is the cosine between r and the source vector after projecting μ out of both. Without that projection, any residual-shaped reconstruction looks similar to anything that has a μ component. The <i>match rank</i> asks whether r is closer to its own head than to the other 9 heads decoded at the same token.</dd>
</dl>
</section>`);

// ---------- summary table ----------
p(`<section aria-labelledby="table" style="display:flex;flex-direction:column;gap:14px">
<h2 id="table">All heads at a glance</h2>
<p class="meta prose">“Content hit” marks the heads whose decodes, in my judgment, track the tokens the head attends to. The other columns are computed.</p>
<div class="table-wrap"><table>
<thead><tr><th>Text</th><th>Head</th><th>Norm</th><th>Top attended token</th><th>cos(head, resid)</th><th>AR cos, raw</th><th>AR cos, mean+</th><th>Match raw</th><th>Match mean+</th><th>My read</th></tr></thead><tbody>`);
for (const ex of res.examples) {
  for (const it of items.filter((x) => x.ex === ex.idx && x.kind === "head" && x.variant === "raw")) {
    const mp = items.find((x) => x.ex === ex.idx && x.label === it.label && x.variant === "mean+");
    const a = it.attn_top[0];
    const hit = hits[`${ex.idx}|${it.label}`];
    p(`<tr><td>${ex.idx + 1}</td><td><a href="#h-${ex.idx}-${it.label}">${it.label}</a></td><td class="num">${f(it.raw_norm)}</td><td>${tokChip(a.tok)} <span class="meta">${f(a.w)}</span></td><td class="num">${f(it.cos_with_resid)}</td><td class="num">${f(it.ar_cos_perp_src, 3)}</td><td class="num">${f(mp.ar_cos_perp_src, 3)}</td><td class="num">${it.match_rank}/${it.match_of}</td><td class="num">${mp.match_rank}/${mp.match_of}</td><td>${hit ? `<span class="chip">${esc(hit)}</span>` : ""}</td></tr>`);
  }
}
p(`</tbody></table></div></section>`);

// ---------- per-example ----------
p(`<nav class="toc" aria-label="Examples">${res.examples.map((ex) => `<a href="#ex-${ex.idx}">${ex.idx + 1}. ${esc(ex.name)}</a>`).join("")}<a href="#controls">Controls</a></nav>`);

const decodeBlock = (it, title, extraClass = "") => {
  const score = it.kind === "head" ? `AR ${f(it.ar_cos_perp_src, 3)} · rank ${it.match_rank}/${it.match_of}` : it.ar_cos_perp_src != null ? `AR ${f(it.ar_cos_perp_src, 3)}` : `AR cos ${f(it.ar_cos_input, 3)}`;
  return `<div class="decode ${extraClass}"><div class="label"><span>${title}</span><span class="score">${score}</span></div><div class="text">${expl(it.greedy)}</div></div>`;
};

for (const ex of res.examples) {
  const ctx = ex.context;
  const pre = ctx.endsWith(ex.token) ? ctx.slice(0, ctx.length - ex.token.length) : ctx;
  const tokText = ctx.endsWith(ex.token) ? ex.token : "";
  p(`<section class="example" id="ex-${ex.idx}">
<div class="prose"><div class="eyebrow">Text ${ex.idx + 1} · token ${ex.pos} of ${ex.n_tokens}</div><h2>${esc(ex.name)}</h2>
<p class="meta">The highlighted token is the position whose activations are decoded. Faded text comes after it, so the model has not seen it yet. Layer-20 residual norm ${f(ex.resid_norm, 1)}; median head-output norm at this token ${f(ex.head_norm_median)}.</p></div>
<div class="context">${esc(pre)}<mark>${esc(tokText)}</mark><span class="after">${esc(ex.after)}</span></div>`);

  const residRaw = items.find((x) => x.ex === ex.idx && x.kind === "resid" && x.variant === "raw");
  p(`<div class="card"><div class="card-head"><h3>Baseline: the full layer-20 residual</h3><span class="meta">what the NLA was trained on</span></div>
<div class="decodes">${decodeBlock(residRaw, "Residual", "baseline")}</div></div>`);

  for (const it of items.filter((x) => x.ex === ex.idx && x.kind === "head" && x.variant === "raw")) {
    const mp = items.find((x) => x.ex === ex.idx && x.label === it.label && x.variant === "mean+");
    const hit = hits[`${ex.idx}|${it.label}`];
    const s = sampledRaw.get(`${ex.idx}|${it.label}`);
    p(`<div class="card" id="h-${ex.idx}-${it.label}">
<div class="card-head"><h3>Layer ${it.layer}, head ${it.head} <span class="mono meta">(${it.label})</span></h3>${hit ? `<span class="chip">${esc(hit)}</span>` : ""}</div>
<div class="facts">
<div class="row">Norm ${f(it.raw_norm)} (rank ${it.norm_rank_in_layer} of 28 in its layer) · cos with residual ${f(it.cos_with_resid)} · cos with mean residual ${f(it.cos_with_mu)}</div>
<div class="row">Attends to: ${it.attn_top.map((a) => `${tokChip(a.tok)}<span class="mono">@${a.src_pos} ${f(a.w)}</span>`).join(" ")}</div>
<div class="row lens">Direct logit lens: ${it.lens_top.slice(0, 6).map(tokChip).join(" ")}</div>
</div>
<div class="decodes">${decodeBlock(it, "Raw head output")}${decodeBlock(mp, "Mean + head direction")}</div>
${s ? `<details><summary>Raw head output, sampled at T=1</summary><div class="text">${expl(s)}</div></details>` : ""}
</div>`);
  }
  p(`</section>`);
}

// ---------- controls ----------
p(`<section class="example" id="controls"><div class="prose"><div class="eyebrow">Controls</div><h2>Vectors with no head behind them</h2>
<p class="meta">Random Gaussian directions show what the AV produces when there is nothing to find. The mean residual shows what the mean+ variant adds to every head.</p></div>`);
const mu = items.find((x) => x.kind === "mean");
p(`<div class="card"><div class="card-head"><h3>Mean residual direction μ</h3></div><div class="decodes">${decodeBlock(mu, "μ alone")}</div></div>`);
for (const label of [...new Set(items.filter((x) => x.kind === "random").map((x) => x.label))]) {
  const raw = items.find((x) => x.label === label && x.variant === "raw");
  const mp = items.find((x) => x.label === label && x.variant === "mean+");
  const s = sampledRaw.get(`null|${label}`);
  p(`<div class="card"><div class="card-head"><h3>${esc(label.replace("random gaussian", "Random Gaussian"))}</h3></div>
<div class="decodes">${decodeBlock(raw, "Raw")}${decodeBlock(mp, "Mean + direction")}</div>
${s ? `<details><summary>Raw, sampled at T=1</summary><div class="text">${expl(s)}</div></details>` : ""}</div>`);
}
p(`</section>`);

p(`<footer class="prose"><p>Ran on a Modal H100 with HF transformers 4.57.1. The NLA weights are from Fraser-Taliente et al., <a href="https://transformer-circuits.pub/2026/nla/index.html">Natural Language Autoencoders</a> (2026).</p></footer>`);
p(`</main>`);
process.stdout.write(P.join("\n") + "\n");
