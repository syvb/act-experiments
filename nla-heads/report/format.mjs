// node format.mjs results.json > report.md
import fs from "node:fs";

const res = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = [];
const p = (s = "") => out.push(s);
const f = (x, d = 3) => (x == null ? "–" : x.toFixed(d));
const q = (s) => JSON.stringify(s);
const block = (s) => {
  p("```text");
  p((s ?? "(none)").replace(/<\/?explanation>/g, "").trim());
  p("```");
};
const variants = (ex, label) => res.items.filter((x) => x.ex === ex && x.label === label);
const byVariant = (its, v) => its.find((x) => (x.variant ?? "raw") === v);

const decodeSection = (its) => {
  for (const v of ["raw", "mean+"]) {
    const it = byVariant(its, v);
    if (!it) continue;
    const scores = [`AR cos(recon, input) ${f(it.ar_cos_input)}`];
    if (it.ar_cos_perp_src != null) scores.push(`AR mean-removed cos with ${it.kind === "resid" ? "residual" : "source vector"} ${f(it.ar_cos_perp_src)}`);
    if (it.kind === "head") scores.push(`mean-removed cos with context residual ${f(it.ar_cos_perp_resid)}`, `match rank ${it.match_rank}/${it.match_of}`);
    p(`**${v === "raw" ? "Raw vector" : "Mean + vector"}** (${scores.join(" · ")}):`);
    p();
    block(it.greedy);
    if (it.sampled_t1) {
      p(`Sampled, T=1:`);
      block(it.sampled_t1);
    }
    p();
  }
};

if (res.pool_stats) {
  const s = res.pool_stats;
  p(`Mean-direction estimate from ${s.n_pool} residuals: mean cos(residual, mean) = ${f(s.mean_cos_resid_mu)}; mean pairwise cos between residuals = ${f(s.mean_pairwise_cos_resid)}; typical off-mean component of a unit residual = ${f(s.dev_norm_unit)}.`);
  p();
}

for (const ex of res.examples) {
  p(`## Example ${ex.idx + 1}: ${ex.name}`);
  p();
  p(`Position ${ex.pos} of ${ex.n_tokens}, token ${q(ex.token)}. Layer-20 residual norm ${f(ex.resid_norm, 1)}; median head-output norm at this position (all layers) ${f(ex.head_norm_median, 2)}.`);
  p();
  p("Context the model has seen up to and including the token (the token is at the very end):");
  p();
  block(ex.context);
  p();
  const labels = [...new Set(res.items.filter((x) => x.ex === ex.idx).map((x) => x.label))];
  for (const label of labels) {
    const its = variants(ex.idx, label);
    const it = its[0];
    p(`### ${label}`);
    p();
    const bits = [`raw norm ${f(it.raw_norm, 2)}`];
    if (it.kind === "head") {
      bits.push(`norm rank in layer ${it.norm_rank_in_layer}/28`);
      bits.push(`cos(head, residual) ${f(it.cos_with_resid)}`);
    }
    if (it.cos_with_mu != null) bits.push(`cos(vector, mean residual) ${f(it.cos_with_mu)}`);
    p(`- ${bits.join(" · ")}`);
    if (it.attn_top) p(`- attends to: ${it.attn_top.map((a) => `${q(a.tok)}@${a.src_pos} (${f(a.w, 2)})`).join(", ")}`);
    if (it.lens_top) p(`- direct logit lens top: ${it.lens_top.map(q).join(" ")}`);
    p();
    decodeSection(its);
  }
}

p("## Controls");
p();
for (const label of [...new Set(res.items.filter((x) => x.ex === null).map((x) => x.label))]) {
  p(`### ${label}`);
  p();
  decodeSection(res.items.filter((x) => x.ex === null && x.label === label));
}
process.stdout.write(out.join("\n") + "\n");
