// node analyze.mjs ../modal_app/results_200.json judgments.jsonl   -> summary.json, summary.md
import fs from "node:fs";

const [resultsPath, judgPath = "judgments.jsonl"] = process.argv.slice(2);
const res = JSON.parse(fs.readFileSync(resultsPath, "utf8"));
const nEx = res.examples.length;

// latest successful record per key (failed attempts are retried and appended later)
const byKey = new Map();
for (const line of fs.readFileSync(judgPath, "utf8").split("\n")) {
  if (!line.trim()) continue;
  const r = JSON.parse(line);
  if (!r.error) byKey.set(r.key, r);
}

const units = [];
for (const it of res.items) {
  if (it.kind === "mean" || (it.kind === "resid" && it.variant !== "raw")) continue;
  const ex = it.kind === "random" ? Number(it.label.match(/#(\d+)/)[1]) % nEx : it.ex;
  const id = `${it.kind}|${it.variant}|${ex}|${it.label}`;
  const get = (t) => byKey.get(`${id}|${t}`);
  const top = it.attn_top?.[0];
  units.push({
    id, kind: it.kind, variant: it.variant, ex, label: it.label, layer: it.layer, head: it.head,
    raw_norm: it.raw_norm, norm_rank: it.norm_rank_in_layer, ar_perp: it.ar_cos_perp_src, match_rank: it.match_rank,
    sink: top ? top.src_pos === 0 && top.w > 0.5 : null,
    attn_top: it.attn_top, text: it.greedy,
    t: get("rubric_true"), s: get("rubric_shuf"), a: get("afc"),
  });
}

const mean = (xs) => (xs.length ? xs.reduce((p, q) => p + q, 0) / xs.length : NaN);
function wilson(k, n, z = 1.96) {
  if (!n) return [NaN, NaN];
  const p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n), w = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [(c - w) / d, (c + w) / d];
}
function bootCI(xs, B = 4000) {
  if (xs.length < 2) return [NaN, NaN];
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
  const ms = [];
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < xs.length; i++) s += xs[Math.floor(rnd() * xs.length)];
    ms.push(s / xs.length);
  }
  ms.sort((p, q) => p - q);
  return [ms[Math.floor(0.025 * B)], ms[Math.floor(0.975 * B)]];
}

function stats(us) {
  const paired = us.filter((u) => u.t && u.s);
  const diffs = paired.map((u) => u.t.score - u.s.score);
  const afc = us.filter((u) => u.a);
  const k = afc.filter((u) => u.a.is_correct).length;
  const hi = afc.filter((u) => u.a.confidence !== "low");
  const khi = hi.filter((u) => u.a.is_correct).length;
  // the judge favors "A", so also average the hit rates for true=A and true=B (50% for an uninformed judge)
  const hitRate = (c) => mean(afc.filter((u) => u.a.correct === c).map((u) => +u.a.is_correct));
  return {
    n: us.length,
    n_judged: paired.length,
    rubric_true: mean(paired.map((u) => u.t.score)),
    rubric_shuf: mean(paired.map((u) => u.s.score)),
    rubric_diff: mean(diffs),
    rubric_diff_ci: bootCI(diffs),
    share_true_ge2: mean(paired.map((u) => +(u.t.score >= 2))),
    share_shuf_ge2: mean(paired.map((u) => +(u.s.score >= 2))),
    afc_n: afc.length,
    afc_acc: k / afc.length,
    afc_ci: wilson(k, afc.length),
    afc_chose_a: mean(afc.map((u) => +(u.a.choice === "A"))),
    afc_balanced_acc: (hitRate("A") + hitRate("B")) / 2,
    afc_confident_n: hi.length,
    afc_confident_acc: hi.length ? khi / hi.length : NaN,
  };
}

const groups = {
  "residual (positive control)": units.filter((u) => u.kind === "resid"),
  "heads, raw": units.filter((u) => u.kind === "head" && u.variant === "raw"),
  "heads, mean+": units.filter((u) => u.kind === "head" && u.variant === "mean+"),
  "random, raw (negative control)": units.filter((u) => u.kind === "random" && u.variant === "raw"),
  "random, mean+ (negative control)": units.filter((u) => u.kind === "random" && u.variant === "mean+"),
};
const rawHeads = groups["heads, raw"];
const mpHeads = groups["heads, mean+"];
const bucket = (l) => (l <= 6 ? "layers 0-6" : l <= 13 ? "layers 7-13" : l <= 20 ? "layers 14-20" : "layers 21-27");
const splits = {};
for (const [name, hs] of [["raw", rawHeads], ["mean+", mpHeads]]) {
  for (const b of ["layers 0-6", "layers 7-13", "layers 14-20", "layers 21-27"]) splits[`${name}: ${b}`] = hs.filter((u) => bucket(u.layer) === b);
  splits[`${name}: attends mostly to first token`] = hs.filter((u) => u.sink);
  splits[`${name}: attends elsewhere`] = hs.filter((u) => u.sink === false);
  splits[`${name}: AR match rank 1 of 10`] = hs.filter((u) => u.match_rank === 1);
  splits[`${name}: AR match rank 2-10`] = hs.filter((u) => u.match_rank > 1);
  splits[`${name}: head norm >= 3`] = hs.filter((u) => u.raw_norm >= 3);
  splits[`${name}: head norm < 3`] = hs.filter((u) => u.raw_norm < 3);
}

const summary = {
  model: "claude-haiku-5-5",
  n_records: byKey.size,
  groups: Object.fromEntries(Object.entries(groups).map(([k, v]) => [k, stats(v)])),
  splits: Object.fromEntries(Object.entries(splits).map(([k, v]) => [k, stats(v)])),
};
const ranked = (hs) =>
  hs.filter((u) => u.t && u.s).sort((p, q) => q.t.score - p.t.score || (q.t.score - q.s.score) - (p.t.score - p.s.score));
summary.top_raw_heads = ranked(rawHeads).slice(0, 15).map((u) => ({
  id: u.id, ex: u.ex, label: u.label, score: u.t.score, shuffled_score: u.s.score, afc_correct: u.a?.is_correct,
  matching_elements: u.t.matching_elements, attn_top: u.attn_top?.slice(0, 3).map((a) => a.tok),
}));
fs.writeFileSync("summary.json", JSON.stringify(summary, null, 1));

const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");
const pct = (x) => (Number.isFinite(x) ? `${(100 * x).toFixed(0)}%` : "–");
const row = (name, s) =>
  `| ${name} | ${s.n_judged} | ${f(s.rubric_true)} | ${f(s.rubric_shuf)} | ${f(s.rubric_diff)} [${f(s.rubric_diff_ci[0])}, ${f(s.rubric_diff_ci[1])}] | ${pct(s.share_true_ge2)} / ${pct(s.share_shuf_ge2)} | ${pct(s.afc_acc)} [${pct(s.afc_ci[0])}, ${pct(s.afc_ci[1])}] | ${pct(s.afc_balanced_acc)} | ${pct(s.afc_chose_a)} | ${pct(s.afc_confident_acc)} (n=${s.afc_confident_n}) |`;
const header = "| Group | n | Rubric, true passage | Rubric, wrong passage | Difference [95% CI] | Score ≥2 true / wrong | Forced choice acc. [95% CI] | Balanced acc. | Chose A | Acc. when not low confidence |\n|---|---|---|---|---|---|---|---|---|---|";
const md = [
  `# Haiku judgments of NLA head decodes`,
  ``,
  `${byKey.size} judgments. Rubric is 0-3; forced-choice chance is 50%. Balanced accuracy averages the hit rates when the true passage is A and when it is B, so the judge's preference for A can't move it off 50% by itself.`,
  ``,
  `## Main groups`,
  header,
  ...Object.entries(summary.groups).map(([k, s]) => row(k, s)),
  ``,
  `## Head splits`,
  header,
  ...Object.entries(summary.splits).map(([k, s]) => row(k, s)),
  ``,
  `## Highest-rated raw head decodes`,
  ...summary.top_raw_heads.map((u) => `- text ${u.ex + 1} ${u.label}: score ${u.score} (wrong passage ${u.shuffled_score}), forced choice ${u.afc_correct ? "right" : "wrong"}; attends ${u.attn_top?.map((t) => JSON.stringify(t)).join(" ")}; matches: ${u.matching_elements.map((m) => JSON.stringify(m)).join(", ")}`),
];
fs.writeFileSync("summary.md", md.join("\n") + "\n");
console.log(md.join("\n"));
