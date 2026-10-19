// node examples.mjs ../results/results_200.json ../judge/judgments.jsonl > examples_200.md
import fs from "node:fs";
import crypto from "node:crypto";

const res = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const judg = new Map();
for (const line of fs.readFileSync(process.argv[3], "utf8").split("\n")) {
  if (!line.trim()) continue;
  const r = JSON.parse(line);
  if (!r.error) judg.set(r.key, r);
}
const nEx = res.examples.length;

const out = [];
const p = (s = "") => out.push(s);
const f = (x, d = 2) => (x == null ? "–" : x.toFixed(d));
const q = (s) => JSON.stringify(s);
const esc = (s) => s.replace(/[\\`*_[\]<>~$]/g, (c) => `\\${c}`);
const quote = (s) =>
  s
    .split("\n")
    .map((l) => (l.trim() ? `> ${esc(l.trim())}` : ">"))
    .join("\n");

// the passage as the judge saw it: system prompt dropped, chat markup as a transcript, final token marked ⟦like this⟧
function passage(ex) {
  const ctx = ex.context;
  const marked = ctx.endsWith(ex.token) ? `${ctx.slice(0, ctx.length - ex.token.length)}⟦${ex.token}⟧` : `${ctx}⟦⟧`;
  return marked
    .replace(/<\|im_start\|>system\n[\s\S]*?<\|im_end\|>\n/, "")
    .replace(/<\|im_start\|>user\n/g, "User: ")
    .replace(/<\|im_end\|>\n?/g, "\n")
    .replace(/<\|im_start\|>assistant\n/g, "Assistant: ")
    .trim();
}
function showPassage(exIdx) {
  const ex = res.examples[exIdx];
  p(`Text ${exIdx + 1}, *${esc(ex.name)}*:`);
  p();
  if (ex.name.startsWith("code")) {
    p("```text");
    p(passage(ex));
    p("```");
  } else {
    p(passage(ex).split("\n").map((l) => `> ${esc(l.trim())}`).join("  \n"));
  }
  p();
  const next = ex.after.length > 90 ? ex.after.slice(0, ex.after.lastIndexOf(" ", 90)) + " …" : ex.after;
  p(`The text actually continues: "${esc(next.replace(/\n/g, "↵"))}"`);
  p();
}

const item = (kind, variant, ex, label) =>
  res.items.find((it) => it.kind === kind && it.variant === variant && it.label === label && (ex == null || it.ex === ex));
const unitEx = (it) => (it.kind === "random" ? Number(it.label.match(/#(\d+)/)[1]) % nEx : it.ex);
const unitId = (it) => `${it.kind}|${it.variant}|${unitEx(it)}|${it.label}`;

function verdict(it) {
  const id = unitId(it);
  const t = judg.get(`${id}|rubric_true`);
  const s = judg.get(`${id}|rubric_shuf`);
  const a = judg.get(`${id}|afc`);
  if (!t) return "not judged";
  const bits = [`judge ${t.score}/3 (on a wrong passage ${s?.score ?? "–"}/3)`];
  if (a) bits.push(`forced choice ${a.is_correct ? "right" : "wrong"} (${a.confidence} confidence)`);
  if (t.matching_elements.length) bits.push(`matched: ${t.matching_elements.map((m) => `“${m.replace(/^"(.*)"$/, "$1")}”`).join(", ")}`);
  return esc(bits.join(" · "));
}
function showDecode(it, name) {
  const ar = it.kind === "head" ? ` · AR ranks this head ${it.match_rank} of ${it.match_of}` : "";
  p(`**${name}** · ${verdict(it)}${ar}`);
  p();
  p(quote(it.greedy.replace(/<\/?explanation>/g, "").trim()));
  p();
}
function showHead(ex, label, variants = ["raw", "mean+"]) {
  const raw = item("head", "raw", ex, label);
  p(`### ${label} on text ${ex + 1}`);
  p();
  p(
    `Layer ${raw.layer}, head ${raw.head} · output norm ${f(raw.raw_norm)} (rank ${raw.norm_rank_in_layer}/28 in its layer) · attends to ` +
      raw.attn_top.slice(0, 3).map((a) => `${q(a.tok)} (${f(a.w)})`).join(", "),
  );
  p();
  showPassage(ex);
  for (const v of variants) showDecode(item("head", v, ex, label), v === "raw" ? "Raw" : "Mean+");
}

// ---------- picks ----------
// residual decodes: the judge's best, a typical one, and one it under-scored
const RESID = [17, 0, 6];
// every head here has at least one decode the judge scored 1 or more on its own passage
const BEST = [
  [13, "L13H16"],
  [4, "L9H25"],
  [2, "L4H1"],
  [10, "L9H26"],
  [11, "L18H3"],
  [14, "L21H5"],
];
// typical: a seeded pick of heads both of whose decodes scored 0, one from each of these texts
const typical = [2, 5, 8, 15].map((ex) => {
  const zero = res.items.filter((it) => {
    if (it.kind !== "head" || it.variant !== "raw" || it.ex !== ex) return false;
    const m = item("head", "mean+", ex, it.label);
    return judg.get(`${unitId(it)}|rubric_true`)?.score === 0 && judg.get(`${unitId(m)}|rubric_true`)?.score === 0;
  });
  const h = (s) => crypto.createHash("sha256").update(s).digest("hex");
  return [ex, zero.sort((a, b) => (h(a.label) < h(b.label) ? -1 : 1))[0].label];
});
const cjk = (s) => (s.match(/[㐀-鿿]/g) ?? []).length;
const CHINESE = res.items
  .filter((it) => it.kind === "head" && it.variant === "raw")
  .sort((a, b) => cjk(b.greedy) - cjk(a.greedy))
  .filter((it) => !BEST.some(([e, l]) => e === it.ex && l === it.label))
  .slice(0, 2);

// ---------- page ----------
p(`# Example NLA decodes of attention-head outputs`);
p();
p(
  `Decodes from the 200-head run (\`results/results_200.json\`), with the Haiku judge's verdicts from \`judge/judgments.jsonl\`. See \`README.md\` for the method and the aggregate numbers.`,
);
p();
p(
  `Each decode is the greedy output of the layer-20 NLA verbalizer when a vector is written over its \`㈎\` token embedding. Every vector was taken at the token marked ⟦like this⟧ in its passage. The verbalizer always writes three paragraphs: the format and topic of the text, a phrase it thinks is in the text (often invented), and the final token with a guess at what comes next.`,
);
p();
p(`- **Raw**: the head's output (its 128-dim slice of the attention output times its 128 columns of \`o_proj\`), rescaled to norm 150.`);
p(`- **Mean+**: the mean residual direction plus the head's direction with that component removed, so it looks more like a real residual.`);
p(`- **Judge**: Haiku's 0–3 rubric score against the true passage, the same score against a different passage (how lenient it is), and whether it picked the true passage over a distractor in a forced choice. It runs strict: the residual decode of text 7 below names the exact final token and scored 1.`);
p(`- **AR rank**: round-tripping the decode through the NLA reconstructor and comparing with all 10 sampled heads of the same text, where this head ranks (1 = best match).`);
p();

p(`## Positive control: the full layer-20 residual`);
p();
p(`What the NLA does with the input it was trained on. These get the topic, often the exact final token, and a plausible continuation.`);
p();
for (const ex of RESID) {
  p(`### Residual, text ${ex + 1}`);
  p();
  showPassage(ex);
  showDecode(item("resid", "raw", ex, "layer-20 residual (baseline)"), "Residual");
}

p(`## The best head decodes`);
p();
p(
  `Six of the 16 heads where at least one decode scored above 0. What gets through is usually a word the head attends to strongly: L13H16 attends to "lock" and " air" and the decodes talk about an airlock; L21H5 attends to " snow" and " white" and both decodes mention snow. The rest of each decode is still invented.`,
);
p();
for (const [ex, label] of BEST) showHead(ex, label);

p(`## Typical head decodes`);
p();
p(`Most head decodes look like these: fluent, on-template, and about something unrelated to the passage. The judge scored both decodes of each 0.`);
p();
for (const [ex, label] of typical) showHead(ex, label);

p(`## Raw decodes that drift into Chinese`);
p();
p(`32 of the 200 raw head decodes contain substantial Chinese text; none of the mean+ or residual decodes do. These are the two with the most Chinese characters, excluding heads shown above.`);
p();
for (const it of CHINESE) showHead(it.ex, it.label, ["raw"]);

p(`## Controls`);
p();
p(`Random Gaussian vectors are judged against an arbitrary passage (text = vector number mod 20).`);
p();
for (const v of ["raw", "mean+"]) {
  const it = item("random", v, null, "random gaussian #0");
  p(`### Random vector #0, ${v}`);
  p();
  showPassage(unitEx(it));
  showDecode(it, v === "raw" ? "Raw" : "Mean+");
}
p(`### The mean residual direction alone`);
p();
p(`The direction that mean+ adds to every head. Not judged, since it has no passage.`);
p();
p(quote(item("mean", "raw", null, "mean residual direction (control)").greedy.replace(/<\/?explanation>/g, "").trim()));
p();
process.stdout.write(out.join("\n"));
