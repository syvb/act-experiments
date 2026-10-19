// Haiku judges NLA explanations of attention-head outputs.
//
//   node judge.mjs ../modal_app/results_200.json            # run (resumes from judgments.jsonl)
//   node judge.mjs ../modal_app/results_200.json --dry-run  # print sample prompts and call counts, no API calls
//
// Three judgments per decode:
//   rubric_true  0-3 score against the passage the vector came from
//   rubric_shuf  same decode scored against a different passage (judge-leniency baseline)
//   afc          which of two passages (true vs distractor, random order) the description came from
import fs from "node:fs";
import crypto from "node:crypto";
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";

const MODEL = "claude-haiku-5-5";
const PRICE_IN = 0.1 / 1e6; // $/token, prompts under 100K tokens
const PRICE_OUT = 0.5 / 1e6;
const CONCURRENCY = 16;
const OUT = process.env.JUDGE_OUT ?? "judgments.jsonl";

const [resultsPath, ...flags] = process.argv.slice(2);
const dryRun = flags.includes("--dry-run");
const res = JSON.parse(fs.readFileSync(resultsPath, "utf8"));
const nEx = res.examples.length;

// ---------- passages ----------
const MARK_L = "⟦";
const MARK_R = "⟧";
function passage(ex) {
  let ctx = ex.context;
  const tok = ex.token;
  const head = ctx.endsWith(tok) ? ctx.slice(0, ctx.length - tok.length) : ctx;
  const marked = ctx.endsWith(tok) ? `${head}${MARK_L}${tok}${MARK_R}` : `${ctx}${MARK_L}${MARK_R}`;
  // render Qwen chat markup as a plain transcript
  return marked
    .replace(/<\|im_start\|>system\n[\s\S]*?<\|im_end\|>\n/, "")
    .replace(/<\|im_start\|>user\n/g, "User: ")
    .replace(/<\|im_end\|>\n?/g, "\n")
    .replace(/<\|im_start\|>assistant\n/g, "Assistant: ")
    .trim();
}
const passages = res.examples.map(passage);
const explanation = (s) => s.replace(/<\/?explanation>/g, "").trim();

// ---------- units ----------
const units = [];
for (const it of res.items) {
  if (it.kind === "mean") continue;
  if (it.kind === "resid" && it.variant !== "raw") continue;
  let ex = it.ex;
  if (it.kind === "random") ex = Number(it.label.match(/#(\d+)/)[1]) % nEx; // controls get an arbitrary passage
  units.push({
    id: `${it.kind}|${it.variant}|${ex}|${it.label}`,
    kind: it.kind,
    variant: it.variant,
    ex,
    label: it.label,
    layer: it.layer ?? null,
    head: it.head ?? null,
    text: explanation(it.greedy),
  });
}
const SHUF = 7; // offsets coprime with nEx, so every passage is paired with a different one
const DISTRACT = 13;
const rand01 = (s) => crypto.createHash("sha256").update(s).digest().readUInt32BE(0) / 2 ** 32;

// ---------- prompts ----------
const SYSTEM = `You evaluate descriptions produced by an automated text-analysis tool. The tool is shown a text passage and writes a short description of what the passage up to a marked final token is about. Its descriptions follow a fixed template (format/genre, a quoted phrase, what comes after the final token) and often contain invented quotes and details. Your job is to judge how much real, specific information about a passage a description contains.

Ignore generic statements that would fit almost any text, such as "structured article format", "informal tone", "a list", "mid-sentence", or "expects a noun". Count only content that is specific to the passage: its topic, entities, objects, distinctive words, or the identity of the marked final token.`;

const rubricPrompt = (pass, desc) => `<passage>
${pass}
</passage>

The final token the tool was looking at is marked ${MARK_L}like this${MARK_R}.

<description>
${desc}
</description>

Rate how much specific, correct information about the passage the description contains:
0 = none: unrelated topics; any overlap is generic.
1 = weak: shares only the broad domain or genre (e.g. cooking, programming, news, fiction), with no specific content.
2 = partial: correctly mentions at least one specific element of the passage (a distinctive topic, entity, object, or word), mixed with unrelated material.
3 = strong: mostly about the passage's actual content, capturing several specific elements or the marked final token.

First list the specific elements of the description that match the passage (quote them briefly; empty list if none), then give the score.`;

const afcPrompt = (a, b, desc) => `<passage_A>
${a}
</passage_A>

<passage_B>
${b}
</passage_B>

In each passage the final token is marked ${MARK_L}like this${MARK_R}.

<description>
${desc}
</description>

The description was generated from exactly one of these two passages. Which one is it more likely to have been generated from? If nothing in it distinguishes them, still pick one and say your confidence is low.`;

const RubricSchema = z.object({
  matching_elements: z.array(z.string()),
  score: z.number().int().min(0).max(3),
});
const AfcSchema = z.object({
  choice: z.enum(["A", "B"]),
  confidence: z.enum(["low", "medium", "high"]),
});

function tasksFor(u) {
  const shufEx = (u.ex + SHUF) % nEx;
  const disEx = (u.ex + DISTRACT) % nEx;
  const trueFirst = rand01(u.id) < 0.5;
  return [
    { task: "rubric_true", schema: RubricSchema, prompt: rubricPrompt(passages[u.ex], u.text), meta: { passage_ex: u.ex } },
    { task: "rubric_shuf", schema: RubricSchema, prompt: rubricPrompt(passages[shufEx], u.text), meta: { passage_ex: shufEx } },
    {
      task: "afc",
      schema: AfcSchema,
      prompt: trueFirst ? afcPrompt(passages[u.ex], passages[disEx], u.text) : afcPrompt(passages[disEx], passages[u.ex], u.text),
      meta: { correct: trueFirst ? "A" : "B", distractor_ex: disEx },
    },
  ];
}

const jobs = units.flatMap((u) => tasksFor(u).map((t) => ({ ...t, unit: u, key: `${u.id}|${t.task}` })));

if (dryRun) {
  console.log(`${units.length} decodes, ${jobs.length} judge calls`);
  console.log(JSON.stringify(Object.fromEntries(Object.entries(Object.groupBy(units, (u) => `${u.kind}/${u.variant}`)).map(([k, v]) => [k, v.length]))));
  const h = jobs.find((j) => j.unit.kind === "head" && j.task === "rubric_true");
  const a = jobs.find((j) => j.unit.kind === "head" && j.task === "afc");
  console.log("\n===== SYSTEM =====\n" + SYSTEM);
  console.log("\n===== rubric_true example =====\n" + h.prompt);
  console.log("\n===== afc example (correct: " + a.meta.correct + ") =====\n" + a.prompt);
  console.log("\nschema:", JSON.stringify(zodOutputFormat(RubricSchema).schema));
  process.exit(0);
}

// ---------- run ----------
const done = new Set();
if (fs.existsSync(OUT)) {
  for (const line of fs.readFileSync(OUT, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const r = JSON.parse(line);
    if (!r.error) done.add(r.key);
  }
}
const todo = jobs.filter((j) => !done.has(j.key));
console.log(`${jobs.length} judge calls total, ${done.size} already done, ${todo.length} to run`);

const client = new Anthropic({ maxRetries: 6 });
const out = fs.createWriteStream(OUT, { flags: "a" });
let usageIn = 0;
let usageOut = 0;
let finished = 0;
let failed = 0;

async function judge(job) {
  const rec = { key: job.key, unit_id: job.unit.id, task: job.task, ...job.meta };
  try {
    const response = await client.messages.parse({
      model: MODEL,
      max_tokens: 4000,
      system: SYSTEM,
      output_config: { effort: "low", format: zodOutputFormat(job.schema) },
      messages: [{ role: "user", content: job.prompt }],
    });
    usageIn += response.usage.input_tokens;
    usageOut += response.usage.output_tokens;
    if (response.stop_reason === "refusal") {
      rec.refusal = response.stop_details?.category ?? true;
    } else if (!response.parsed_output) {
      rec.error = `no parsed output (stop_reason ${response.stop_reason})`;
    } else {
      Object.assign(rec, response.parsed_output);
      if (job.task === "afc") rec.is_correct = rec.choice === job.meta.correct;
    }
  } catch (error) {
    if (error instanceof Anthropic.AuthenticationError) {
      console.error("Authentication failed: set ANTHROPIC_API_KEY in this environment.");
      process.exit(1);
    } else if (error instanceof Anthropic.APIError) {
      rec.error = `API ${error.status}: ${error.message}`;
    } else {
      rec.error = String(error?.message ?? error);
    }
  }
  if (rec.error) failed++;
  out.write(JSON.stringify(rec) + "\n");
  finished++;
  if (finished % 50 === 0 || finished === todo.length) {
    const cost = usageIn * PRICE_IN + usageOut * PRICE_OUT;
    console.log(`${finished}/${todo.length} done, ${failed} failed, $${cost.toFixed(3)} so far`);
  }
}

let next = 0;
await Promise.all(
  Array.from({ length: CONCURRENCY }, async () => {
    while (next < todo.length) await judge(todo[next++]);
  }),
);
out.end();
const cost = usageIn * PRICE_IN + usageOut * PRICE_OUT;
console.log(`finished: ${usageIn} input + ${usageOut} output tokens, $${cost.toFixed(3)}; ${failed} failed (rerun to retry them)`);
