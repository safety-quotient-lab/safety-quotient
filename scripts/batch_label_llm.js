// Batch LLM labeler — labels texts on all 10 PSQ dimensions via Claude
//
// Samples texts from composite-ground-truth.jsonl that don't yet have LLM labels,
// labels each on all 10 dimensions in a SINGLE API call (cost-efficient),
// and appends results to data/train-llm.jsonl.
//
// Usage:
//   node scripts/batch_label_llm.js --count 3500              # uses haiku by default, resumes
//   node scripts/batch_label_llm.js --count 100 --dry-run     # test without API calls
//   node scripts/batch_label_llm.js --count 500 --no-resume   # re-label already-labeled texts
//   node scripts/batch_label_llm.js --count 3500 --rpm 60     # slower rate limit
//   node scripts/batch_label_llm.js --model claude-sonnet-4-20250514  # use sonnet

import { readFileSync, appendFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { config } from "dotenv";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

config({ path: join(ROOT, ".env") });

import { loadInstruments, getAllDimensionIds, getDimension, buildSystemPrompt, buildUserPrompt } from "../src/detector.js";

// --- Parse CLI args ---

function parseArgs() {
  const args = { count: 100, rpm: 50, dryRun: false, resume: true, model: "claude-haiku-4-5-20251001" };
  const argv = process.argv;
  for (let i = 2; i < argv.length; i++) {
    switch (argv[i]) {
      case "--count": args.count = parseInt(argv[++i], 10); break;
      case "--rpm": args.rpm = parseInt(argv[++i], 10); break;
      case "--model": args.model = argv[++i]; break;
      case "--dry-run": args.dryRun = true; break;
      case "--resume": args.resume = true; break;
      case "--no-resume": args.resume = false; break;
      default: break;
    }
  }
  return args;
}

// --- Load existing LLM-labeled texts ---

function loadLabeledTexts() {
  const labeled = new Set();
  const llmPath = join(ROOT, "data", "train-llm.jsonl");
  if (existsSync(llmPath)) {
    const lines = readFileSync(llmPath, "utf-8").trim().split("\n");
    for (const line of lines) {
      if (!line.trim()) continue;
      const rec = JSON.parse(line);
      // Use first 200 chars as key (texts may be long)
      labeled.add(rec.text.slice(0, 200));
    }
  }
  return labeled;
}

// --- Sample texts from composite ---

function sampleTexts(count, alreadyLabeled) {
  const compositePath = join(ROOT, "data", "composite-ground-truth.jsonl");
  const lines = readFileSync(compositePath, "utf-8").trim().split("\n");

  // Collect unlabeled texts, track source for stratification
  const bySource = {};
  for (const line of lines) {
    if (!line.trim()) continue;
    const rec = JSON.parse(line);
    if (rec.teacher === "llm" || rec.teacher === "llm_labeled") continue;
    const key = rec.text.slice(0, 200);
    if (alreadyLabeled.has(key)) continue;
    const source = rec.source || "unknown";
    if (!bySource[source]) bySource[source] = [];
    bySource[source].push(rec.text);
  }

  console.log("\nAvailable unlabeled texts by source:");
  const sources = Object.entries(bySource).sort((a, b) => b[1].length - a[1].length);
  let totalAvailable = 0;
  for (const [src, texts] of sources) {
    console.log(`  ${src}: ${texts.length}`);
    totalAvailable += texts.length;
  }
  console.log(`  TOTAL: ${totalAvailable}`);

  if (totalAvailable < count) {
    console.log(`\n  WARNING: Only ${totalAvailable} unlabeled texts available (requested ${count})`);
    count = totalAvailable;
  }

  // Stratified sampling: proportional to source size, with minimum per source
  const sampled = [];
  const minPerSource = Math.min(50, Math.floor(count / sources.length));

  // First pass: minimum per source
  for (const [src, texts] of sources) {
    const shuffled = texts.sort(() => Math.random() - 0.5);
    const take = Math.min(minPerSource, shuffled.length);
    for (let i = 0; i < take; i++) {
      sampled.push({ text: shuffled[i], source: src });
    }
  }

  // Second pass: fill remaining proportionally
  const remaining = count - sampled.length;
  if (remaining > 0) {
    const sampledTexts = new Set(sampled.map(s => s.text.slice(0, 200)));
    const pool = [];
    for (const [src, texts] of sources) {
      for (const text of texts) {
        if (!sampledTexts.has(text.slice(0, 200))) {
          pool.push({ text, source: src });
        }
      }
    }
    pool.sort(() => Math.random() - 0.5);
    for (let i = 0; i < Math.min(remaining, pool.length); i++) {
      sampled.push(pool[i]);
    }
  }

  // Shuffle final order
  sampled.sort(() => Math.random() - 0.5);
  return sampled.slice(0, count);
}

// --- Build multi-dimension system prompt ---

function buildAllDimsPrompt(instruments) {
  const dimIds = getAllDimensionIds(instruments);
  const dimBlocks = dimIds.map(id => {
    const dim = getDimension(instruments, id);
    return `## ${dim.name} (${id})
${dim.description}

Scoring (0-10):
  0 = ${dim.scoring["0"]}
  5 = ${dim.scoring["5"]}
  10 = ${dim.scoring["10"]}
${dim.calibration_anchors ? `
Calibration examples:
${dim.calibration_anchors.slice(0, 3).map(a =>
  `  "${a.text.slice(0, 80)}..." → ${a.expected_score}/10 (${a.rationale.slice(0, 60)})`
).join("\n")}` : ""}`;
  }).join("\n\n");

  return `You are a psychoemotional impact evaluator (PJE framework). Score content on ALL 10 dimensions of the Psychoemotional Safety Quotient (PSQ).

For each dimension, score 0-10 where 5 is neutral/default. Only deviate from 5 when specific textual evidence justifies it. Absence of signal = 5 with low confidence.

DIMENSIONS:

${dimBlocks}

SCORING RULES:
1. 5 is the DEFAULT for each dimension. Adjust only with specific evidence.
2. Score below 5 requires evidence of negative impact on THAT dimension.
3. Score above 5 requires evidence of positive impact on THAT dimension.
4. Hostility does NOT automatically lower unrelated dimensions.
5. Use the full 0-10 range. Differentiate severity.
6. Confidence = proportion of instrument items that matched evidence. Few matches = low confidence.

RESPONSE FORMAT — valid JSON only, no other text:
{
  "dimensions": {
${dimIds.map(id => `    "${id}": { "score": <0-10>, "confidence": <0.0-1.0> }`).join(",\n")}
  }
}

CRITICAL: Return ONLY the JSON object. No explanation, no markdown, no preamble.`;
}

// --- Claude API caller ---

class BatchClaude {
  constructor(apiKey, rpm, model) {
    this.apiKey = apiKey;
    this.minInterval = 60000 / rpm;
    this.lastRequest = 0;
    this.client = null;
    this.model = model;
  }

  async init() {
    const Anthropic = (await import("@anthropic-ai/sdk")).default;
    this.client = new Anthropic({ apiKey: this.apiKey });
  }

  async evaluate(systemPrompt, userPrompt) {
    // Pace
    const now = Date.now();
    const elapsed = now - this.lastRequest;
    if (elapsed < this.minInterval) {
      await new Promise(r => setTimeout(r, this.minInterval - elapsed));
    }

    const maxRetries = 5;
    let backoff = 30000;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const response = await this.client.messages.create({
          model: this.model,
          max_tokens: 2048,
          system: systemPrompt,
          messages: [{ role: "user", content: userPrompt }],
        });
        this.lastRequest = Date.now();
        const text = response.content[0].text;
        const match = text.match(/\{[\s\S]*\}/);
        if (!match) throw new Error(`No JSON in response: ${text.slice(0, 200)}`);
        return JSON.parse(match[0]);
      } catch (err) {
        this.lastRequest = Date.now();
        const isRateLimit = err.status === 429 || err.message?.includes("429");
        if (isRateLimit && attempt < maxRetries) {
          console.log(`  [429] retry ${attempt}/${maxRetries} in ${(backoff / 1000).toFixed(0)}s`);
          await new Promise(r => setTimeout(r, backoff));
          backoff *= 2;
          continue;
        }
        throw err;
      }
    }
  }
}

// --- Main ---

async function main() {
  const args = parseArgs();
  console.log(`PSQ Batch LLM Labeler`);
  console.log(`  Count: ${args.count}`);
  console.log(`  RPM: ${args.rpm}`);
  console.log(`  Model: ${args.model}`);
  console.log(`  Dry run: ${args.dryRun}`);
  console.log(`  Resume: ${args.resume}`);

  const instruments = loadInstruments();
  const alreadyLabeled = loadLabeledTexts();
  console.log(`\nAlready labeled: ${alreadyLabeled.size} texts`);

  // Sample texts
  const texts = sampleTexts(args.count, alreadyLabeled);
  console.log(`\nSampled ${texts.length} texts for labeling`);

  if (args.dryRun) {
    console.log("\n[DRY RUN] Would label these texts:");
    for (let i = 0; i < Math.min(5, texts.length); i++) {
      console.log(`  ${i + 1}. [${texts[i].source}] ${texts[i].text.slice(0, 80)}...`);
    }
    console.log(`  ... and ${texts.length - 5} more`);
    const estTime = texts.length / args.rpm;
    console.log(`\nEstimated time: ${estTime.toFixed(0)} minutes (${(estTime / 60).toFixed(1)} hours)`);
    return;
  }

  // Build system prompt
  const systemPrompt = buildAllDimsPrompt(instruments);
  console.log(`\nSystem prompt: ${systemPrompt.length} chars`);

  // Init Claude
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("ERROR: Set ANTHROPIC_API_KEY in .env");
    process.exit(1);
  }
  const claude = new BatchClaude(apiKey, args.rpm, args.model);
  await claude.init();

  const outPath = join(ROOT, "data", "train-llm.jsonl");
  let labeled = 0;
  let errors = 0;
  const startTime = Date.now();

  console.log(`\nLabeling ${texts.length} texts...`);
  console.log(`  Output: ${outPath}`);
  console.log(`  ETA: ~${(texts.length / args.rpm).toFixed(0)} minutes\n`);

  for (let i = 0; i < texts.length; i++) {
    const { text, source } = texts[i];
    const userPrompt = `CONTENT TO EVALUATE:\n\n${text}`;

    try {
      const result = await claude.evaluate(systemPrompt, userPrompt);

      // Build record in train-llm.jsonl format
      const record = {
        text,
        source: source,
        teacher: "llm",
        dimensions: {},
      };

      const dims = result.dimensions || result;
      for (const [dimId, val] of Object.entries(dims)) {
        if (val && typeof val.score === "number") {
          record.dimensions[dimId] = {
            score: Math.max(0, Math.min(10, val.score)),
            confidence: Math.max(0, Math.min(1, val.confidence || 0.5)),
          };
        }
      }

      const dimCount = Object.keys(record.dimensions).length;
      if (dimCount > 0) {
        appendFileSync(outPath, JSON.stringify(record) + "\n");
        labeled++;
      } else {
        console.log(`  [${i + 1}] WARNING: no dimensions in response`);
        errors++;
      }

      // Progress
      if ((i + 1) % 10 === 0 || i === texts.length - 1) {
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = labeled / elapsed * 60;
        const eta = (texts.length - i - 1) / rate;
        process.stdout.write(
          `\r  ${i + 1}/${texts.length} labeled (${labeled} ok, ${errors} err) ` +
          `${rate.toFixed(1)}/min, ETA ${eta.toFixed(0)}min`
        );
      }
    } catch (err) {
      errors++;
      console.log(`\n  [${i + 1}] ERROR: ${err.message?.slice(0, 100)}`);
      // Continue with next text
    }
  }

  const totalTime = (Date.now() - startTime) / 1000;
  console.log(`\n\nDone!`);
  console.log(`  Labeled: ${labeled} texts (${errors} errors)`);
  console.log(`  Time: ${(totalTime / 60).toFixed(1)} minutes`);
  console.log(`  Rate: ${(labeled / totalTime * 60).toFixed(1)} texts/min`);
  console.log(`  Output: ${outPath}`);
  console.log(`  Total in train-llm.jsonl: ${alreadyLabeled.size + labeled} records`);
}

main().catch(err => {
  console.error("Fatal:", err);
  process.exit(1);
});
