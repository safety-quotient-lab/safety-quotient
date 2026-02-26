// SafetyQuotient — Validation & Correlation Script
//
// Runs Berkeley stratified samples through a dimension and computes:
//   1. Per-item scores + ground truth comparison
//   2. Pearson r correlation with hate_speech_score
//   3. Reliability test (3x same item)
//
// Usage:
//   node src/validate.js                          # full validation (hostility_index)
//   node src/validate.js --dimension all          # full PSQ profile per item
//   node src/validate.js --reliability-only       # just reliability test (3 runs)

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { config } from "dotenv";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

config({ path: join(ROOT, ".env") });

import { createProvider } from "./providers.js";
import { loadInstruments, getDimension, getAllDimensionIds, detect, aggregatePSQ } from "./detector.js";

// --- Parse CLI args ---

function parseArgs(argv) {
  const args = {
    provider: "openrouter",
    dimension: "hostility_index",
    reliabilityOnly: false,
    reliabilityRuns: 3,
    rpm: null,
  };

  for (let i = 2; i < argv.length; i++) {
    switch (argv[i]) {
      case "--provider":        args.provider = argv[++i]; break;
      case "--dimension":       args.dimension = argv[++i]; break;
      case "--reliability-only": args.reliabilityOnly = true; break;
      case "--reliability-runs": args.reliabilityRuns = parseInt(argv[++i], 10); break;
      case "--rpm":             args.rpm = parseInt(argv[++i], 10); break;
    }
  }

  return args;
}

// --- Pearson correlation ---

function pearsonR(xs, ys) {
  const n = xs.length;
  if (n < 3) return { r: NaN, n };

  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;

  let sumXY = 0, sumX2 = 0, sumY2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    sumXY += dx * dy;
    sumX2 += dx * dx;
    sumY2 += dy * dy;
  }

  const denom = Math.sqrt(sumX2 * sumY2);
  const r = denom === 0 ? 0 : sumXY / denom;

  return { r: Math.round(r * 1000) / 1000, n };
}

// --- Main ---

async function main() {
  const args = parseArgs(process.argv);
  const instruments = loadInstruments();

  const rpmOpts = args.rpm ? { rpm: args.rpm } : {};
  const provider = createProvider(args.provider, process.env, rpmOpts);

  const berkeleyData = JSON.parse(readFileSync(join(ROOT, "data", "berkeley-stratified.json"), "utf-8"));
  const reliabilityItem = JSON.parse(readFileSync(join(ROOT, "data", "reliability-item.json"), "utf-8"))[0];

  // --- Reliability test ---
  if (args.reliabilityOnly || true) {
    console.log(`\n${"═".repeat(70)}`);
    console.log("RELIABILITY TEST");
    console.log(`${"─".repeat(70)}`);
    console.log(`Item: "${reliabilityItem.text.slice(0, 60)}..."`);
    console.log(`Runs: ${args.reliabilityRuns}\n`);

    const isProfile = args.dimension === "all";
    const dimIds = isProfile ? getAllDimensionIds(instruments) : [args.dimension];
    const reliabilityScores = [];
    const reliabilityPSQs = [];

    for (let run = 1; run <= args.reliabilityRuns; run++) {
      console.log(`  Run ${run}/${args.reliabilityRuns}:`);

      if (isProfile) {
        const dimResults = [];
        for (const dimId of dimIds) {
          const dimension = getDimension(instruments, dimId);
          const result = await detect(provider, dimension, reliabilityItem.text);
          const name = dimension.name.padEnd(24);
          console.log(`    ${name} ${result.score}/10  conf ${result.confidence}`);
          dimResults.push(result);
        }
        const profile = aggregatePSQ(dimResults);
        console.log(`    → PSQ: ${profile.psq}/100`);
        reliabilityPSQs.push(profile.psq);
      } else {
        const dimension = getDimension(instruments, dimIds[0]);
        const result = await detect(provider, dimension, reliabilityItem.text);
        console.log(`    ${dimension.name}: ${result.score}/10  conf ${result.confidence}`);
        reliabilityScores.push(result.score);
      }
    }

    if (reliabilityScores.length > 0) {
      const mean = reliabilityScores.reduce((a, b) => a + b, 0) / reliabilityScores.length;
      const variance = reliabilityScores.reduce((a, s) => a + (s - mean) ** 2, 0) / reliabilityScores.length;
      const sd = Math.sqrt(variance);
      const range = Math.max(...reliabilityScores) - Math.min(...reliabilityScores);
      console.log(`\n  Reliability (${args.dimension}):`);
      console.log(`    Scores: [${reliabilityScores.join(", ")}]`);
      console.log(`    Mean: ${mean.toFixed(2)}, SD: ${sd.toFixed(2)}, Range: ${range}`);
    }

    if (reliabilityPSQs.length > 0) {
      const mean = reliabilityPSQs.reduce((a, b) => a + b, 0) / reliabilityPSQs.length;
      const variance = reliabilityPSQs.reduce((a, s) => a + (s - mean) ** 2, 0) / reliabilityPSQs.length;
      const sd = Math.sqrt(variance);
      const range = Math.max(...reliabilityPSQs) - Math.min(...reliabilityPSQs);
      console.log(`\n  Reliability (PSQ):`);
      console.log(`    PSQs: [${reliabilityPSQs.join(", ")}]`);
      console.log(`    Mean: ${mean.toFixed(2)}, SD: ${sd.toFixed(2)}, Range: ${range.toFixed(1)}`);
    }

    if (args.reliabilityOnly) {
      console.log(`${"═".repeat(70)}`);
      return;
    }
  }

  // --- Full validation run ---
  console.log(`\n${"═".repeat(70)}`);
  console.log("VALIDATION RUN");
  console.log(`${"─".repeat(70)}`);
  console.log(`Provider: ${provider.name}`);
  console.log(`Dimension: ${args.dimension}`);
  console.log(`Items: ${berkeleyData.length}\n`);

  const isProfile = args.dimension === "all";
  const dimIds = isProfile ? getAllDimensionIds(instruments) : [args.dimension];

  const results = [];
  let errors = 0;

  for (let i = 0; i < berkeleyData.length; i++) {
    const item = berkeleyData[i];
    const preview = item.text.slice(0, 60).replace(/\n/g, " ");
    console.log(`[${i + 1}/${berkeleyData.length}] ${item.stratum}: "${preview}..."`);

    try {
      if (isProfile) {
        const dimResults = [];
        for (const dimId of dimIds) {
          const dimension = getDimension(instruments, dimId);
          const result = await detect(provider, dimension, item.text);
          dimResults.push(result);
        }
        const profile = aggregatePSQ(dimResults);
        console.log(`  PSQ: ${profile.psq}/100  (prot ${profile.protective_avg.toFixed(1)}, threat ${profile.threat_avg.toFixed(1)}, excl ${profile.excluded})`);
        results.push({
          text: item.text.slice(0, 60),
          stratum: item.stratum,
          hate_speech_score: item.labels.hate_speech_score,
          psq: profile.psq,
          protective_avg: profile.protective_avg,
          threat_avg: profile.threat_avg,
          excluded: profile.excluded,
          dimResults,
        });
      } else {
        const dimension = getDimension(instruments, dimIds[0]);
        const result = await detect(provider, dimension, item.text);
        console.log(`  ${result.score}/10  conf ${result.confidence}  (ground truth: hs=${item.labels.hate_speech_score})`);
        results.push({
          text: item.text.slice(0, 60),
          stratum: item.stratum,
          hate_speech_score: item.labels.hate_speech_score,
          score: result.score,
          confidence: result.confidence,
        });
      }
    } catch (err) {
      errors++;
      console.log(`  ERROR: ${err.message}`);
    }
  }

  // --- Correlation ---
  console.log(`\n${"═".repeat(70)}`);
  console.log("CORRELATION ANALYSIS");
  console.log(`${"─".repeat(70)}`);

  if (isProfile) {
    // PSQ vs hate_speech_score (expect negative: higher PSQ = lower hate speech)
    const xs = results.map(r => r.hate_speech_score);
    const ys = results.map(r => r.psq);
    const corr = pearsonR(xs, ys);
    console.log(`\n  PSQ vs hate_speech_score:`);
    console.log(`    Pearson r = ${corr.r}  (n=${corr.n})`);
    console.log(`    Expected: negative (higher hate_speech → lower PSQ)`);

    // Also per-dimension correlations for threat dimensions
    console.log(`\n  Per-dimension correlations with hate_speech_score:`);
    for (const dimId of dimIds) {
      const dimScores = results.map(r => {
        const dr = r.dimResults.find(d => d.dimension === dimId);
        return dr ? dr.score : null;
      }).filter(s => s !== null);
      const hsScores = results.slice(0, dimScores.length).map(r => r.hate_speech_score);
      const c = pearsonR(hsScores, dimScores);
      const name = dimId.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()).padEnd(26);
      console.log(`    ${name} r=${c.r}`);
    }
  } else {
    // Single dim score vs hate_speech_score (expect negative: higher hate_speech → lower dim score)
    const xs = results.map(r => r.hate_speech_score);
    const ys = results.map(r => r.score);
    const corr = pearsonR(xs, ys);
    console.log(`\n  ${args.dimension} score vs hate_speech_score:`);
    console.log(`    Pearson r = ${corr.r}  (n=${corr.n})`);
    console.log(`    Expected: negative (higher hate_speech → lower ${args.dimension} score)`);

    // Scatter table
    console.log(`\n  Item-level results:`);
    console.log(`    ${"Stratum".padEnd(8)} ${"HS Score".padEnd(10)} ${"PSQ Score".padEnd(10)} ${"Conf".padEnd(6)} Text`);
    console.log(`    ${"─".repeat(70)}`);
    for (const r of results) {
      console.log(`    ${r.stratum.padEnd(8)} ${String(r.hate_speech_score).padEnd(10)} ${String(r.score).padEnd(10)} ${String(r.confidence).padEnd(6)} ${r.text}...`);
    }
  }

  // --- Summary ---
  console.log(`\n${"═".repeat(70)}`);
  console.log("SUMMARY");
  console.log(`${"─".repeat(70)}`);
  console.log(`  Items evaluated: ${results.length}`);
  console.log(`  Errors: ${errors}`);

  if (!isProfile && results.length > 0) {
    const scores = results.map(r => r.score);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const confs = results.map(r => r.confidence);
    const meanConf = confs.reduce((a, b) => a + b, 0) / confs.length;
    console.log(`  Mean score: ${mean.toFixed(2)}/10`);
    console.log(`  Mean confidence: ${meanConf.toFixed(2)}`);
    console.log(`  Score range: ${Math.min(...scores)} – ${Math.max(...scores)}`);

    // By stratum
    for (const stratum of ["high", "mid", "low"]) {
      const group = results.filter(r => r.stratum === stratum);
      if (group.length > 0) {
        const avg = (group.reduce((a, r) => a + r.score, 0) / group.length).toFixed(2);
        console.log(`  ${stratum.padEnd(6)} stratum avg: ${avg}/10  (n=${group.length})`);
      }
    }
  }

  if (isProfile && results.length > 0) {
    const psqs = results.map(r => r.psq);
    const mean = psqs.reduce((a, b) => a + b, 0) / psqs.length;
    console.log(`  Mean PSQ: ${mean.toFixed(2)}/100`);
    console.log(`  PSQ range: ${Math.min(...psqs)} – ${Math.max(...psqs)}`);

    for (const stratum of ["high", "mid", "low"]) {
      const group = results.filter(r => r.stratum === stratum);
      if (group.length > 0) {
        const avg = (group.reduce((a, r) => a + r.psq, 0) / group.length).toFixed(2);
        console.log(`  ${stratum.padEnd(6)} stratum avg PSQ: ${avg}/100  (n=${group.length})`);
      }
    }
  }

  console.log(`${"═".repeat(70)}`);
}

main().catch(err => {
  console.error(`Fatal: ${err.message}`);
  process.exit(1);
});
