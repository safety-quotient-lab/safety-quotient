// SafetyQuotient — PSQ Content Evaluator
//
// Usage:
//   node src/run.js --content "some text"                  # single dimension (default: hostility_index)
//   node src/run.js --dimension all --content "some text"  # full PSQ profile (all 10 dimensions)
//   node src/run.js --dimension hostility_index,trust_conditions --content "text"
//   node src/run.js --provider openrouter                  # specific provider
//   node src/run.js --provider compare                     # run all available providers, compare
//   echo "some text" | node src/run.js --stdin             # pipe content in
//   node src/run.js --file data/berkeley-stratified.json   # run against a data file
//   node src/run.js --rpm 5                                # override requests-per-minute

import { readFileSync, existsSync } from "node:fs";
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
  const args = { provider: "openrouter", dimension: "hostility_index" };

  for (let i = 2; i < argv.length; i++) {
    switch (argv[i]) {
      case "--provider":
        args.provider = argv[++i];
        break;
      case "--dimension":
        args.dimension = argv[++i];
        break;
      case "--content":
        args.content = argv[++i];
        break;
      case "--context":
        args.context = argv[++i];
        break;
      case "--file":
        args.file = argv[++i];
        break;
      case "--stdin":
        args.stdin = true;
        break;
      case "--limit":
        args.limit = parseInt(argv[++i], 10);
        break;
      case "--rpm":
        args.rpm = parseInt(argv[++i], 10);
        break;
      default:
        if (!argv[i].startsWith("--")) {
          args.content = argv[i];
        }
    }
  }

  return args;
}

// --- Read stdin ---

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf-8").trim();
}

// --- Format single dimension result ---

function formatResult(result, groundTruth) {
  const lines = [];
  lines.push(`\n${"═".repeat(70)}`);
  lines.push(`PROVIDER: ${result.provider}  |  DIMENSION: ${result.dimension}  |  ${result.elapsed_ms}ms`);
  lines.push(`${"─".repeat(70)}`);
  lines.push(`SCORE: ${result.score}/10  |  CONFIDENCE: ${result.confidence}`);
  lines.push(`RATIONALE: ${result.rationale}`);

  if (result.subscale_scores) {
    lines.push(`\nSUBSCALE SCORES:`);
    for (const [subId, sub] of Object.entries(result.subscale_scores)) {
      const items = sub.triggered_items?.join(", ") || "none";
      lines.push(`  ${subId}: ${sub.score}/10  [items: ${items}]`);
      if (sub.evidence?.length) {
        for (const e of sub.evidence) {
          lines.push(`    → "${e}"`);
        }
      }
    }
  }

  if (groundTruth) {
    lines.push(`\nGROUND TRUTH: ${JSON.stringify(groundTruth)}`);
  }

  lines.push(`${"═".repeat(70)}`);
  return lines.join("\n");
}

// --- Format PSQ profile ---

function formatProfile(profile, provider) {
  const W = 70;
  const lines = [];
  lines.push(`\n${"█".repeat(W)}`);
  lines.push(`  PSQ PROFILE  |  PROVIDER: ${provider}`);
  lines.push(`${"█".repeat(W)}`);

  // Bar chart helper: score 0–10 mapped to 30-char bar
  const bar = (score, role) => {
    const filled = Math.round((score / 10) * 30);
    const ch = role === "protective" ? "▓" : "░";
    return ch.repeat(filled) + "·".repeat(30 - filled);
  };

  const dimLine = (d) => {
    const name = d.dimension.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    const padded = name.padEnd(24);
    const tag = d.included ? "" : "  [excluded]";
    return `    ${padded} ${String(d.score).padStart(2)}/10  ${bar(d.score, d.role)}  conf ${d.confidence}${tag}`;
  };

  lines.push(`\n  PROTECTIVE FACTORS (avg ${profile.protective_avg.toFixed(2)}/10):`);
  for (const d of profile.dimensions.filter(d => d.role === "protective")) {
    lines.push(dimLine(d));
  }

  lines.push(`\n  THREAT FACTORS (avg ${profile.threat_avg.toFixed(2)}/10 inverted):`);
  for (const d of profile.dimensions.filter(d => d.role === "threat")) {
    lines.push(dimLine(d));
  }

  lines.push(`\n${"─".repeat(W)}`);
  const psqBar = "█".repeat(Math.round(profile.psq / 100 * 40)) + "·".repeat(40 - Math.round(profile.psq / 100 * 40));
  lines.push(`  PSQ SCORE: ${profile.psq}/100  [${psqBar}]`);
  const exNote = profile.excluded > 0 ? `  (${profile.excluded} excluded, conf < 0.6)` : "";
  lines.push(`  Protective: ${profile.protective_avg.toFixed(2)}/10 (n=${profile.protective_n})  |  Threat: ${profile.threat_avg.toFixed(2)}/10 (n=${profile.threat_n})${exNote}`);
  lines.push(`${"█".repeat(W)}`);

  return lines.join("\n");
}

// --- Resolve dimension list ---

function resolveDimensions(instruments, dimArg) {
  if (dimArg === "all") {
    return getAllDimensionIds(instruments);
  }
  const ids = dimArg.split(",").map(s => s.trim());
  // Validate each
  for (const id of ids) {
    getDimension(instruments, id);
  }
  return ids;
}

// --- Main ---

async function main() {
  const args = parseArgs(process.argv);
  const instruments = loadInstruments();
  const dimensionIds = resolveDimensions(instruments, args.dimension);
  const isProfile = dimensionIds.length > 1;

  console.log(`\nSafetyQuotient — ${isProfile ? "Full PSQ Profile" : getDimension(instruments, dimensionIds[0]).name + " Detector"} (POC)`);
  console.log(`Provider: ${args.provider}`);
  console.log(`Dimensions: ${dimensionIds.join(", ")}`);

  // Determine content source
  let contentItems = [];

  if (args.content) {
    contentItems = [{ text: args.content, labels: null, stratum: "user-input" }];
  } else if (args.stdin) {
    const text = await readStdin();
    contentItems = [{ text, labels: null, stratum: "stdin" }];
  } else if (args.file) {
    const data = JSON.parse(readFileSync(args.file, "utf-8"));
    contentItems = Array.isArray(data) ? data : [data];
  } else {
    // Default: look for berkeley stratified samples
    const defaultFile = join(ROOT, "data", "berkeley-stratified.json");
    if (existsSync(defaultFile)) {
      contentItems = JSON.parse(readFileSync(defaultFile, "utf-8"));
    } else {
      console.log("\nNo content provided. Run one of:");
      console.log('  node src/run.js --content "your text here"');
      console.log("  node src/run.js --file data/berkeley-stratified.json");
      console.log("  echo 'text' | node src/run.js --stdin");
      console.log("  npm run fetch-samples  (downloads test data first)");
      process.exit(1);
    }
  }

  const limit = args.limit || contentItems.length;
  contentItems = contentItems.slice(0, limit);

  const totalCalls = contentItems.length * dimensionIds.length;
  console.log(`\nEvaluating ${contentItems.length} item(s) × ${dimensionIds.length} dimension(s) = ${totalCalls} API calls\n`);

  // Determine providers
  let providers = [];
  if (args.provider === "compare") {
    const available = [];
    if (process.env.OPENROUTER_API_KEY) available.push("openrouter");
    if (process.env.ANTHROPIC_API_KEY) available.push("claude");
    if (process.env.CLOUDFLARE_ACCOUNT_ID && process.env.CLOUDFLARE_API_TOKEN) available.push("workersai");

    if (available.length === 0) {
      console.error("No API keys configured in .env — can't compare providers");
      process.exit(1);
    }

    const rpmOpts = args.rpm ? { rpm: args.rpm } : {};
    providers = available.map(name => createProvider(name, process.env, rpmOpts));
    console.log(`Comparing providers: ${available.join(", ")}\n`);
  } else {
    const rpmOpts = args.rpm ? { rpm: args.rpm } : {};
    providers = [createProvider(args.provider, process.env, rpmOpts)];
  }

  // Run evaluations
  const allResults = []; // flat list of all results
  let callNum = 0;

  for (let i = 0; i < contentItems.length; i++) {
    const item = contentItems[i];
    const preview = item.text.slice(0, 80).replace(/\n/g, " ");
    console.log(`\n[${i + 1}/${contentItems.length}] ${item.stratum || "?"}: "${preview}..."`);

    for (const provider of providers) {
      const itemDimResults = [];

      for (const dimId of dimensionIds) {
        callNum++;
        const dimension = getDimension(instruments, dimId);
        const progress = `(${callNum}/${totalCalls})`;

        try {
          const result = await detect(provider, dimension, item.text, args.context);

          if (!isProfile) {
            // Single dimension: show full detail
            console.log(formatResult(result, item.labels));
          } else {
            // Multi-dimension: show compact line
            const name = dimension.name.padEnd(26);
            console.log(`  ${progress} ${name} → ${result.score}/10  conf ${result.confidence}  (${result.elapsed_ms}ms)`);
          }

          itemDimResults.push(result);
          allResults.push({ item, result });
        } catch (err) {
          console.error(`  ${progress} ERROR (${provider.name}/${dimId}): ${err.message}`);
          allResults.push({ item, error: err.message, provider: provider.name, dimension: dimId });
        }
      }

      // If profile mode, aggregate and display PSQ
      if (isProfile && itemDimResults.length > 0) {
        const profile = aggregatePSQ(itemDimResults);
        console.log(formatProfile(profile, provider.name));
      }
    }
  }

  // Summary
  if (allResults.length > 1) {
    console.log(`\n${"═".repeat(70)}`);
    console.log("SUMMARY");
    console.log(`${"─".repeat(70)}`);

    const scored = allResults.filter(r => r.result?.score !== undefined);
    if (scored.length > 0) {
      if (isProfile) {
        // Group by provider, then show per-dimension averages
        const byProvider = {};
        for (const r of scored) {
          const p = r.result.provider;
          if (!byProvider[p]) byProvider[p] = [];
          byProvider[p].push(r);
        }

        for (const [provider, items] of Object.entries(byProvider)) {
          console.log(`\n  ${provider}:`);
          const byDim = {};
          for (const r of items) {
            if (!byDim[r.result.dimension]) byDim[r.result.dimension] = [];
            byDim[r.result.dimension].push(r.result);
          }
          for (const [dimId, results] of Object.entries(byDim)) {
            const avg = (results.reduce((a, r) => a + r.score, 0) / results.length).toFixed(2);
            const avgConf = (results.reduce((a, r) => a + r.confidence, 0) / results.length).toFixed(2);
            const name = dimId.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()).padEnd(26);
            console.log(`    ${name} avg=${avg}  conf=${avgConf}  n=${results.length}`);
          }
        }
      } else {
        const byProvider = {};
        for (const r of scored) {
          const p = r.result.provider;
          if (!byProvider[p]) byProvider[p] = [];
          byProvider[p].push(r);
        }

        for (const [provider, items] of Object.entries(byProvider)) {
          const scores = items.map(r => r.result.score);
          const avg = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2);
          const confidences = items.map(r => r.result.confidence);
          const avgConf = (confidences.reduce((a, b) => a + b, 0) / confidences.length).toFixed(2);
          console.log(`  ${provider}: avg score=${avg}, avg confidence=${avgConf}, n=${items.length}`);
        }
      }
    }

    const errors = allResults.filter(r => r.error);
    if (errors.length > 0) {
      console.log(`\n  Errors: ${errors.length}`);
    }

    console.log(`${"═".repeat(70)}`);
  }
}

main().catch(err => {
  console.error(`Fatal: ${err.message}`);
  process.exit(1);
});
