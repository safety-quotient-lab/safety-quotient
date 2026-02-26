// SafetyQuotient — Bluesky Jetstream Ingestion
//
// Connects to Bluesky's Jetstream WebSocket and pipes real-time posts
// through the PSQ detector.
//
// Usage:
//   node src/ingest.js                                    # stream + evaluate (hostility_index only)
//   node src/ingest.js --dimension all                    # full PSQ profile per post
//   node src/ingest.js --dimension hostility_index,trust_conditions
//   node src/ingest.js --lang en                          # filter to English posts only
//   node src/ingest.js --min-length 80                    # skip short posts
//   node src/ingest.js --sample 0.1                       # sample 10% of posts
//   node src/ingest.js --dry-run                          # print posts without evaluating
//   node src/ingest.js --limit 10                         # stop after 10 evaluations
//   node src/ingest.js --rpm 8                            # override requests-per-minute

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { config } from "dotenv";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

config({ path: join(ROOT, ".env") });

import { createProvider } from "./providers.js";
import { loadInstruments, getDimension, getAllDimensionIds, detect, aggregatePSQ } from "./detector.js";

// --- Constants ---

const JETSTREAM_INSTANCES = [
  "wss://jetstream1.us-east.bsky.network/subscribe",
  "wss://jetstream2.us-east.bsky.network/subscribe",
  "wss://jetstream1.us-west.bsky.network/subscribe",
  "wss://jetstream2.us-west.bsky.network/subscribe",
];

// --- Parse CLI args ---

function parseArgs(argv) {
  const args = {
    provider: "openrouter",
    dimension: "hostility_index",
    lang: null,
    minLength: 40,
    sample: 1.0,
    dryRun: false,
    limit: Infinity,
    rpm: null,
  };

  for (let i = 2; i < argv.length; i++) {
    switch (argv[i]) {
      case "--provider":    args.provider = argv[++i]; break;
      case "--dimension":   args.dimension = argv[++i]; break;
      case "--lang":        args.lang = argv[++i]; break;
      case "--min-length":  args.minLength = parseInt(argv[++i], 10); break;
      case "--sample":      args.sample = parseFloat(argv[++i]); break;
      case "--dry-run":     args.dryRun = true; break;
      case "--limit":       args.limit = parseInt(argv[++i], 10); break;
      case "--rpm":         args.rpm = parseInt(argv[++i], 10); break;
    }
  }

  return args;
}

// --- Resolve dimensions ---

function resolveDimensions(instruments, dimArg) {
  if (dimArg === "all") return getAllDimensionIds(instruments);
  const ids = dimArg.split(",").map(s => s.trim());
  for (const id of ids) getDimension(instruments, id);
  return ids;
}

// --- Format compact result ---

function formatCompact(result) {
  return `${result.dimension}: ${result.score}/10 (conf ${result.confidence})`;
}

// --- Format PSQ profile (compact for streaming) ---

function formatStreamProfile(profile) {
  const dims = profile.dimensions
    .filter(d => d.included)
    .map(d => {
      const name = d.dimension.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
      return `    ${name.padEnd(24)} ${String(d.score).padStart(2)}/10  conf ${d.confidence}  (${d.role})`;
    });

  const excluded = profile.dimensions.filter(d => !d.included).length;
  const exNote = excluded > 0 ? `  (${excluded} excluded)` : "";

  return [
    `  PSQ: ${profile.psq}/100  |  Protective: ${profile.protective_avg.toFixed(1)}/10 (n=${profile.protective_n})  |  Threat: ${profile.threat_avg.toFixed(1)}/10 (n=${profile.threat_n})${exNote}`,
    ...dims,
  ].join("\n");
}

// --- Jetstream connection ---

function connectJetstream(onPost, onError) {
  const instance = JETSTREAM_INSTANCES[Math.floor(Math.random() * JETSTREAM_INSTANCES.length)];
  const url = `${instance}?wantedCollections=app.bsky.feed.post`;

  console.log(`Connecting to ${url}`);

  const ws = new WebSocket(url);

  ws.addEventListener("open", () => {
    console.log("Connected to Jetstream\n");
  });

  ws.addEventListener("message", (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (
        msg.kind === "commit" &&
        msg.commit?.operation === "create" &&
        msg.commit?.collection === "app.bsky.feed.post" &&
        msg.commit?.record?.text
      ) {
        onPost({
          text: msg.commit.record.text,
          did: msg.did,
          langs: msg.commit.record.langs || [],
          createdAt: msg.commit.record.createdAt,
          rkey: msg.commit.rkey,
        });
      }
    } catch (err) {
      // skip malformed messages
    }
  });

  ws.addEventListener("error", (err) => {
    onError(err);
  });

  ws.addEventListener("close", (event) => {
    console.log(`\nDisconnected (code ${event.code}). Reconnecting in 5s...`);
    setTimeout(() => connectJetstream(onPost, onError), 5000);
  });

  return ws;
}

// --- Main ---

async function main() {
  const args = parseArgs(process.argv);
  const instruments = loadInstruments();
  const dimensionIds = resolveDimensions(instruments, args.dimension);
  const isProfile = dimensionIds.length > 1;

  console.log(`\nSafetyQuotient — Bluesky Live Ingestion`);
  console.log(`Provider: ${args.provider}`);
  console.log(`Dimensions: ${dimensionIds.join(", ")}`);
  console.log(`Filters: lang=${args.lang || "any"}, minLength=${args.minLength}, sample=${args.sample}`);
  if (args.dryRun) console.log(`Mode: DRY RUN (no evaluations)`);
  if (args.limit < Infinity) console.log(`Limit: ${args.limit} evaluations`);

  const rpmOpts = args.rpm ? { rpm: args.rpm } : {};
  const provider = createProvider(args.provider, process.env, rpmOpts);

  let evalCount = 0;
  let skipCount = 0;
  let errorCount = 0;
  let postsSeen = 0;
  const startTime = Date.now();

  // Queue: posts arrive faster than we can evaluate, so buffer them
  const queue = [];
  let processing = false;

  async function processQueue() {
    if (processing) return;
    processing = true;

    while (queue.length > 0 && evalCount < args.limit) {
      const post = queue.shift();
      evalCount++;

      const preview = post.text.slice(0, 80).replace(/\n/g, " ");
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
      console.log(`\n[#${evalCount} | ${elapsed}s | seen:${postsSeen} skip:${skipCount} err:${errorCount}]`);
      console.log(`  @${post.did.slice(8, 20)}... "${preview}${post.text.length > 80 ? "..." : ""}"`);

      if (args.dryRun) continue;

      try {
        if (isProfile) {
          const dimResults = [];
          for (const dimId of dimensionIds) {
            const dimension = getDimension(instruments, dimId);
            const result = await detect(provider, dimension, post.text);
            dimResults.push(result);
          }
          const profile = aggregatePSQ(dimResults);
          console.log(formatStreamProfile(profile));
        } else {
          const dimension = getDimension(instruments, dimensionIds[0]);
          const result = await detect(provider, dimension, post.text);
          console.log(`  ${formatCompact(result)}`);
          if (result.rationale) {
            console.log(`  ${result.rationale.slice(0, 120)}`);
          }
        }
      } catch (err) {
        errorCount++;
        console.error(`  ERROR: ${err.message}`);
      }
    }

    processing = false;

    if (evalCount >= args.limit) {
      console.log(`\nReached limit of ${args.limit} evaluations. Exiting.`);
      process.exit(0);
    }
  }

  function onPost(post) {
    postsSeen++;

    // Language filter
    if (args.lang && !post.langs.includes(args.lang)) {
      skipCount++;
      return;
    }

    // Length filter
    if (post.text.length < args.minLength) {
      skipCount++;
      return;
    }

    // Sample filter
    if (args.sample < 1.0 && Math.random() > args.sample) {
      skipCount++;
      return;
    }

    queue.push(post);
    processQueue();
  }

  function onError(err) {
    console.error(`WebSocket error: ${err.message || err}`);
  }

  const ws = connectJetstream(onPost, onError);

  // Graceful shutdown
  process.on("SIGINT", () => {
    console.log(`\n\nShutting down...`);
    console.log(`Posts seen: ${postsSeen}`);
    console.log(`Evaluated: ${evalCount}`);
    console.log(`Skipped: ${skipCount}`);
    console.log(`Errors: ${errorCount}`);
    console.log(`Duration: ${((Date.now() - startTime) / 1000).toFixed(0)}s`);
    ws.close();
    process.exit(0);
  });
}

main().catch(err => {
  console.error(`Fatal: ${err.message}`);
  process.exit(1);
});
