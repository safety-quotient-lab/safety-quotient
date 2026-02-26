// Fetch sample data from HuggingFace datasets for testing
// Usage: node src/fetch-samples.js [dataset] [count]
//
// Datasets:
//   berkeley   — UC Berkeley Measuring Hate Speech (10 subscales, continuous)
//   civil      — Jigsaw Civil Comments (continuous toxicity 0-1)
//   goemo      — GoEmotions (27 emotion labels)

import { writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(__dirname, "..", "data");

const DATASETS = {
  berkeley: {
    name: "UC Berkeley Measuring Hate Speech",
    url: "https://datasets-server.huggingface.co/rows?dataset=ucberkeley-dlab/measuring-hate-speech&config=default&split=train&offset=0&length=",
    extract: (row) => ({
      text: row.row.text,
      labels: {
        hate_speech_score: row.row.hate_speech_score,
        sentiment: row.row.sentiment,
        respect: row.row.respect,
        insult: row.row.insult,
        humiliate: row.row.humiliate,
        dehumanize: row.row.dehumanize,
        violence: row.row.violence,
        hatespeech: row.row.hatespeech,
        annotator_severity: row.row.annotator_severity
      }
    })
  },
  civil: {
    name: "Jigsaw Civil Comments",
    url: "https://datasets-server.huggingface.co/rows?dataset=google/civil_comments&config=default&split=train&offset=0&length=",
    extract: (row) => ({
      text: row.row.text,
      labels: {
        toxicity: row.row.toxicity,
        severe_toxicity: row.row.severe_toxicity,
        obscene: row.row.obscene,
        threat: row.row.threat,
        insult: row.row.insult,
        identity_attack: row.row.identity_attack,
        sexual_explicit: row.row.sexual_explicit
      }
    })
  },
  goemo: {
    name: "GoEmotions",
    url: "https://datasets-server.huggingface.co/rows?dataset=google-research-datasets/go_emotions&config=simplified&split=train&offset=0&length=",
    extract: (row) => ({
      text: row.row.text,
      labels: {
        labels: row.row.labels
      }
    })
  }
};

async function fetchSamples(datasetKey, count) {
  const dataset = DATASETS[datasetKey];
  if (!dataset) {
    console.error(`Unknown dataset: ${datasetKey}`);
    console.error(`Available: ${Object.keys(DATASETS).join(", ")}`);
    process.exit(1);
  }

  console.log(`Fetching ${count} samples from ${dataset.name}...`);

  const url = dataset.url + count;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`HuggingFace API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();

  if (!data.rows || data.rows.length === 0) {
    throw new Error("No rows returned from HuggingFace API");
  }

  const samples = data.rows.map(dataset.extract);

  mkdirSync(DATA_DIR, { recursive: true });
  const outPath = join(DATA_DIR, `${datasetKey}-samples.json`);
  writeFileSync(outPath, JSON.stringify(samples, null, 2));

  console.log(`Saved ${samples.length} samples to ${outPath}`);
  console.log(`Sample text preview: "${samples[0].text.slice(0, 100)}..."`);
  console.log(`Sample labels:`, samples[0].labels);

  return samples;
}

// --- Fetch a stratified sample: some toxic, some neutral, some positive ---

async function fetchStratified(datasetKey, perStratum) {
  const dataset = DATASETS[datasetKey];
  if (!dataset) {
    console.error(`Unknown dataset: ${datasetKey}`);
    process.exit(1);
  }

  // Fetch a larger batch and stratify
  const batchSize = perStratum * 20;
  console.log(`Fetching ${batchSize} samples from ${dataset.name} to stratify...`);

  const url = dataset.url + batchSize;
  const response = await fetch(url);
  const data = await response.json();

  if (!data.rows) throw new Error("No rows returned");

  const all = data.rows.map(dataset.extract);

  let high = [], mid = [], low = [];

  if (datasetKey === "berkeley") {
    high = all.filter(s => s.labels.hate_speech_score > 0.5).slice(0, perStratum);
    mid = all.filter(s => s.labels.hate_speech_score > -0.5 && s.labels.hate_speech_score <= 0.5).slice(0, perStratum);
    low = all.filter(s => s.labels.hate_speech_score <= -0.5).slice(0, perStratum);
  } else if (datasetKey === "civil") {
    high = all.filter(s => s.labels.toxicity > 0.5).slice(0, perStratum);
    mid = all.filter(s => s.labels.toxicity > 0.1 && s.labels.toxicity <= 0.5).slice(0, perStratum);
    low = all.filter(s => s.labels.toxicity <= 0.1).slice(0, perStratum);
  } else {
    // Default: just take sequential
    high = all.slice(0, perStratum);
    mid = all.slice(perStratum, perStratum * 2);
    low = all.slice(perStratum * 2, perStratum * 3);
  }

  const stratified = [
    ...high.map(s => ({ ...s, stratum: "high" })),
    ...mid.map(s => ({ ...s, stratum: "mid" })),
    ...low.map(s => ({ ...s, stratum: "low" }))
  ];

  mkdirSync(DATA_DIR, { recursive: true });
  const outPath = join(DATA_DIR, `${datasetKey}-stratified.json`);
  writeFileSync(outPath, JSON.stringify(stratified, null, 2));

  console.log(`Stratified sample: ${high.length} high, ${mid.length} mid, ${low.length} low`);
  console.log(`Saved to ${outPath}`);

  return stratified;
}

// --- CLI ---

const args = process.argv.slice(2);
const datasetKey = args[0] || "berkeley";
const mode = args[1] || "stratified";
const count = parseInt(args[2] || "5", 10);

if (mode === "stratified") {
  await fetchStratified(datasetKey, count);
} else {
  await fetchSamples(datasetKey, count);
}
