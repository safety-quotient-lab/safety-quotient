import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

export function loadInstruments() {
  const raw = readFileSync(join(ROOT, "instruments.json"), "utf-8");
  return JSON.parse(raw);
}

export function getDimension(instruments, dimensionId) {
  const dim = instruments.dimensions[dimensionId];
  if (!dim) {
    const available = Object.keys(instruments.dimensions).join(", ");
    throw new Error(`Unknown dimension: ${dimensionId}. Available: ${available}`);
  }
  return dim;
}

export function buildSystemPrompt(dimension) {
  const instrumentBlocks = dimension.instruments.map(inst => {
    const subscaleBlocks = inst.subscales.map(sub => {
      const itemBlocks = sub.items.map(item => {
        const indicators = item.textual_indicators.map(i => `      - ${i}`).join("\n");
        return `    ${item.id} [${item.direction}]: ${item.construct}\n    textual indicators:\n${indicators}`;
      }).join("\n\n");

      return `  ${sub.name} (${sub.id}):\n${itemBlocks}`;
    }).join("\n\n");

    return `INSTRUMENT: ${inst.name} (${inst.abbreviation}, ${inst.authors}, ${inst.year})\nConstruct: ${inst.construct}\n\n${subscaleBlocks}`;
  }).join("\n\n---\n\n");

  return `You are a psychoemotional impact evaluator (PJE framework). You measure the directional force content exerts on a specific dimension of psychoemotional safety — its impact vector. You are not diagnosing the author or the reader. You are measuring what this content does to the safety environment it enters.

DIMENSION: ${dimension.name}
${dimension.description}

REFERENCE INSTRUMENTS AND ITEMS:

${instrumentBlocks}

---

SCORING (0–10 scale):
  0 = ${dimension.scoring["0"]}
  1 = ${dimension.scoring["1"]}
  2 = ${dimension.scoring["2"]}
  3 = ${dimension.scoring["3"]}
  4 = ${dimension.scoring["4"]}
  5 = ${dimension.scoring["5"]}
  6 = ${dimension.scoring["6"]}
  7 = ${dimension.scoring["7"]}
  8 = ${dimension.scoring["8"]}
  9 = ${dimension.scoring["9"]}
  10 = ${dimension.scoring["10"]}

${dimension.calibration_anchors ? `CALIBRATION ANCHORS — use these to ground your scoring:
${dimension.calibration_anchors.map(a => `  Content: "${a.text}"
  → Score: ${a.expected_score}/10, Confidence: ${a.expected_confidence}
  → ${a.rationale}`).join("\n\n")}

` : ""}SCORING RULES — these override all other considerations:
1. 5 is the DEFAULT. Start at 5 and adjust only when specific textual evidence justifies it.
2. Scoring below 5 REQUIRES specific textual evidence of negative impact on THIS dimension.
3. Scoring above 5 REQUIRES specific textual evidence of positive impact on THIS dimension.
4. ABSENCE of signal ≠ negative signal. Content that does not address this dimension = score 5, confidence below 0.4.
5. Offensive content on one dimension (e.g. hostility) does NOT automatically lower scores on unrelated dimensions (e.g. contractual clarity).
6. Use the full 0–10 range. 0 and 10 are extremes. Differentiate severity: a slur (2) ≠ a direct threat (0).

RESPONSE FORMAT — valid JSON only, no other text:
{
  "dimension": "${dimension.id}",
  "score": <integer 0–10>,
  "confidence": <0.0 to 1.0>,
  "subscale_scores": {
    "<subscale_id>": {
      "score": <integer 0–10>,
      "triggered_items": ["<item_id>", ...],
      "evidence": ["<exact quote or pattern from content>", ...]
    }
  },
  "rationale": "<2-3 sentence explanation starting with which specific indicators were or were not triggered>",
  "instruments_used": ["<abbreviation>", ...],
  "disclaimer": "This evaluation assesses content, not individuals. It is not a clinical diagnosis."
}

FINAL CHECK before responding:
- Did you start at 5 and adjust based on evidence? If no evidence relates to this dimension, your score must be 5.
- Is your confidence proportional to how many instrument items actually matched? Few matches = low confidence.
- Would a different evaluator reach the same score from the same evidence?`;
}

export function buildUserPrompt(content, context) {
  let prompt = `CONTENT TO EVALUATE:\n\n${content}`;
  if (context) {
    prompt += `\n\nCONTEXT: ${context}`;
  }
  return prompt;
}

// --- Dimension classification ---

const THREAT_DIMENSIONS = new Set([
  "threat_exposure", "hostility_index", "authority_dynamics", "energy_dissipation"
]);

const PROTECTIVE_DIMENSIONS = new Set([
  "regulatory_capacity", "resilience_baseline", "trust_conditions",
  "cooling_capacity", "defensive_architecture", "contractual_clarity"
]);

// Factor structure from EFA (promax rotation, §26-27 of distillation-research.md)
// Three levels of abstraction — all empirically derived, perfect simple structure.
// DA (defensive_architecture) has no primary loading >0.35 at 5-factor level.
const FACTORS_5 = {
  hostility_threat:    ["hostility_index", "threat_exposure", "cooling_capacity"],
  relational_contract: ["contractual_clarity", "trust_conditions"],
  internal_resources:  ["resilience_baseline", "regulatory_capacity"],
  power_dynamics:      ["authority_dynamics"],
  stress_energy:       ["energy_dissipation"],
  // defensive_architecture loads weakly everywhere — assigned to internal_resources at 3-factor
};

const FACTORS_3 = {
  threat_hostility:    ["hostility_index", "threat_exposure"],
  relational_safety:   ["trust_conditions", "contractual_clarity", "authority_dynamics"],
  coping_resources:    ["resilience_baseline", "regulatory_capacity"],
  // CC, ED, DA have no primary loading at 3-factor level
};

const FACTORS_2 = {
  content_hazard:      ["hostility_index", "threat_exposure", "cooling_capacity",
                        "energy_dissipation", "defensive_architecture", "regulatory_capacity"],
  relational_safety:   ["trust_conditions", "contractual_clarity"],
  // AD, RB have no primary loading at 2-factor level
};

// Legacy alias for backward compatibility
const CLUSTERS = FACTORS_5;

export function getAllDimensionIds(instruments) {
  return Object.keys(instruments.dimensions);
}

// --- PSQ Aggregation ---
// Confidence-weighted averages:  wavg = sum(score_i * conf_i) / sum(conf_i)
//
// Each dimension score is 0–10. 5 = neutral.
// For threat dimensions, score is inverted (0 = max threat → 10, 10 = no threat → 0).
//
// Linear formula:
//   PSQ = ((protective_wavg - threat_wavg + MAX) / (2 * MAX)) * 100
//   When protective=10, threat=0 → (10-0+10)/20*100 = 100
//   When protective=0, threat=10 → (0-10+10)/20*100 = 0
//   When both=5 (neutral)        → (5-5+10)/20*100  = 50

export const MAX_SCORE = 10;
export const CONFIDENCE_THRESHOLD = 0.6;

export function aggregatePSQ(dimensionResults) {
  const protective = [];
  const threat = [];
  let excluded = 0;

  for (const r of dimensionResults) {
    const score = r.score;
    const conf = r.confidence ?? 0.5;
    if (score === undefined || score === null) continue;

    // Gate: exclude low-confidence dimensions from aggregation
    if (conf < CONFIDENCE_THRESHOLD) {
      excluded++;
      continue;
    }

    if (PROTECTIVE_DIMENSIONS.has(r.dimension)) {
      protective.push({ score, conf });
    } else if (THREAT_DIMENSIONS.has(r.dimension)) {
      // Invert: 0 (max threat) → 10 threat points, 10 (mitigates) → 0
      threat.push({ score: MAX_SCORE - score, conf });
    }
  }

  const wavg = (entries) => {
    if (entries.length === 0) return MAX_SCORE / 2; // neutral default
    const sumWeighted = entries.reduce((a, e) => a + e.score * e.conf, 0);
    const sumWeights = entries.reduce((a, e) => a + e.conf, 0);
    return sumWeights > 0 ? sumWeighted / sumWeights : MAX_SCORE / 2;
  };

  const protectiveAvg = wavg(protective);
  const threatAvg = wavg(threat);

  // Linear formula: maps [prot=0,threat=10] → 0 and [prot=10,threat=0] → 100
  const psq = Math.min(100, Math.max(0,
    ((protectiveAvg - threatAvg + MAX_SCORE) / (2 * MAX_SCORE)) * 100
  ));

  // Hierarchical reporting: 3 levels of factor structure + g-PSQ
  const dimLookup = {};
  for (const r of dimensionResults) {
    if (r.score !== undefined && r.score !== null) {
      dimLookup[r.dimension] = { score: r.score, confidence: r.confidence ?? 0.5 };
    }
  }

  // Compute cluster scores for a given factor map
  const computeClusters = (factorMap) => {
    const result = {};
    for (const [name, memberDims] of Object.entries(factorMap)) {
      let sumW = 0, sumC = 0, n = 0;
      for (const dim of memberDims) {
        const d = dimLookup[dim];
        if (!d) continue;
        sumW += d.score * d.confidence;
        sumC += d.confidence;
        n++;
      }
      result[name] = {
        score: Math.round((sumC > 0 ? sumW / sumC : 5) * 100) / 100,
        confidence: Math.round((n > 0 ? sumC / n : 0) * 1000) / 1000,
        dimensions: memberDims,
      };
    }
    return result;
  };

  // g-PSQ: confidence-weighted mean of all dimensions
  let gSumW = 0, gSumC = 0, gN = 0;
  for (const d of Object.values(dimLookup)) {
    gSumW += d.score * d.confidence;
    gSumC += d.confidence;
    gN++;
  }

  return {
    psq: Math.round(psq * 10) / 10,
    protective_avg: Math.round(protectiveAvg * 1000) / 1000,
    threat_avg: Math.round(threatAvg * 1000) / 1000,
    protective_n: protective.length,
    threat_n: threat.length,
    excluded,
    hierarchy: {
      // Three levels: 2-factor (most abstract) → 3-factor → 5-factor (most granular)
      factors_2: computeClusters(FACTORS_2),
      factors_3: computeClusters(FACTORS_3),
      factors_5: computeClusters(FACTORS_5),
      g_psq: {
        score: Math.round((gSumC > 0 ? gSumW / gSumC : 5) * 100) / 100,
        confidence: Math.round((gN > 0 ? gSumC / gN : 0) * 1000) / 1000,
      },
    },
    dimensions: dimensionResults.map(r => ({
      dimension: r.dimension,
      score: r.score,
      confidence: r.confidence,
      included: r.confidence >= CONFIDENCE_THRESHOLD,
      role: PROTECTIVE_DIMENSIONS.has(r.dimension) ? "protective" : "threat"
    }))
  };
}

export async function detect(provider, dimension, content, context) {
  const systemPrompt = buildSystemPrompt(dimension);
  const userPrompt = buildUserPrompt(content, context);

  const start = Date.now();
  const result = await provider.evaluate(systemPrompt, userPrompt);
  const elapsed = Date.now() - start;

  return {
    provider: provider.name,
    dimension: dimension.id,
    elapsed_ms: elapsed,
    ...result.parsed
  };
}
