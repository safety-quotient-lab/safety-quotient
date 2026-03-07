/**
 * PSQ Scoring Endpoint — HTTP server exposing StudentProvider ONNX inference
 * as psychology-agent/machine-response/v3 responses.
 *
 * Endpoints:
 *   GET  /health  — liveness check; reports model readiness and calibration version
 *   POST /score   — score text across all 10 PSQ dimensions
 *
 * Request body (POST /score):
 *   {
 *     "text": "<text to score>",          // required
 *     "session_id": "<client session id>",// optional — generated if absent
 *     "context": "<use case>"             // optional — enables v3.1 context-aware scoring
 *                                         // valid: "moderation" | "persuasion" | "negotiation"
 *                                         //        | "workplace" | "therapeutic"
 *   }
 *
 * Response: psychology-agent/machine-response/v3 JSON (v3.1 when context is specified)
 * Context-aware response adds: context_weighted_composite (0-10), context_weights_used
 *
 * Usage:
 *   PSQ_PORT=3000 node src/server.js
 *   npm run serve
 */

import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { StudentProvider } from "./student.js";
import { aggregatePSQ, CONFIDENCE_THRESHOLD } from "./detector.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const listeningPort = parseInt(process.env.PSQ_PORT || "3000", 10);
const listeningHost = process.env.PSQ_HOST || "127.0.0.1";

// Calibration metadata — matches calibration.json fitted 2026-03-06
const CALIBRATION_VERSION = "isotonic-v2-2026-03-06";
const CALIBRATION_METHOD = "isotonic regression per dimension, n=1897 val";
const VALIDATION_BASIS = "Dreaddit n=2760, Pearson r=0.684";

// Context-aware scoring: load weight configuration from src/context-weights.json
let contextWeightsConfig = {};
try {
  contextWeightsConfig = JSON.parse(
    readFileSync(join(ROOT, "src", "context-weights.json"), "utf-8")
  );
} catch {
  console.warn("[PSQ] context-weights.json not found — context-aware scoring unavailable");
}
const VALID_CONTEXTS = new Set(Object.keys(contextWeightsConfig));

/**
 * Compute a context-weighted composite score (0-10 scale).
 * Distinct from psq_composite (0-100, protective/threat formula in detector.js).
 * This is a simple weighted mean of per-dimension scores, elevated by context weights.
 *
 * Formula: Σ(weight_i × score_i) / Σ(weight_i)
 * Unspecified dimensions default to weight=1.0.
 */
function computeContextWeightedComposite(dimensionScores, contextName) {
  const config = contextWeightsConfig[contextName];
  if (!config) return null;
  const weights = config.weights;
  let weightedSum = 0;
  let totalWeight = 0;
  for (const [dimName, dimData] of Object.entries(dimensionScores)) {
    const weight = weights[dimName] ?? 1.0;
    weightedSum += weight * dimData.score;
    totalWeight += weight;
  }
  return Math.round((weightedSum / totalWeight) * 100) / 100;
}

// PSQ-Lite covers 3 dimensions (inferred from machine-response-v3-spec.md
// standard limitations block: 7 dims listed as NOT in PSQ-Lite)
const PSQ_LITE_DIMENSION_NAMES = new Set([
  "threat_exposure",
  "hostility_index",
  "trust_conditions",
]);

// Standard PSQ-Full limitations block — from machine-response-v3-spec.md
const STANDARD_LIMITATIONS = [
  {
    limitation_id: "confidence-is-static-r",
    severity: "MEDIUM",
    description:
      "Per-dimension confidence values are static held-out Pearson r estimates, not per-prediction reliability signals. The model confidence head is anti-calibrated (outputs constants regardless of input) and is discarded at inference time. Confidence = r-based reliability prior, constant per dimension across all inputs.",
    affected_dimensions: "all",
    mitigation:
      "Do not interpret confidence as varying by input text. Use confidence as a dimension-level reliability weight (higher r = more validated dimension). Composite score uses these weights correctly. See calibration_note on each dimension for the specific r value.",
  },
  {
    limitation_id: "dreaddit-distribution",
    severity: "MEDIUM",
    description:
      "Model trained on Dreaddit (Reddit stress posts). Performance on clinical, non-English, or non-Western populations is unvalidated.",
    affected_dimensions: "all",
    mitigation:
      "Flag WEIRD assumption for any non-Dreaddit-distribution text. Do not use for clinical decision support without additional validation.",
  },
  {
    limitation_id: "psq-lite-coverage-gap",
    severity: "MEDIUM",
    description:
      "PSQ-Lite covers 3 dimensions. PSQ-Full dimensions not in PSQ-Lite (e.g. energy_dissipation) may carry the highest clinical signal for some text types.",
    affected_dimensions: [
      "energy_dissipation",
      "cooling_capacity",
      "resilience_baseline",
      "defensive_architecture",
      "regulatory_capacity",
      "contractual_clarity",
      "authority_dynamics",
    ],
    mitigation:
      "Use dimensions[].psq_lite_mapped to identify coverage gaps when integrating PSQ-Lite and PSQ-Full in a triage pattern.",
  },
];

/**
 * Convert StudentProvider.score() output to psychology-agent/machine-response/v3 or v3.1.
 * When contextName is provided and valid, the response includes context_weighted_composite
 * and context_weights_used (v3.1 additive extension — backward-compatible with v3 consumers).
 */
function buildV3Response(studentResult, sessionId, contextName = null) {
  const { scores: dimensionScores, hierarchy, elapsed_ms } = studentResult;

  // Convert dimension score map to array for aggregatePSQ
  const dimensionResultsArray = Object.entries(dimensionScores).map(
    ([dimensionName, dimensionData]) => ({
      dimension: dimensionName,
      score: dimensionData.score,
      confidence: dimensionData.confidence,
    })
  );

  const aggregation = aggregatePSQ(dimensionResultsArray);

  const includedDimensionCount = dimensionResultsArray.filter(
    (dimensionResult) => dimensionResult.confidence >= CONFIDENCE_THRESHOLD
  ).length;

  const compositeStatus = includedDimensionCount > 0 ? "scored" : "excluded";
  const compositeValue =
    compositeStatus === "scored"
      ? Math.round(aggregation.psq * 10) / 10
      : null;

  const v3Dimensions = Object.entries(dimensionScores).map(
    ([dimensionName, dimensionData]) => ({
      dimension: dimensionName,
      score: dimensionData.score,
      raw_score: dimensionData.raw_score,
      confidence: dimensionData.confidence,
      meets_threshold: dimensionData.confidence >= CONFIDENCE_THRESHOLD,
      psq_lite_mapped: PSQ_LITE_DIMENSION_NAMES.has(dimensionName),
      psq_lite_dimension: PSQ_LITE_DIMENSION_NAMES.has(dimensionName)
        ? dimensionName
        : null,
      calibration_note: dimensionData.r_confidence !== null && dimensionData.r_confidence !== undefined
        ? `confidence = held-out Pearson r (static per dimension, not per-prediction); r=${dimensionData.r_confidence}`
        : null,
    })
  );

  // Context-aware scoring (v3.1 extension — additive; v3 consumers may ignore these fields)
  const contextWeightedComposite = contextName
    ? computeContextWeightedComposite(dimensionScores, contextName)
    : null;
  const contextWeightsUsed = contextName && contextWeightsConfig[contextName]
    ? contextWeightsConfig[contextName].weights
    : null;

  return {
    schema: contextName ? "psychology-agent/machine-response/v3.1" : "psychology-agent/machine-response/v3",
    session_id: sessionId,
    turn: 1,
    timestamp: new Date().toISOString(),
    from: "psq-sub-agent",
    to: "psychology-agent",

    scope_declaration: {
      in_scope:
        "Psychoemotional safety scoring across 10 dimensions for English text with distribution similar to Dreaddit (Reddit stress posts, r/stress, r/anxiety etc.)",
      out_of_scope:
        "Clinical diagnosis, individual mental health assessment, non-English text, clinical populations, non-Dreaddit-distribution text, authorship attribution",
      validation_basis: VALIDATION_BASIS,
    },

    source: {
      classification: "trusted",
      fetch_accessible: false,
      fetch_method: "http-post",
      source_confidence: 0.684,
      source_confidence_basis:
        "Held-out Pearson r=0.684, n=2760, Dreaddit validation set; isotonic calibration +3.5–21.6% MAE improvement per dimension",
    },

    scores: {
      calibration_applied: true,
      calibration_method: CALIBRATION_METHOD,
      calibration_version: CALIBRATION_VERSION,
      psq_composite: {
        value: compositeValue,
        scale: "0-100",
        status: compositeStatus,
        usable: compositeStatus === "scored",
      },
      ...(contextName && {
        context: contextName,
        context_weighted_composite: {
          value: contextWeightedComposite,
          scale: "0-10",
          note: "Weighted mean of per-dimension scores (0-10 scale). Distinct from psq_composite (0-100 protective/threat formula). Higher = safer on context-relevant dimensions.",
        },
        context_weights_used: contextWeightsUsed,
      }),
    },

    dimensions: v3Dimensions,
    limitations: STANDARD_LIMITATIONS,

    claims: [
      {
        claim_id: "psq-composite-score",
        text:
          compositeStatus === "scored"
            ? `PSQ-Full composite score: ${compositeValue}/100 (${includedDimensionCount} of 10 dimensions met confidence threshold, weighted by protective/threat role).`
            : "PSQ-Full composite excluded: no dimensions met confidence threshold (r-based proxy < 0.6). Do not use 50/100 fallback for decision-making.",
        confidence: 0.684,
        confidence_basis: "Held-out r=0.684 on Dreaddit distribution",
        independently_verified: false,
      },
    ],

    action_gate: {
      gate_condition: "psq_composite.usable === true",
      gate_status: compositeStatus === "scored" ? "open" : "closed",
      gate_note:
        compositeStatus === "scored"
          ? "Composite score available. Downstream use permitted subject to limitations (Dreaddit distribution, WEIRD assumptions, no clinical validation)."
          : "All dimensions excluded — confidence below threshold. Composite is 50/100 fallback placeholder. Use individual dimension scores with appropriate caution.",
    },

    setl: 0.05, // Structural inference output — negligible editorial layer
    epistemic_flags: [],

    // Extended field: hierarchical factor structure (3 levels + g-PSQ).
    // Not in v3 base schema — included as extension for psychology-agent consumers.
    hierarchy,

    elapsed_ms,
  };
}

// ---- HTTP helpers ----

function sendJsonResponse(res, statusCode, body) {
  const jsonBody = JSON.stringify(body, null, 2);
  res.writeHead(statusCode, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(jsonBody),
  });
  res.end(jsonBody);
}

async function readRequestBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf-8");
}

// ---- Provider initialization ----

const scoringProvider = new StudentProvider();
let providerIsReady = false;

async function initializeScoringProvider() {
  try {
    await scoringProvider.init();
    providerIsReady = true;
    console.log(
      `PSQ scoring endpoint ready — http://${listeningHost}:${listeningPort}`
    );
    console.log(`  POST /score  — score text (returns machine-response/v3)`);
    console.log(`  GET  /health — liveness check`);
  } catch (error) {
    console.error(`Failed to initialize StudentProvider: ${error.message}`);
    process.exit(1);
  }
}

// ---- Request handler ----

const server = createServer(async (req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    sendJsonResponse(res, 200, {
      status: "ok",
      model: "psq-student",
      calibration_version: CALIBRATION_VERSION,
      ready: providerIsReady,
    });
    return;
  }

  if (req.method === "POST" && req.url === "/score") {
    if (!providerIsReady) {
      sendJsonResponse(res, 503, {
        error: "Model initializing — retry in a moment",
      });
      return;
    }

    let parsedRequest;
    try {
      const rawBody = await readRequestBody(req);
      parsedRequest = JSON.parse(rawBody);
    } catch {
      sendJsonResponse(res, 400, { error: "Request body must be valid JSON" });
      return;
    }

    const { text, session_id: clientSessionId, context: requestedContext } = parsedRequest;

    if (!text || typeof text !== "string" || text.trim().length === 0) {
      sendJsonResponse(res, 400, {
        error: "Request must include a non-empty 'text' field (string)",
      });
      return;
    }

    // Validate optional context parameter
    let resolvedContext = null;
    if (requestedContext !== undefined && requestedContext !== null) {
      if (!VALID_CONTEXTS.has(requestedContext)) {
        sendJsonResponse(res, 400, {
          error: `Unknown context '${requestedContext}'. Valid values: ${[...VALID_CONTEXTS].join(", ")}`,
        });
        return;
      }
      resolvedContext = requestedContext;
    }

    const sessionId = clientSessionId || `psq-${randomUUID()}`;

    try {
      const studentResult = await scoringProvider.score(text.trim());
      const v3Response = buildV3Response(studentResult, sessionId, resolvedContext);
      sendJsonResponse(res, 200, v3Response);
    } catch (error) {
      console.error(`Inference error [${sessionId}]: ${error.message}`);
      sendJsonResponse(res, 500, {
        error: "Inference failed",
        detail: error.message,
        session_id: sessionId,
      });
    }
    return;
  }

  sendJsonResponse(res, 404, {
    error: "Not found",
    available_endpoints: ["GET /health", "POST /score"],
  });
});

await initializeScoringProvider();
server.listen(listeningPort, listeningHost);
