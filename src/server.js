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
 *     "session_id": "<client session id>" // optional — generated if absent
 *   }
 *
 * Response: psychology-agent/machine-response/v3 JSON
 *
 * Usage:
 *   PSQ_PORT=3000 node src/server.js
 *   npm run serve
 */

import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { StudentProvider } from "./student.js";
import { aggregatePSQ, CONFIDENCE_THRESHOLD } from "./detector.js";

const listeningPort = parseInt(process.env.PSQ_PORT || "3000", 10);
const listeningHost = "127.0.0.1";

// Calibration metadata — matches calibration.json fitted 2026-03-06
const CALIBRATION_VERSION = "isotonic-v1-2026-03-06";
const CALIBRATION_METHOD = "isotonic regression per dimension, n=1897 val";
const VALIDATION_BASIS = "Dreaddit n=2760, Pearson r=0.684";

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
    limitation_id: "anti-calibration-confidence",
    severity: "HIGH",
    description:
      "Confidence outputs are anti-calibrated — all 10 dimensions return confidence < 0.6 regardless of text. Composite score falls back to 50/100 default when all dimensions excluded.",
    affected_dimensions: "all",
    mitigation:
      "Do not use per-dimension confidence values as reliability indicators. Use calibrated scores directly. Composite score is usable only when at least one dimension meets threshold.",
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
 * Convert StudentProvider.score() output to psychology-agent/machine-response/v3.
 */
function buildV3Response(studentResult, sessionId) {
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
      calibration_note: null,
    })
  );

  return {
    schema: "psychology-agent/machine-response/v3",
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

    const { text, session_id: clientSessionId } = parsedRequest;

    if (!text || typeof text !== "string" || text.trim().length === 0) {
      sendJsonResponse(res, 400, {
        error: "Request must include a non-empty 'text' field (string)",
      });
      return;
    }

    const sessionId = clientSessionId || `psq-${randomUUID()}`;

    try {
      const studentResult = await scoringProvider.score(text.trim());
      const v3Response = buildV3Response(studentResult, sessionId);
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
