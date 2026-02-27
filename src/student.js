/**
 * StudentProvider — local ONNX inference for PSQ scoring.
 *
 * Scores all 10 dimensions in one forward pass (~20ms, zero API cost).
 * Supports both DistilBERT and DeBERTa-v3-small encoders.
 *
 * Requirements:
 *   npm install onnxruntime-node @huggingface/transformers
 *
 * Usage:
 *   import { StudentProvider } from "./student.js";
 *   const provider = new StudentProvider();
 *   await provider.init();
 *   const result = await provider.score("Some text to evaluate");
 *   // result = { scores: { threat_exposure: { score: 4.2, confidence: 0.65 }, ... }, elapsed_ms: 18 }
 */
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const DIMENSIONS = [
  "threat_exposure", "hostility_index", "authority_dynamics",
  "energy_dissipation", "regulatory_capacity", "resilience_baseline",
  "trust_conditions", "cooling_capacity", "defensive_architecture",
  "contractual_clarity",
];

// Cluster definitions from halo experiment (§18 of distillation-research.md)
const CLUSTERS = {
  interpersonal_climate: ["authority_dynamics", "contractual_clarity", "trust_conditions", "threat_exposure"],
  internal_resources: ["regulatory_capacity", "resilience_baseline", "defensive_architecture"],
  bridge: ["cooling_capacity", "energy_dissipation", "hostility_index"],
};

export class StudentProvider {
  constructor(options = {}) {
    this.name = "student";
    this.modelDir = options.modelDir || join(ROOT, "models", "psq-student");
    this.quantized = options.quantized !== false; // default: use quantized
    this.maxLength = options.maxLength || 128;
    this.session = null;
    this.tokenizer = null;
    this.calibration = null; // per-dimension isotonic regression maps
  }

  async init() {
    const ort = await import("onnxruntime-node");

    // Load model config for max_length
    try {
      const config = JSON.parse(readFileSync(join(this.modelDir, "config.json"), "utf-8"));
      this.maxLength = config.max_length || this.maxLength;
    } catch { /* use default */ }

    // Load ONNX model
    const modelFile = this.quantized ? "model_quantized.onnx" : "model.onnx";
    const modelPath = join(this.modelDir, modelFile);

    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["cpu"],
    });

    // Load tokenizer via @huggingface/transformers (handles WordPiece, BPE, Unigram/SentencePiece)
    const { AutoTokenizer } = await import("@huggingface/transformers");
    const tokenizerDir = join(this.modelDir, "tokenizer");
    this.tokenizer = await AutoTokenizer.from_pretrained(tokenizerDir, { local_files_only: true });

    // Load calibration map if available (fitted by scripts/calibrate.py)
    try {
      const calPath = join(this.modelDir, "calibration.json");
      this.calibration = JSON.parse(readFileSync(calPath, "utf-8"));
    } catch {
      this.calibration = null; // no calibration — use raw scores
    }
  }

  /**
   * Piecewise linear interpolation (equivalent to numpy.interp).
   * Used by both score and confidence calibration.
   */
  interpolate(value, xs, ys) {
    if (value <= xs[0]) return ys[0];
    if (value >= xs[xs.length - 1]) return ys[ys.length - 1];
    for (let i = 1; i < xs.length; i++) {
      if (value <= xs[i]) {
        const t = (value - xs[i - 1]) / (xs[i] - xs[i - 1]);
        return ys[i - 1] + t * (ys[i] - ys[i - 1]);
      }
    }
    return ys[ys.length - 1];
  }

  /**
   * Apply per-dimension isotonic calibration to de-compress score ranges.
   * Falls back to raw score if no calibration is available for a dimension.
   */
  calibrate(dimName, rawScore) {
    if (!this.calibration || !this.calibration[dimName]) return rawScore;
    const cal = this.calibration[dimName];

    if (cal.method === "isotonic" && cal.x_thresholds && cal.y_thresholds) {
      return this.interpolate(rawScore, cal.x_thresholds, cal.y_thresholds);
    }

    if (cal.method === "linear") {
      return Math.max(0, Math.min(10, rawScore * cal.scale + cal.shift));
    }

    return rawScore; // identity or unknown method
  }

  /**
   * Apply confidence calibration: map raw model confidence to actual accuracy.
   * Without calibration, raw confidence is inverted for 6/10 dimensions.
   */
  calibrateConfidence(dimName, rawConf) {
    if (!this.calibration || !this.calibration[dimName]) return rawConf;
    const confCal = this.calibration[dimName].confidence_calibration;
    if (!confCal) return rawConf;

    if (confCal.method === "isotonic" && confCal.x_thresholds && confCal.y_thresholds) {
      return this.interpolate(rawConf, confCal.x_thresholds, confCal.y_thresholds);
    }

    if (confCal.method === "linear") {
      return Math.max(0, Math.min(1, rawConf * confCal.scale + confCal.shift));
    }

    return rawConf;
  }

  /**
   * Tokenize text using @huggingface/transformers.
   * Handles WordPiece (DistilBERT), Unigram/SentencePiece (DeBERTa-v3), and BPE.
   */
  tokenize(text) {
    const encoded = this.tokenizer(text, {
      max_length: this.maxLength,
      padding: "max_length",
      truncation: true,
    });

    const ids = Array.from(encoded.input_ids.data);
    const attentionMask = Array.from(encoded.attention_mask.data);

    return { ids, attentionMask };
  }

  /**
   * Score text across all 10 PSQ dimensions in one forward pass.
   * Returns { scores: { dim_name: { score, confidence } }, elapsed_ms }
   */
  async score(text) {
    if (!this.session) await this.init();

    const start = performance.now();

    const { ids, attentionMask } = this.tokenize(text);

    const ort = await import("onnxruntime-node");
    const inputIds = new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, this.maxLength]);
    const maskTensor = new ort.Tensor("int64", BigInt64Array.from(attentionMask.map(BigInt)), [1, this.maxLength]);

    const results = await this.session.run({
      input_ids: inputIds,
      attention_mask: maskTensor,
    });

    const scores = results.scores.data;   // Float32Array, length 10
    const confs = results.confidences.data; // Float32Array, length 10

    const elapsed = performance.now() - start;

    const dimensionScores = {};
    for (let i = 0; i < DIMENSIONS.length; i++) {
      const raw = scores[i];
      const calibrated = this.calibrate(DIMENSIONS[i], raw);
      const rawConf = confs[i];
      const calibratedConf = this.calibrateConfidence(DIMENSIONS[i], rawConf);
      dimensionScores[DIMENSIONS[i]] = {
        score: Math.round(Math.max(0, Math.min(10, calibrated)) * 100) / 100,
        confidence: Math.round(Math.max(0, Math.min(1, calibratedConf)) * 1000) / 1000,
      };
    }

    return {
      provider: "student",
      scores: dimensionScores,
      hierarchy: StudentProvider.computeHierarchy(dimensionScores),
      elapsed_ms: Math.round(elapsed),
    };
  }

  /**
   * Compute hierarchical reporting: cluster subscales + g-PSQ general factor.
   * Additive layer — the 10 individual dimensions remain primary.
   */
  static computeHierarchy(dimensionScores) {
    const clusters = {};
    for (const [clusterName, memberDims] of Object.entries(CLUSTERS)) {
      let sumWeighted = 0, sumWeights = 0, sumConf = 0, n = 0;
      for (const dim of memberDims) {
        const d = dimensionScores[dim];
        if (!d || d.score === undefined) continue;
        const conf = d.confidence ?? 0.5;
        sumWeighted += d.score * conf;
        sumWeights += conf;
        sumConf += conf;
        n++;
      }
      const score = sumWeights > 0 ? sumWeighted / sumWeights : 5;
      const confidence = n > 0 ? sumConf / n : 0;
      clusters[clusterName] = {
        score: Math.round(score * 100) / 100,
        confidence: Math.round(confidence * 1000) / 1000,
        dimensions: memberDims,
      };
    }

    // g-PSQ: confidence-weighted mean of all 10 dimensions
    let totalWeighted = 0, totalWeights = 0, totalConf = 0, totalN = 0;
    for (const dim of DIMENSIONS) {
      const d = dimensionScores[dim];
      if (!d || d.score === undefined) continue;
      const conf = d.confidence ?? 0.5;
      totalWeighted += d.score * conf;
      totalWeights += conf;
      totalConf += conf;
      totalN++;
    }
    const gScore = totalWeights > 0 ? totalWeighted / totalWeights : 5;
    const gConf = totalN > 0 ? totalConf / totalN : 0;

    return {
      clusters,
      g_psq: {
        score: Math.round(gScore * 100) / 100,
        confidence: Math.round(gConf * 1000) / 1000,
      },
    };
  }

  /**
   * Compatibility wrapper matching LLM provider evaluate() interface.
   * Ignores systemPrompt/userPrompt and extracts text to score.
   */
  async evaluate(systemPrompt, userPrompt) {
    // Extract the content from user prompt (format: "CONTENT TO EVALUATE:\n\n<text>")
    const match = userPrompt.match(/CONTENT TO EVALUATE:\s*\n\s*\n([\s\S]+?)(?:\n\nCONTEXT:|$)/);
    const text = match ? match[1].trim() : userPrompt;

    // Extract dimension from system prompt
    const dimMatch = systemPrompt.match(/DIMENSION:\s*(.+)/);
    const dimName = dimMatch ? dimMatch[1].trim() : null;

    const result = await this.score(text);

    // Find the matching dimension
    const dimId = dimName
      ? DIMENSIONS.find(d => dimName.toLowerCase().includes(d.replace(/_/g, " ")))
      : null;

    if (dimId && result.scores[dimId]) {
      const dim = result.scores[dimId];
      return {
        raw: JSON.stringify(result.scores),
        parsed: {
          dimension: dimId,
          score: dim.score,
          confidence: dim.confidence,
          subscale_scores: {},
          rationale: `Student model prediction (${result.elapsed_ms}ms). No subscale or evidence analysis available.`,
          instruments_used: ["student-model"],
          disclaimer: "This evaluation assesses content, not individuals. It is not a clinical diagnosis.",
        },
      };
    }

    // Return all dimensions if no specific one requested
    return {
      raw: JSON.stringify(result.scores),
      parsed: result.scores,
    };
  }
}
