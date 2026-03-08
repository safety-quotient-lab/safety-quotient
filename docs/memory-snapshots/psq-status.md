<!-- PROVENANCE: Restored 2026-03-06 by /cycle Step 11 orphan check
     Source: docs/memory-snapshots/psq-status.md -->

# PSQ Sub-Agent Status (managed in its own context)

**Production endpoint:** ✓ https://psq.unratified.org/score — live, TLS, Hetzner CX Ashburn
**Score calibration:** ✓ isotonic-v2-2026-03-06. Quantile-binned isotonic (n_bins=20).
  All 10 dims calibrated. HI dead zone resolved (B2 (HI calibration dead zone) fix, Session 26).
  Historical MAE improvement: +3.5–21.6% per dimension vs. raw.
**Confidence calibration:** ✓ B1 (confidence head dead) FIXED (Session 26). r_confidence field added to score output.
  calibration_note surfaces held-out Pearson r per dimension. scale=0 behavior (intentional
  constant function overriding anti-calibrated head) now explicit. Limitation:
  confidence-is-static-r (MEDIUM — not HIGH; behavior is intentional design).
**Model transfer:** ✓ rsync complete. SHA256 verified (Hetzner matches Chromebook source).
  41 files, 531 MB. best.pt on Hetzner; local copy lost.
**Service:** systemd psq-server active. 84ms inference. onnxruntime-node postinstall fix.
**Wrangler secret:** PSQ_ENDPOINT_URL → https://psq.unratified.org
**Firewall:** ufw SSH + HTTP/HTTPS only. Port 3000 closed from public.
**Integration:** psq-scoring session turn 7 complete — 5 ICESCR texts scored, B2 (HI calibration dead zone) validated.
**B3 (TE uniformity) CLOSED 2026-03-07:**
  v29 REJECTED (TE=0.734, overall=0.668). v30 single-task ceiling=0.762.
  F3 (500 texts): drove v31 (TE=0.773, overall=0.679) — REJECTED.
  F3b (700 texts): drove v32 (TE=0.739, overall=0.676) — REJECTED.
  F4 (350 texts: 200 prosocial + 150 esconv, distribution-rebalanced): drove v33 (TE=0.742, overall=0.672) — REJECTED.
  5 consecutive rejections. 1,550 total expansion texts. SE(r)≈0.10 noise floor at n=99 binding.
  B3 CLOSED — v23 TE=0.795 accepted as production ceiling. F1 (recalibrate n_bins=20) deferred permanently.
**Known open issues:**
  - DA validity (authority_dynamics construct)
  - DA trend: 0.588 (v23) → 0.531 (v29) → 0.501 (v31) → 0.558 (v32) → 0.544 (v33); noisy, not a trend
  - AD: 0.713 (v23) → 0.671 (v31) → 0.732 (v32) → 0.678 (v33); stochastic variance (SE(r)≈0.10)
  - AD range compression CONFIRMED HIGH (2026-03-07, journal §40): effective output range 3.84–6.38
    for Dreaddit texts; 5.13–6.38 for formal authority texts. Max abuse anchor (expected 0) → 5.13.
    Direction reversal: coercive authority (expected 2.5) → 5.67 > neutral (5.13). UCC/extreme-adco
    batches did NOT correct compression (Q7 answered: No). Held-out r=0.713 valid as ordinal only;
    absolute scores uninterpretable for formal authority texts. Context-weighted composite (workplace,
    AD weight=2.0) affected — AD contribution near-constant 5.1–5.7 for formal texts. See journal §40.
  - CO weakness (r=0.534–0.538 across v23/v32/v33)
  - No human validation (only Dreaddit training data)
  - WEIRD assumptions
  - B3 CLOSED — TE ceiling accepted at v23 0.795
  - HI range compression: effective output range ~3.44–7.98 (4.5 of 10 points). Floor compressed:
    explicit slur+aggression (expected 0-1) scores 3.44; death threat scores 4.81. Ceiling
    compressed: conflict resolution (expected 9-10) scores 7.26. Mid-range (4-7) calibrated with
    slight upward bias. Analogous to AD compression. Cause: Dreaddit training data lacks extreme
    hostility and extreme warmth examples. HI direction anomaly (smoke test) resolved as construct
    nuance — stress venting (no target) correctly scores higher HI than policy critique (other-
    attribution of hostile intent). See journal §39.
**PSQ-Lite:** TE + HI(raw) + TC adopted by unratified-agent for advocacy content (provisional).
Do not duplicate PSQ improvement work in this context.
