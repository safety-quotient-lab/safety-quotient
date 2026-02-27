"""
Estimate LLM label requirements per dimension to reach psychometric quality.

Uses empirical data from training history + signal-to-noise analysis to project
how many LLM labels each dimension needs to reach target Pearson r >= 0.70.

Models two independent factors:
  1. Signal quality: effective LLM signal / total effective signal (S/N ratio)
  2. Text-observability ceiling: inherent limit on how well a construct can be
     measured from text alone (estimated from best observed r across versions)

Usage:
  python scripts/estimate_labels.py
  python scripts/estimate_labels.py --target 0.75
  python scripts/estimate_labels.py --remove-bad-proxy
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIMENSIONS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]

DIM_SHORT = {
    "threat_exposure": "threat",
    "hostility_index": "hostility",
    "authority_dynamics": "authority",
    "energy_dissipation": "energy",
    "regulatory_capacity": "regulatory",
    "resilience_baseline": "resilience",
    "trust_conditions": "trust",
    "cooling_capacity": "cooling",
    "defensive_architecture": "defensive",
    "contractual_clarity": "contractual",
}

# Proxy quality assessment — does the proxy data HELP or HURT?
# Based on empirical analysis across v2d/v3/v3b versions
PROXY_QUALITY = {
    "threat_exposure": "good",       # Berkeley IRT + Civil Comments: strong signal
    "hostility_index": "good",       # Berkeley IRT + Civil Comments + UCC: strong
    "authority_dynamics": "harmful",  # Politeness: compressed range, noise. UCC: near-random
    "energy_dissipation": "good",    # Dreaddit stress: clean proxy
    "regulatory_capacity": "fair",   # ESConv good, GoEmotions/EmpDialogues indirect
    "resilience_baseline": "fair",   # EmpDialogues indirect, GoEmotions indirect
    "trust_conditions": "fair",      # UCC mediocre, GoEmotions indirect
    "cooling_capacity": "fair",      # UCC mediocre, GoEmotions indirect
    "defensive_architecture": "weak", # Prosocial: safety labels ≠ defense mechanisms
    "contractual_clarity": "fair",   # Casino: small but relevant
}

# Training parameters
CONF_POWER = 2.0
CONF_THRESHOLD = 0.15
LLM_WEIGHT = 5.0
COMPOSITE_WEIGHT = 1.5


def load_source_stats():
    """Load per-dimension, per-source statistics from training data."""
    stats = {d: {} for d in DIMENSIONS}

    for fname in ["data/composite-ground-truth.jsonl", "data/train-llm.jsonl"]:
        path = ROOT / fname
        if not path.exists():
            continue
        tag = "llm_gold" if "train-llm" in fname else "composite"

        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                rdims = rec.get("dimensions", {})
                source = rec.get("teacher", rec.get("source", "proxy"))
                src_name = rec.get("source", "?")
                is_llm = source in ("llm", "llm_labeled")

                for d in DIMENSIONS:
                    if d not in rdims:
                        continue
                    conf = rdims[d].get("confidence", 0.5)

                    key = f"LLM({tag})" if is_llm else src_name
                    weight = LLM_WEIGHT if is_llm else COMPOSITE_WEIGHT

                    if key not in stats[d]:
                        stats[d][key] = {
                            "count": 0, "active": 0, "conf_sum": 0.0,
                            "weight": weight, "is_llm": is_llm,
                        }
                    stats[d][key]["count"] += 1
                    if conf > CONF_THRESHOLD:
                        stats[d][key]["active"] += 1
                        stats[d][key]["conf_sum"] += conf

    return stats


def compute_effective_signal(sources):
    """Compute effective signal per source group."""
    results = []
    for key, info in sources.items():
        if info["active"] == 0:
            continue
        avg_conf = info["conf_sum"] / info["active"]
        conf_sq = avg_conf ** CONF_POWER
        eff = info["active"] * info["weight"] * conf_sq
        results.append({
            "source": key,
            "count": info["active"],
            "weight": info["weight"],
            "avg_conf": avg_conf,
            "eff_signal": eff,
            "is_llm": info["is_llm"],
        })
    return results


def load_best_r():
    """Load best observed test r for each dimension across all versions."""
    best = {d: 0.0 for d in DIMENSIONS}
    model_dir = ROOT / "models" / "psq-student"

    for f in model_dir.glob("*_test_results.json"):
        with open(f) as fh:
            data = json.load(fh)
        for d in DIMENSIONS:
            r = data.get(d, {}).get("r", 0)
            if r is not None and r > best[d]:
                best[d] = r

    return best


def estimate_ceiling(best_r):
    """Estimate text-observability ceiling from best observed r.

    The ceiling is above the best observed r (we haven't reached it yet).
    Use a simple heuristic: ceiling = best_r + headroom, where headroom
    depends on how much the dimension has improved across versions.
    Conservative: assume 15-25% headroom above best observed.
    """
    ceilings = {}
    for d in DIMENSIONS:
        r = best_r[d]
        pq = PROXY_QUALITY[d]

        if pq == "good":
            # Good proxy data means we're closer to ceiling
            headroom = 0.10
        elif pq == "fair":
            # Fair proxy means moderate headroom
            headroom = 0.15
        elif pq == "weak":
            # Weak proxy means we might be far from ceiling
            headroom = 0.20
        else:  # harmful
            # Harmful proxy means best_r is suppressed; real ceiling is much higher
            headroom = 0.25

        ceilings[d] = min(r + headroom, 0.95)  # cap at 0.95

    return ceilings


def fit_k(best_r, ceiling, useful_frac, total_eff):
    """Fit the learning rate constant k from observed data.

    Solves: best_r = ceiling * (1 - exp(-k * useful_frac * sqrt(total_eff)))
    for k.
    """
    if ceiling <= 0 or best_r <= 0 or best_r >= ceiling:
        return 0.04  # fallback
    x = useful_frac * math.sqrt(max(total_eff, 1))
    if x <= 0:
        return 0.04
    ratio = best_r / ceiling
    # ratio = 1 - exp(-k * x)  →  exp(-k*x) = 1 - ratio  →  k = -ln(1-ratio) / x
    inner = 1.0 - ratio
    if inner <= 0:
        return 0.10  # nearly at ceiling
    k = -math.log(inner) / x
    return max(k, 0.001)  # floor


def project_r(useful_signal_fraction, ceiling, total_effective_samples, k=0.04):
    """Project expected r given useful signal fraction, ceiling, and fitted k.

    Model: r = ceiling * (1 - exp(-k * useful_frac * sqrt(total_eff)))

    "Useful signal" = LLM signal + quality-weighted proxy signal.
    k is fitted per-dimension from empirical best_r.
    """
    x = useful_signal_fraction * math.sqrt(max(total_effective_samples, 1))
    return ceiling * (1 - math.exp(-k * x))


# How much useful signal does each proxy quality tier contribute?
# 1.0 = as good as LLM, 0.0 = pure noise
PROXY_SIGNAL_WEIGHT = {
    "good": 0.70,     # Berkeley hostility, Civil Comments threat, Dreaddit stress
    "fair": 0.40,     # ESConv, Casino, GoEmotions — partially relevant
    "weak": 0.10,     # Prosocial — loosely related
    "harmful": 0.00,  # Politeness authority — actively misleading
}


def compute_useful_fraction(dim, sources, extra_gold_eff=0.0, remove_bad_proxy=False):
    """Compute fraction of effective signal that is 'useful' (LLM + quality-weighted proxy)."""
    pq = PROXY_QUALITY[dim]
    proxy_weight = PROXY_SIGNAL_WEIGHT[pq]

    useful = extra_gold_eff  # LLM gold always useful
    total = extra_gold_eff

    for src in sources:
        eff = src["eff_signal"]
        if remove_bad_proxy and not src["is_llm"] and pq in ("harmful", "weak"):
            continue
        total += eff
        if src["is_llm"]:
            useful += eff
        else:
            useful += eff * proxy_weight

    if total == 0:
        return 0.0, 0.0
    return useful / total, total


def find_needed_llm_gold(dim, sources, ceiling, target_r, current_llm_gold_count,
                         llm_gold_avg_conf, k, remove_bad_proxy=False):
    """Binary search for number of LLM-gold labels needed to reach target r."""

    if ceiling < target_r:
        return None, "ceiling"  # Can't reach target

    # Build base sources (everything except LLM-gold)
    base_sources = [s for s in sources if s["source"] != "LLM(llm_gold)"]
    llm_gold_conf_sq = llm_gold_avg_conf ** CONF_POWER

    # Search from 0 to 5000
    for n in range(0, 5001, 10):
        gold_eff = n * LLM_WEIGHT * llm_gold_conf_sq
        useful_frac, total = compute_useful_fraction(
            dim, base_sources, extra_gold_eff=gold_eff,
            remove_bad_proxy=remove_bad_proxy)
        r = project_r(useful_frac, ceiling, total, k=k)
        if r >= target_r:
            return n, "ok"

    return 5000, "insufficient"


def main():
    parser = argparse.ArgumentParser(description="Estimate LLM label requirements")
    parser.add_argument("--target", type=float, default=0.70,
                        help="Target Pearson r (default: 0.70)")
    parser.add_argument("--remove-bad-proxy", action="store_true",
                        help="Model removing harmful/weak proxy data")
    args = parser.parse_args()

    print("Loading training data...")
    stats = load_source_stats()
    best_r = load_best_r()
    ceilings = estimate_ceiling(best_r)

    print(f"\n{'='*95}")
    print(f"  PSQ Label Requirement Estimator — Target r >= {args.target:.2f}")
    if args.remove_bad_proxy:
        print(f"  Mode: WITH removal of harmful/weak proxy data")
    print(f"{'='*95}")

    # Per-dimension analysis
    print(f"\n  {'Dim':<14s} {'Proxy':>7s} {'BestR':>6s} {'Ceil':>6s} "
          f"{'LLM#':>6s} {'Proxy#':>7s} {'S/N':>6s} {'Useful%':>7s} "
          f"{'k':>6s} {'ProjR':>6s} {'Need':>6s} {'Strategy'}")
    print(f"  {'-'*14} {'-'*7} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*7} {'-'*6} {'-'*7} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*30}")

    recommendations = []

    for dim in DIMENSIONS:
        short = DIM_SHORT[dim]
        pq = PROXY_QUALITY[dim]
        br = best_r[dim]
        ceil = ceilings[dim]

        sources = compute_effective_signal(stats[dim])

        # Current state
        llm_signal = sum(s["eff_signal"] for s in sources if s["is_llm"])
        proxy_signal = sum(s["eff_signal"] for s in sources if not s["is_llm"])
        total_signal = llm_signal + proxy_signal
        llm_count = sum(s["count"] for s in sources if s["is_llm"])
        proxy_count = sum(s["count"] for s in sources if not s["is_llm"])

        if total_signal > 0:
            sn_ratio = llm_signal / max(proxy_signal, 0.01)
            llm_frac = llm_signal / total_signal
        else:
            sn_ratio = 0
            llm_frac = 0

        # Fit k from observed best_r, then project
        useful_frac, _ = compute_useful_fraction(dim, sources)
        dim_k = fit_k(br, ceil, useful_frac, total_signal)
        proj_r = project_r(useful_frac, ceil, total_signal, k=dim_k)

        # Find LLM-gold avg conf
        gold_sources = [s for s in sources if s["source"] == "LLM(llm_gold)"]
        if gold_sources:
            gold_avg_conf = gold_sources[0]["avg_conf"]
            current_gold = gold_sources[0]["count"]
        else:
            gold_avg_conf = 0.50
            current_gold = 0

        # Estimate needed
        needed, status = find_needed_llm_gold(
            dim, sources, ceil, args.target, current_gold,
            gold_avg_conf, k=dim_k, remove_bad_proxy=args.remove_bad_proxy
        )

        # Strategy recommendation
        if br >= args.target:
            strategy = "Already achieved"
            needed_str = "  —  "
        elif status == "ceiling":
            strategy = f"Ceiling {ceil:.2f} < target; redefine construct"
            needed_str = " ceil"
        elif status == "insufficient":
            strategy = "Needs construct redesign or better proxy"
            needed_str = ">5000"
        elif needed <= current_gold:
            strategy = f"Sufficient (have {current_gold}); improve training"
            needed_str = f"  ={current_gold:3d}"
        else:
            delta = needed - current_gold
            if pq in ("harmful", "weak") and not args.remove_bad_proxy:
                strategy = f"+{delta} LLM OR remove {pq} proxy (try --remove-bad-proxy)"
            else:
                strategy = f"+{delta} LLM-gold labels"
            needed_str = f"{needed:5d}"

        print(f"  {short:<14s} {pq:>7s} {br:>6.3f} {ceil:>6.2f} "
              f"{llm_count:>6d} {proxy_count:>7d} {sn_ratio:>6.1f} {useful_frac:>6.0%} "
              f"{dim_k:>6.4f} {proj_r:>6.3f} {needed_str:>6s} {strategy}")

        recommendations.append({
            "dimension": dim,
            "proxy_quality": pq,
            "best_r": round(br, 4),
            "ceiling": round(ceil, 2),
            "current_llm": llm_count,
            "current_proxy": proxy_count,
            "sn_ratio": round(sn_ratio, 2),
            "llm_fraction": round(llm_frac, 3),
            "useful_fraction": round(useful_frac, 3),
            "projected_r": round(proj_r, 4),
            "needed_llm_gold": needed,
            "current_llm_gold": current_gold,
            "delta_needed": max(0, needed - current_gold) if status == "ok" else None,
            "status": status,
        })

    # Summary
    print(f"\n{'='*95}")
    print(f"  SUMMARY")
    print(f"{'='*95}")

    already = [r for r in recommendations if r["best_r"] >= args.target]
    achievable = [r for r in recommendations
                  if r["best_r"] < args.target and r["status"] == "ok"]
    ceiling_blocked = [r for r in recommendations if r["status"] == "ceiling"]
    hard = [r for r in recommendations if r["status"] == "insufficient"]

    if already:
        print(f"\n  Already at r >= {args.target:.2f} ({len(already)} dims):")
        for r in already:
            print(f"    {DIM_SHORT[r['dimension']]}: best r = {r['best_r']:.3f}")

    if achievable:
        total_new = sum(r["delta_needed"] or 0 for r in achievable)
        print(f"\n  Achievable with more LLM labels ({len(achievable)} dims, +{total_new} labels total):")
        for r in sorted(achievable, key=lambda x: x["delta_needed"] or 0):
            delta = r["delta_needed"] or 0
            print(f"    {DIM_SHORT[r['dimension']]:<14s} need {r['needed_llm_gold']:>4d} LLM-gold "
                  f"(have {r['current_llm_gold']:>3d}, +{delta:>3d} new)")

    if ceiling_blocked:
        print(f"\n  Blocked by ceiling ({len(ceiling_blocked)} dims):")
        for r in ceiling_blocked:
            print(f"    {DIM_SHORT[r['dimension']]}: ceiling = {r['ceiling']:.2f} < target {args.target:.2f}")

    if hard:
        print(f"\n  Needs fundamental rethink ({len(hard)} dims):")
        for r in hard:
            print(f"    {DIM_SHORT[r['dimension']]}: current best = {r['best_r']:.3f}, "
                  f"proxy quality = {r['proxy_quality']}")

    # Cost estimate
    if achievable:
        total_new = sum(r["delta_needed"] or 0 for r in achievable)
        # ~10 API calls per label × $0.003 per call (Claude Haiku)
        cost_low = total_new * 10 * 0.003
        # ~10 API calls per label × $0.015 per call (Claude Sonnet)
        cost_high = total_new * 10 * 0.015
        print(f"\n  Estimated cost for +{total_new} labels:")
        print(f"    Haiku:  ~${cost_low:.0f}")
        print(f"    Sonnet: ~${cost_high:.0f}")

    # Save
    out_path = ROOT / "models" / "psq-student" / "label_estimates.json"
    with open(out_path, "w") as f:
        json.dump({
            "target_r": args.target,
            "remove_bad_proxy": args.remove_bad_proxy,
            "dimensions": recommendations,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
