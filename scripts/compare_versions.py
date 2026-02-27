"""
Compare PSQ student model versions side-by-side.

Auto-detects versioned result files in models/psq-student/ or accepts explicit
version names. Prints a formatted comparison table.

Usage:
  python scripts/compare_versions.py                          # auto-detect all versions
  python scripts/compare_versions.py v2d v3b                  # specific versions
  python scripts/compare_versions.py --metric mse             # compare MSE instead of r
  python scripts/compare_versions.py --metric all             # r + MSE + n
  python scripts/compare_versions.py --use-val                # use validation (best_results) instead of test
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "models" / "psq-student"

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


def find_versions():
    """Auto-detect available version result files."""
    versions = {}

    for f in sorted(SAVE_DIR.glob("*_test_results.json")):
        name = f.stem.replace("_test_results", "")
        versions.setdefault(name, {})["test"] = f

    for f in sorted(SAVE_DIR.glob("*_best_results.json")):
        name = f.stem.replace("_best_results", "")
        versions.setdefault(name, {})["best"] = f

    return versions


def load_results(path, use_val=False):
    """Load results from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if "val_results" in data and use_val:
        return data["val_results"]
    return data


def get_val(results, dim, metric):
    """Extract a metric value for a dimension."""
    entry = results.get(dim, {})
    if isinstance(entry, dict):
        return entry.get(metric)
    return None


def fmt(val, metric):
    """Format a value for display."""
    if val is None:
        return "   —  "
    if metric == "r":
        return f"{val:+.3f}"
    elif metric == "mse":
        return f"{val:.2f}"
    elif metric == "n":
        return f"{int(val):5d}"
    return f"{val:.3f}"


def print_table(version_data, metric="r"):
    """Print formatted comparison table."""
    names = list(version_data.keys())
    if not names:
        print("No versions found.")
        return

    dim_w = 14
    col_w = max(8, max(len(n) for n in names) + 2)

    # Header
    header = f"  {'Dimension':<{dim_w}}"
    for name in names:
        header += f" {name:>{col_w}}"
    if len(names) >= 2:
        header += f" {'Δ(last)':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Rows
    avgs = {name: [] for name in names}
    for dim in DIMENSIONS:
        short = DIM_SHORT.get(dim, dim[:12])
        row = f"  {short:<{dim_w}}"
        vals = []
        for name in names:
            v = get_val(version_data[name], dim, metric)
            vals.append(v)
            if v is not None:
                avgs[name].append(v)
            row += f" {fmt(v, metric):>{col_w}}"

        # Delta: last vs first
        if len(names) >= 2 and vals[0] is not None and vals[-1] is not None:
            d = vals[-1] - vals[0]
            sign = "+" if d >= 0 else ""
            if metric == "r":
                row += f" {sign}{d:.3f}".rjust(col_w + 1)
            elif metric == "mse":
                row += f" {sign}{d:.2f}".rjust(col_w + 1)
            else:
                row += f" {sign}{int(d)}".rjust(col_w + 1)
        elif len(names) >= 2:
            row += " " * (col_w + 1)

        # Bold the best value
        valid = [(v, i) for i, v in enumerate(vals) if v is not None]
        print(row)

    # Separator + averages
    print("  " + "-" * (len(header) - 2))
    row = f"  {'AVERAGE':<{dim_w}}"
    avg_vals = []
    for name in names:
        # Prefer stored _avg_r
        stored = version_data[name].get("_avg_r") if metric == "r" else None
        if stored is not None:
            avg_vals.append(stored)
        else:
            vals = avgs[name]
            stored = sum(vals) / len(vals) if vals else None
            avg_vals.append(stored)
        row += f" {fmt(stored, metric):>{col_w}}"

    if len(names) >= 2 and avg_vals[0] is not None and avg_vals[-1] is not None:
        d = avg_vals[-1] - avg_vals[0]
        sign = "+" if d >= 0 else ""
        if metric in ("r", "mse"):
            row += f" {sign}{d:.3f}".rjust(col_w + 1)

    print(row)

    # Bar chart for r
    if metric == "r" and names:
        last = names[-1]
        print(f"\n  Pearson r by Dimension ({last}):")
        sorted_dims = sorted(DIMENSIONS,
                             key=lambda d: -(get_val(version_data[last], d, "r") or -1))
        for dim in sorted_dims:
            short = DIM_SHORT.get(dim, dim[:12])
            r = get_val(version_data[last], dim, "r")
            if r is not None:
                bar_len = max(0, int(r * 40))
                bar = "#" * bar_len
                print(f"    {short:<14s} |{bar:<40s}| {r:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compare PSQ model versions")
    parser.add_argument("versions", nargs="*", help="Version names (e.g., v2d v3b)")
    parser.add_argument("--metric", default="r", choices=["r", "mse", "n", "all"])
    parser.add_argument("--use-val", action="store_true",
                        help="Use validation results from best_results.json")
    args = parser.parse_args()

    available = find_versions()

    if args.versions:
        requested = args.versions
    else:
        requested = sorted(available.keys())

    # Load data
    version_data = {}
    for name in requested:
        if name not in available:
            print(f"  Warning: no results for '{name}'", file=sys.stderr)
            continue
        info = available[name]
        if not args.use_val and "test" in info:
            version_data[name] = load_results(info["test"])
        elif "best" in info:
            version_data[name] = load_results(info["best"], use_val=True)
        else:
            print(f"  Warning: no loadable results for '{name}'", file=sys.stderr)

    if not version_data:
        print("No version data found. Available:", list(available.keys()))
        return

    metrics = ["r", "mse", "n"] if args.metric == "all" else [args.metric]

    print(f"\n{'='*60}")
    print(f"  PSQ Student Model Comparison")
    print(f"  Versions: {', '.join(version_data.keys())}")
    print(f"  Source: {'validation' if args.use_val else 'test'} results")
    print(f"{'='*60}")

    for m in metrics:
        if len(metrics) > 1:
            print(f"\n--- {m.upper()} ---")
        else:
            print()
        print_table(version_data, metric=m)

    print()


if __name__ == "__main__":
    main()
