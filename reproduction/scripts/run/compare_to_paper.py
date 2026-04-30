"""compare_to_paper.py — read a measured-results JSON and emit pass/fail vs the paper.

Usage:
    python compare_to_paper.py <measured.json> [--tolerance 0.05] [--output report.json]

The JSON format is:
    {
      "section": "4.shard_parity",
      "rows": [
        {"label": "A", "paper": 22.96, "measured": 24.54},
        {"label": "B_v5_beam", "paper": 22.08, "measured": 24.45},
        ...
      ]
    }

Exit code 0 if every row is within tolerance; non-zero with a per-row
diagnostic if any are over.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="measured-results JSON")
    ap.add_argument("--tolerance", type=float, default=0.05, help="±frac (default 0.05 = 5%)")
    ap.add_argument("--output", default=None, help="write structured report to this path")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    rows_out = []
    n_pass = n_fail = 0
    for row in data.get("rows", []):
        label = row["label"]
        paper = row["paper"]
        measured = row["measured"]
        if paper == 0:
            ratio = float("inf") if measured else 1.0
        else:
            ratio = measured / paper
        delta = ratio - 1.0
        within = abs(delta) <= args.tolerance
        if within:
            n_pass += 1
            verdict = "PASS"
        else:
            n_fail += 1
            verdict = "FAIL"
        print(f"  [{verdict}] {label:30s}  paper={paper:8.3f}  measured={measured:8.3f}  delta={delta:+.1%}", flush=True)
        rows_out.append({
            "label": label, "paper": paper, "measured": measured,
            "delta_pct": round(100 * delta, 2), "within_tolerance": within,
        })

    print()
    print(f"summary: {n_pass} pass, {n_fail} fail (tolerance {args.tolerance:+.0%})")

    out = {
        "section": data.get("section", "?"),
        "tolerance": args.tolerance,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "rows": rows_out,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"report written to {args.output}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
