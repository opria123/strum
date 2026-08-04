"""Aggregate and print LaneAcc alongside F1/Precision/Recall from an existing
benchmark_results.json, using the same TP-weighted aggregation eval_benchmark.py
uses for the headline numbers.

lane_accuracy is already computed and stored per-song by eval_benchmark.py
(evaluate_song() -> onset_f1()), but it was not previously surfaced anywhere
in aggregate form. This script does not re-run inference or evaluation --
it only re-aggregates numbers already present in the given results file.

Usage:
    python scripts/report_lane_accuracy.py [benchmark_results.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_benchmark import aggregate  # noqa: E402


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmark_results.json")
    data = json.loads(results_path.read_text())
    per_song = data["per_song"]

    agg = aggregate(per_song)

    print(f"Songs: {len(per_song)}   tolerance_ms: {data.get('tolerance_ms')}")
    print()
    header = f"{'Instrument':<10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'LaneAcc':>9}"
    print(header)
    print("-" * len(header))
    for part, r in agg.items():
        print(
            f"{part:<10} {r['f1'] * 100:>7.1f}% {r['precision'] * 100:>9.1f}% "
            f"{r['recall'] * 100:>7.1f}% {r['lane_accuracy'] * 100:>8.1f}%"
        )


if __name__ == "__main__":
    main()
