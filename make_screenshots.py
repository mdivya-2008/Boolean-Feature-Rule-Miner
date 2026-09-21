"""
make_screenshots.py
===================
Generates the 4 project screenshots (PNG) into  screenshots/ :

    01_top_rules_bar.png            — Top-10 rules: Support vs Confidence bars
    02_coverage_vs_confidence.png   — rule space scatter (colour = F1)
    03_truth_table_heatmap.png      — truth-table heatmap of the best rule
    04_simplification_before_after.png — (A&B)|(A&~B) => A  Q-M demo panel

Usage:
    python make_screenshots.py                # uses data/synthetic.csv
    python make_screenshots.py --input data/synthetic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boolean_logic as bl
import rule_miner as rm
import main as m

ROOT = Path(__file__).resolve().parent
SCREENSHOT_DIR = ROOT / "screenshots"


def main() -> int:
    p = argparse.ArgumentParser(description="Generate the 4 project screenshots.")
    p.add_argument("--input", type=str, default=str(ROOT / "data" / "synthetic.csv"))
    args = p.parse_args()

    df = rm.load_and_validate_csv(args.input)
    res = rm.mine_rules(df, "Y", max_literals=3, min_support=0.05,
                        min_confidence=0.6, top_n=10, verbose=True)
    if res["top"].empty:
        print("No rules found — cannot make screenshots.")
        return 1

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    top = res["top"]

    p1 = SCREENSHOT_DIR / "01_top_rules_bar.png"
    p2 = SCREENSHOT_DIR / "02_coverage_vs_confidence.png"
    p3 = SCREENSHOT_DIR / "03_truth_table_heatmap.png"
    p4 = SCREENSHOT_DIR / "04_simplification_before_after.png"

    m.plot_top_rules_bar(top, p1)
    m.plot_coverage_scatter(top, p2)
    var_order = bl.expression_variables(res["best"]["Rule"])
    m.plot_truth_table_heatmap(res["best"]["Rule"], p3, var_order)

    # classic simplification demo (always reduces: 4 literals -> 1 literal)
    demo = bl.simplify_expression("(A AND B) OR (A AND NOT B)")
    m.plot_simplification_panel(
        demo["original"], demo["simplified_algebraic"],
        demo["simplified_plain"], demo["steps"], p4)

    print(f"\nSaved {len([p1, p2, p3, p4])} screenshots to {SCREENSHOT_DIR}/ :")
    for f in (p1, p2, p3, p4):
        print(f"  - {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
