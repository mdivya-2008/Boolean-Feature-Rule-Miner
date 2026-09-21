"""
main.py
=======
Command-line entry point of the Boolean Feature Rule Miner.

Usage (default matches the project spec):

    python main.py --input data/synthetic.csv --target Y \
        --max_literals 3 --min_support 0.05 --min_confidence 0.6 --top_n 10

What it does:
    1. load + validate the binary CSV
    2. generate candidate Boolean rules (brute force + min_support pruning)
    3. evaluate every rule with AND/OR/NOT gates on 0/1
    4. compute Support / Confidence / Coverage / Precision / Recall / F1
    5. rank by F1 -> Confidence -> Support -> simplicity, show Top-N
    6. deep-dive the BEST rule: truth table + SOP/POS + Q-M simplification
       + a gate-level evaluation trace on real data rows
    7. save 3 matplotlib charts into screenshots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: save PNGs, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import boolean_logic as bl
import rule_miner as rm

ROOT = Path(__file__).resolve().parent
SCREENSHOT_DIR = ROOT / "screenshots"


# ---------------------------------------------------------------------------
# CHARTS  (also imported by make_screenshots.py and app.py)
# ---------------------------------------------------------------------------

def plot_top_rules_bar(rules_df: pd.DataFrame, path=None):
    """Chart 1: grouped bar chart — Support vs Confidence for Top-N rules."""
    x = np.arange(len(rules_df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, rules_df["Support"], w, label="Support  P(R & Y=1)", color="#4C72B0")
    ax.bar(x + w / 2, rules_df["Confidence"], w, label="Confidence  P(Y=1 | R)", color="#DD8452")
    for xi, (s, c) in enumerate(zip(rules_df["Support"], rules_df["Confidence"])):
        ax.text(xi - w / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, c + 0.01, f"{c:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{i + 1}" for i in range(len(rules_df))], rotation=0)
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.15)
    ax.set_title("Top-10 Boolean rules — Support vs Confidence\n"
                 "(rules listed in final rank order)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_coverage_scatter(rules_df: pd.DataFrame, path=None):
    """Chart 2: Coverage vs Accuracy(Confidence) scatter, colour = F1."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sc = ax.scatter(rules_df["Coverage"], rules_df["Confidence"],
                    s=60 + rules_df["F1"] * 140,
                    c=rules_df["F1"], cmap="viridis", edgecolors="k", linewidths=0.5)
    for i, row in rules_df.iterrows():
        ax.annotate(f"R{i + 1}", (row["Coverage"], row["Confidence"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    cb = fig.colorbar(sc)
    cb.set_label("F1 score")
    ax.set_xlabel("Coverage  P(R = 1)")
    ax.set_ylabel("Confidence / Accuracy  P(Y = 1 | R)")
    ax.set_title("Rule space: Coverage vs Confidence (bubble size & colour = F1)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_truth_table_heatmap(expression: str, path=None, var_order: list | None = None):
    """Chart 3: truth-table heatmap of a rule (input columns + output column Y)."""
    tt = bl.generate_truth_table(expression, var_order)
    cols = list(tt.columns)
    data = tt.values
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(cols)), max(4, 0.32 * len(tt) + 1.5)))
    ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks(range(len(tt)))
    ax.set_yticklabels([f"row {i}" for i in range(len(tt))], fontsize=7)
    for i in range(len(tt)):
        for j in range(len(cols)):
            ax.text(j, i, str(data[i, j]), ha="center", va="center",
                    color="white" if data[i, j] == 1 else "black", fontsize=9)
    ax.set_title(f"Truth-table heatmap —  F = {expression}")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_simplification_panel(before: str, after_alg: str, after_plain: str,
                              steps: list[str], path=None):
    """Chart 4: before/after panel for the Boolean simplification demo."""
    fig = plt.figure(figsize=(11, 4.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.30, 0.62, f"BEFORE\n\nF = {before}", family="monospace", fontsize=13,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", fc="#FFF3CD", ec="#B8860B"))
    ax.annotate("", xy=(0.52, 0.62), xytext=(0.44, 0.62),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
    ax.text(0.48, 0.72, "Boolean identities\n+ Quine-McCluskey", ha="center",
            fontsize=8.5, style="italic")
    ax.text(0.72, 0.62, f"AFTER (minimal SOP)\n\nF = {after_alg}\n{after_plain}",
            family="monospace", fontsize=13, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", fc="#D4EDDA", ec="#3C763D"))
    step_text = "\n".join(steps[:3])
    ax.text(0.5, 0.16, "Q-M steps:  " + step_text.replace("\n", "    "),
            family="monospace", fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F2F2F2", ec="#999"))
    ax.set_title("Boolean simplification demo:  (A AND B) OR (A AND NOT B)  =>  A",
                 fontsize=12)
    if path:
        fig.savefig(path, dpi=140)
    return fig


# ---------------------------------------------------------------------------
# DEEP DIVE  (truth table + SOP/POS + simplification + gate trace)
# ---------------------------------------------------------------------------

def deep_dive(best: dict, df: pd.DataFrame, features: list[str]) -> None:
    expr = best["Rule"]
    print("\n" + "-" * 70)
    print(" DEEP DIVE — best rule (truth table + SOP/POS + simplification)")
    print("-" * 70)
    print(f"  IF  {expr}  THEN  Y = 1")
    print(f"  Support={best['Support']:.4f}  Confidence={best['Confidence']:.4f}  "
          f"Coverage={best['Coverage']:.4f}  F1={best['F1']:.4f}")

    # --- truth table (k <= 4 variables) ---
    var_order = bl.expression_variables(expr) if expr not in ("TRUE", "FALSE") else []
    if not var_order:                                   # constant rule:
        var_order = features[:4]                        # show table over dataset features
    tt = bl.generate_truth_table(expr, var_order)
    print(f"\n  Truth table (k = {len(tt.columns) - 1} variables, {len(tt)} rows):")
    print(tt.to_string(index=False))

    # --- SOP / POS canonical forms ---
    sop = bl.to_sop_canonical(expr, var_order or None)
    pos = bl.to_pos_canonical(expr, var_order or None)
    print(f"\n  SOP (Sum-of-Products) : {sop['sum_notation']}")
    if sop["algebraic"] not in ("0", "1"):
        print(f"      algebraic  : F = {sop['algebraic'][:100]}{'...' if len(sop['algebraic']) > 100 else ''}")
    print(f"  POS (Product-of-Sums) : {pos['product_notation']}")
    if pos["algebraic"] not in ("0", "1"):
        print(f"      algebraic  : F = {pos['algebraic'][:100]}{'...' if len(pos['algebraic']) > 100 else ''}")

    # --- Boolean simplification (identities + Quine-McCluskey) ---
    s = bl.simplify_expression(expr, var_order or None)
    print(f"\n  Simplification (Boolean identities + Quine-McCluskey):")
    print(f"      before : {s['original']}   ({s['literal_before']} literals)")
    for st in s["steps"]:
        print(f"      {st}")
    verdict = "already minimal" if s["already_minimal"] else "REDUCED"
    print(f"      after  : {s['simplified_algebraic']}   "
          f"({s['literal_after']} literals)  ->  {verdict}")
    print(f"      plain  : {s['simplified_plain']}")

    # --- gate-level evaluation trace on real data rows ---
    print("\n  Gate-level evaluation trace (real data rows):")
    series = best["series"]
    ones = series[series == 1].index
    zeros = series[series == 0].index
    sample_rows = list(ones[:1]) + list(zeros[:1])
    for idx in sample_rows:
        row = {f: int(df.loc[idx, f]) for f in df.columns if f != "Y"}
        val, trace = bl.evaluate_row(expr, row)
        print(f"\n    Row {idx}:  {row}   ->   F = {val}")
        print(bl.format_trace(trace))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Boolean Feature Rule Miner — mine IF-THEN Boolean rules "
                    "from a binary dataset (EC2201 Unit I).")
    p.add_argument("--input", type=str, default=str(ROOT / "data" / "synthetic.csv"),
                   help="path to the binary CSV (default data/synthetic.csv)")
    p.add_argument("--target", type=str, default="Y", help="target column name")
    p.add_argument("--max_literals", type=int, default=3,
                   help="max literals per AND-term, 1..4 (default 3)")
    p.add_argument("--min_support", type=float, default=0.05,
                   help="minimum support P(R & Y=1), default 0.05")
    p.add_argument("--min_confidence", type=float, default=0.6,
                   help="minimum confidence P(Y=1|R), default 0.6")
    p.add_argument("--top_n", type=int, default=10, help="number of rules to display")
    p.add_argument("--no_charts", action="store_true",
                   help="skip saving charts to screenshots/")
    args = p.parse_args(argv)

    print("=" * 70)
    print(" Boolean Feature Rule Miner  —  EC2201 Unit I Digital Fundamentals")
    print("=" * 70)

    # [1/6] load + validate
    print("\n[1/6] Loading & validating data")
    df = rm.load_and_validate_csv(args.input)
    target = rm.choose_target(df, args.target)
    print(f"  File : {args.input}")
    print(f"  Rows : {len(df)}   Columns : {', '.join(df.columns)}")
    print(f"  Binary validation: OK (only 0/1 values)")

    # [2/6] candidate generation (inside mine_rules, verbose)
    print(f"\n[2/6] Generating candidate rules "
          f"(max_literals={args.max_literals}, min_support={args.min_support})")

    # [3/6] + [4/6] evaluation, metrics, ranking
    print(f"\n[3/6] Evaluating candidates with AND/OR/NOT gates on 0/1 values")
    res = rm.mine_rules(df, target, args.max_literals, args.min_support,
                        args.min_confidence, args.top_n, verbose=True)
    print(f"\n[4/6] Ranking: F1 -> Confidence -> Support -> simplicity "
          f"(min_confidence={args.min_confidence})")
    top = res["top"]
    if top.empty:
        print(f"\n  {res['message']}")
    else:
        print(f"\n  Rules surviving the filter: {len(res['rules'])}   "
              f"(showing Top-{len(top)})")
        print(rm.format_rules_frame(top).to_string(index=False))

    if res.get("warning"):
        print(f"\n  {res['warning']}")

    # [5/6] deep dive on the best rule
    if res["best"]:
        print("\n[5/6] ")
        deep_dive(res["best"], df, res["features"])
    else:
        print("\n[5/6] No best rule — skipped deep dive.")

    # [6/6] charts
    if not args.no_charts and not top.empty:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        p1 = SCREENSHOT_DIR / "01_top_rules_bar.png"
        p2 = SCREENSHOT_DIR / "02_coverage_vs_confidence.png"
        p3 = SCREENSHOT_DIR / "03_truth_table_heatmap.png"
        plot_top_rules_bar(top, p1)
        plot_coverage_scatter(top, p2)
        var_order = bl.expression_variables(res["best"]["Rule"]) if res["best"] else None
        plot_truth_table_heatmap(res["best"]["Rule"], p3, var_order or None)
        print(f"\n[6/6] Charts saved to {SCREENSHOT_DIR}/ :")
        for f in (p1, p2, p3):
            print(f"      - {f.name}")
    else:
        print("\n[6/6] Charts skipped (--no_charts or no rules).")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
