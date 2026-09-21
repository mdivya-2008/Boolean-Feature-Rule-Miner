"""
build_notebook.py
=================
Builds notebooks/demo.ipynb (cell by cell) so the demo notebook stays
in sync with the code.  Regenerate with:

    python notebooks/build_notebook.py

Then (re)execute it to refresh the saved outputs:

    python - <<'EOF'
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read("notebooks/demo.ipynb", as_version=4)
    NotebookClient(nb, timeout=180, kernel_name="python3").execute()
    nbformat.write(nb, "notebooks/demo.ipynb")
    EOF
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "demo.ipynb"


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}

    nb.cells += [
        md("""# Boolean Feature Rule Miner — Full Demo
**EC2201 Unit I · Digital Fundamentals** (Boolean algebra, gates, truth tables, SOP/POS, K-map/Q-M simplification)

This notebook walks through the complete prototype, top to bottom:

1. **Digital-logic core** — AND/OR/NOT gates, expression parser & evaluator *(from scratch, no sklearn)*
2. **Gate-level evaluation trace** on one row
3. **Truth-table generation** (k ≤ 4 variables)
4. **SOP / POS conversion** (canonical forms)
5. **Boolean simplification** — identities + Quine–McCluskey
6. **Synthetic data** — hidden rule `Y = (A AND B) OR (C AND NOT D)` + 5% noise (seed 42)
7. **Rule mining** — brute-force candidates + min-support pruning, Support/Confidence/Coverage/F1
8. **Deep dive** on the best rule (truth table, SOP/POS, simplification, gate trace)
9. **Charts** — Top-10 bars, Coverage vs Confidence scatter, truth-table heatmap

> Run time: well under 2 minutes."""),

        md("## 0 · Setup"),
        code("""import sys
from pathlib import Path

# locate the project root (works from any working directory)
here = Path.cwd().resolve()
ROOT = next((p for p in [here, *here.parents] if (p / "boolean_logic.py").exists()), here)
sys.path.insert(0, str(ROOT))
print("Project root:", ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import boolean_logic as bl
import rule_miner as rm
from data_generator import build_dataset
pd.set_option("display.width", 120)"""),

        md("""## 1 · Digital-logic core — gates, parser, evaluator
The gates operate on **0/1 integers, NumPy arrays and Pandas Series**, so the
*same* gate used in a viva on one row also runs on the whole dataset.

**AND** is 1 only when both inputs are 1 · **OR** is 1 when at least one input is 1 · **NOT** inverts."""),
        code("""# --- gates on single bits ------------------------------------------------
print("Gates on single bits:")
print(f"  AND(1,1)={bl.AND(1,1)}  AND(1,0)={bl.AND(1,0)}  AND(0,0)={bl.AND(0,0)}")
print(f"  OR(0,1) ={bl.OR(0,1)}   OR(0,0) ={bl.OR(0,0)}   OR(1,1) ={bl.OR(1,1)}")
print(f"  NOT(1)  ={bl.NOT(1)}    NOT(0)  ={bl.NOT(0)}")

# --- the same gates on whole columns (element-wise) -----------------------
sample = pd.DataFrame({"A": [1, 0, 1, 1],
                       "B": [1, 1, 0, 0],
                       "C": [0, 1, 1, 1],
                       "D": [1, 0, 1, 0]})
sample["A AND B"]      = bl.AND(sample["A"], sample["B"])
sample["C AND NOT D"]  = bl.AND(sample["C"], bl.NOT(sample["D"]))
sample["rule"]         = bl.OR(sample["A AND B"], sample["C AND NOT D"])
print("\\nGates on Pandas Series (element-wise over 4 rows):")
sample"""),

        md("""## 2 · Expression evaluation + gate-level trace
`evaluate()` parses `(A AND B) OR (C AND NOT D)` (recursive-descent parser,
precedence NOT > AND > OR) and evaluates it.  `evaluate_row()` does the same
on **one row** and records every gate application — the digital-logic view."""),
        code("""expr = "(A AND B) OR (C AND NOT D)"
print("Rule:", expr)
print("Parsed variables:", bl.expression_variables(expr))

# evaluate on the sample frame (whole columns at once)
vals = bl.evaluate(expr, sample)
print("Vectorised evaluation on 4 rows:", list(vals))

# gate-level trace on ONE row: A=0, B=1, C=1, D=0  (rule fires)
row = {"A": 0, "B": 1, "C": 1, "D": 0}
out, trace = bl.evaluate_row(expr, row)
print(f"\\nRow {row}  ->  F = {out}")
print(bl.format_trace(trace))"""),

        md("""## 3 · Truth-table generator (k ≤ 4 variables)
For a rule with k variables we enumerate all 2^k input combinations in binary
counting order and compute the output column `Y` with the gate evaluator."""),
        code("""tt = bl.generate_truth_table("A AND NOT B")
print("Truth table of (A AND NOT B):")
tt

tt_hidden = bl.generate_truth_table(expr)
print(f"\\nTruth table of the hidden rule (k=4 -> {len(tt_hidden)} rows):")
tt_hidden"""),

        md("""## 4 · SOP / POS conversion
Canonical forms are read straight off the truth table:
**SOP** = OR of the minterms where Y=1 · **POS** = AND of the maxterms where Y=0."""),
        code("""sop = bl.to_sop_canonical(expr)
pos = bl.to_pos_canonical(expr)
print("SOP:", sop["sum_notation"])
print("F =", sop["algebraic"])
print()
print("POS:", pos["product_notation"])
print("F =", pos["algebraic"])"""),

        md("""## 5 · Boolean simplification (identities + Quine–McCluskey)
Classic textbook example: **(A AND B) OR (A AND NOT B) = A**
(factoring: A·(B + B') = A·1 = A).  The module does it *mechanically* with the
Quine–McCluskey method: minterms → combine patterns differing in one bit →
prime implicants → Petrick's method selects the minimal cover."
"""),
        code("""s = bl.simplify_expression("(A AND B) OR (A AND NOT B)")
print("BEFORE:", s["original"], f'({s["literal_before"]} literals)')
for line in s["steps"]:
    print("  ", line)
print("AFTER :", s["simplified_algebraic"], f'({s["literal_after"]} literals)')
print()
# identity-laws quick pass (educational)
si = bl.simplify_by_identities("(A AND B) OR (A AND NOT B)")
print("Identity pass:", si["original"], "->", si["simplified"])"""),

        md("""## 6 · Synthetic data (seed 42, 500 rows, 5% label noise)
Ground truth: **Y = (A AND B) OR (C AND NOT D)**; E and F are pure noise
features that a good rule must ignore.  Reproducible via the seed."""),
        code("""df = build_dataset(n_rows=500, n_features=6, noise=0.05, seed=42)
print(f"rows={len(df)}  columns={list(df.columns)}  positive Y = {df['Y'].sum()} ({df['Y'].mean():.1%})")
df.head(10)"""),

        md("""## 7 · Rule mining (brute force + min-support pruning)
Candidates: all conjunctions of 1–3 literals (232 here) **plus** OR-pairs of
two strong conjunctions (so multi-term SOP rules can be mined) **plus** TRUE.
Metrics per rule R → "Y=1":
Support = P(R & Y=1) · Confidence = P(Y=1|R) · Coverage = P(R) · Recall, F1.
Ranked by **F1 → Confidence → Support → fewer literals**."""),
        code("""res = rm.mine_rules(df, "Y", max_literals=3, min_support=0.05,
                   min_confidence=0.6, top_n=10, verbose=True)
print(f"\\nRules surviving min_confidence=0.6: {len(res['rules'])}")
rm.format_rules_frame(res["top"])"""),

        md("""## 8 · Deep dive — the best rule
Truth table + SOP/POS + Q-M simplification + a gate-level trace on real data
rows (one where the rule fires, one where it does not)."
"""),
        code("""best = res["best"]
print("BEST RULE:", best["Rule"])
print(f"Support={best['Support']}  Confidence={best['Confidence']}  "
      f"Coverage={best['Coverage']}  Recall={best['Recall']}  F1={best['F1']}")

var_order = bl.expression_variables(best["Rule"])
tt = bl.generate_truth_table(best["Rule"], var_order)
print("\\nTruth table:")
tt

sop = bl.to_sop_canonical(best["Rule"], var_order)
pos = bl.to_pos_canonical(best["Rule"], var_order)
print(f"\\nSOP: {sop['sum_notation']}")
print(f"POS: {pos['product_notation']}")

s = bl.simplify_expression(best["Rule"], var_order)
print(f"\\nSimplified: {s['simplified_algebraic']}  "
      f"({'already minimal' if s['already_minimal'] else 'reduced'})")
for line in s["steps"]:
    print("  ", line)

# gate trace on two real rows (one F=1, one F=0)
series = best["series"]
for idx in [series[series == 1].index[0], series[series == 0].index[0]]:
    row = {f: int(df.loc[idx, f]) for f in df.columns if f != "Y"}
    out, trace = bl.evaluate_row(best["Rule"], row)
    print(f"\\nRow {idx}: {row} -> F={out}")
    print(bl.format_trace(trace))"""),

        md("## 9 · Charts"),
        code("""import main as m   # chart helpers (also used by make_screenshots.py / app.py)

fig1 = m.plot_top_rules_bar(res["top"])
fig2 = m.plot_coverage_scatter(res["top"])
fig3 = m.plot_truth_table_heatmap(best["Rule"], var_order=var_order)
plt.show()"""),

        md("""## 10 · Conclusion / acceptance checklist
- [x] Core digital-logic module written from scratch (gates, parser, evaluator)
- [x] Hidden rule `(A AND B) OR (C AND NOT D)` recovered at **rank #1** despite 5% label noise
- [x] Best rule shown with full truth table + SOP/POS + Q-M simplification
- [x] Gate-level trace on real data rows
- [x] 15/15 tests pass — `python tests/test_cases.py`
- [x] Charts in `screenshots/` — `python make_screenshots.py`

**Viva talking points:** the miner is an exhaustive (pruned) search over the
Boolean hypothesis space — the same machinery as K-map grouping, generalized
to arbitrary dataset sizes; Confidence of the recovered rule ≈ 0.97 ≈ the
(1 − noise) ceiling, which is exactly what a *perfect* rule should achieve
on 5%-noisy data.""")
    ]
    return nb


def main():
    nb = build()
    nbf.write(nb, OUT)
    print(f"Wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
