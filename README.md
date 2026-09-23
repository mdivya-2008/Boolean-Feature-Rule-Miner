naa n# 🔢 Boolean Feature Rule Miner
https://fg2jh4wq-8501.inc1.devtunnels.ms/
**A working prototype for a college project (EC2201 Unit I — Digital Fundamentals).**

Given a **binary dataset** (features are all 0/1 — symptoms, shopping flags,
sensor bits…), this tool automatically mines **human-readable IF-THEN Boolean
rules** that predict a binary target, e.g.

```
IF (A AND B) OR (C AND NOT D)  THEN  Y = 1
```

Every rule is **evaluated with real digital logic** (AND/OR/NOT gates on 0/1),
checked against a full **truth table**, converted to **SOP/POS**, minimized
with the **Quine–McCluskey** method, and scored with
**Support / Confidence / Coverage / Precision / Recall / F1**.

> The digital-logic core is implemented **from scratch** (no sklearn, no
> external Boolean-algebra libraries) so every step maps 1:1 to the EC2201
> Unit I syllabus — ideal for viva questions.

---

## 1 · Objective

Build a small, fully local and reproducible data-mining tool that:

1. loads and validates a binary (0/1) CSV,
2. brute-force generates candidate Boolean rules (with APRIORI-style
   min-support pruning),
3. evaluates each rule with digital-logic gates,
4. ranks rules by F1 → Confidence → Support → simplicity,
5. explains the best rule with a truth table, SOP/POS, Q-M simplification
   and a gate-level evaluation trace,
6. visualises results (bar chart, scatter, heatmap) and ships a
   15-test suite + Streamlit UI + Jupyter demo.

## 2 · EC2201 Unit I concepts → where they live in the code

| # | Unit I concept | Where it is implemented |
|---|----------------|-------------------------|
| 1 | Logic gates AND / OR / NOT (truth tables in docstrings) | `boolean_logic.py` → `AND()`, `OR()`, `NOT()` |
| 2 | Boolean expression parsing & evaluation | `boolean_logic.py` → `_Parser`, `evaluate()`, `evaluate_row()` |
| 3 | Truth-table construction (2^k rows) | `boolean_logic.py` → `generate_truth_table()` |
| 4 | Canonical SOP (Σm) and POS (ΠM) forms | `boolean_logic.py` → `to_sop_canonical()`, `to_pos_canonical()` |
| 5 | Boolean identities (X·X′=0, X+X′=1, absorption…) | `boolean_logic.py` → `_normalize_ast()`, `simplify_by_identities()` |
| 6 | Quine–McCluskey minimization + prime implicants + Petrick's method | `boolean_logic.py` → `_quine_mccluskey()`, `_select_cover()` |
| 7 | Gate-level evaluation trace (like a logic diagram) | `boolean_logic.py` → `evaluate_row()` + `format_trace()` |
| 8 | Association-rule style metrics (support/confidence) | `rule_miner.py` → `evaluate_candidates()` |
| 9 | APRIORI downward-closure pruning | `rule_miner.py` → `generate_candidates()` (DFS with min-support cut) |

## 3 · Assumptions

* Every column is a **binary 0/1 integer**. Anything else (NaN, `yes`, `2`)
  is a **fault** → clear `ValueError` with per-column details.
* Rules only predict **Y = 1** (positive-class rules), per the spec.
* Candidate space:
  * conjunctions (AND-terms) of **1 … `max_literals`** literals
    (a literal = feature or NOT feature; `A AND NOT A` is pruned as the
    constant 0),
  * **OR-pairs** of two *strong* (≥ min_confidence) 1/2-literal conjunctions —
    generated only when `max_literals ≥ 3` so multi-term rules like
    `(A AND B) OR (C AND NOT D)` can be mined,
  * the constant rule **TRUE** (needed when the target is always 1).
* Truth tables and Q-M are limited to **k ≤ 4 variables** (spec); mined rules
  never exceed 4 literals, so this always holds.
* **Support** follows the spec definition: `P(R AND Y=1) = count(R=1 & Y=1)/N`.
  (Confidence = Precision here, since the rule always predicts 1.)
* Fully offline: no APIs, no deep learning, fixed seed ⇒ bit-identical results.

## 4 · Input CSV format

* Header row required, at least 2 columns (features + target).
* **Every cell must be an integer 0 or 1.**
* Missing values, non-numeric strings and out-of-range numbers are rejected
  with a detailed error (fault case 14).

Example (the bundled `data/synthetic.csv` looks like this):

```csv
A,B,C,D,E,F,Y
0,1,1,0,0,1,1
0,1,0,0,1,1,0
1,1,1,1,1,0,1
1,0,1,0,0,1,1
1,1,0,1,1,0,1
```

## 5 · Installation

```bash
cd boolean-feature-rule-miner
pip install -r requirements.txt        # numpy, pandas, matplotlib, streamlit
```

Python 3.10+ required (developed & tested on 3.13).

## 6 · How to run

### a) Generate the dataset (500 rows, seed 42, 5% label noise)

```bash
python data_generator.py
# custom:
python data_generator.py --n_rows 1000 --n_features 6 --noise 0.10 --seed 123 \
    --output data/synthetic.csv
```

Hidden ground truth: **Y = (A AND B) OR (C AND NOT D)**; E, F are pure noise.

### b) Mine & display Top-N rules

```bash
python main.py --input data/synthetic.csv --target Y \
    --max_literals 3 --min_support 0.05 --min_confidence 0.6 --top_n 10
```

### c) Run the 15 test cases

```bash
python tests/test_cases.py        # standalone (PASS/FAIL per test)
pytest tests/test_cases.py -s     # or via pytest
```

### d) Jupyter notebook (executed outputs included)

```bash
jupyter lab notebooks/demo.ipynb
```

### e) Optional Streamlit web UI

```bash
streamlit run app.py
```

### f) Regenerate the 4 screenshots

```bash
python make_screenshots.py
```

## 7 · Sample output (actual run, seed 42)

```
======================================================================
 Boolean Feature Rule Miner  —  EC2201 Unit I Digital Fundamentals
======================================================================

[1/6] Loading & validating data
  File : data/synthetic.csv
  Rows : 500   Columns : A, B, C, D, E, F, Y
  Binary validation: OK (only 0/1 values)

[2/6] Generating candidate rules (max_literals=3, min_support=0.05)
  Target column : Y  (positive rows: 226/500 = 45.2%)
  Conjunction candidates : 232 generated, 232 passed min_support=0.05
  SOP (2-term) candidates : 190
  Constant candidates     : 1 (TRUE)
  Total candidates        : 423

[4/6] Ranking: F1 -> Confidence -> Support -> simplicity (min_confidence=0.6)
  Rules surviving the filter: 228   (showing Top-10)
 #                           Rule  n_literals  Support  Confidence  Coverage  Precision  Recall     F1
 1     (A AND B) OR (C AND NOT D)           4    0.432      0.9686     0.446     0.9686  0.9558 0.9621
 2           (B) OR (C AND NOT D)           3    0.440      0.7261     0.606     0.7261  0.9735 0.8318
 3               (A AND B) OR (C)           3    0.442      0.7199     0.614     0.7199  0.9779  0.8293
 4           (A AND B) OR (NOT D)           3    0.436      0.6877     0.634     0.6877  0.9646  0.8029
 5 (B AND NOT E) OR (C AND NOT D)           4    0.350      0.8294     0.422     0.8294  0.7743  0.8009
 6           (B AND F) OR (C AND NOT D)     4    0.356      0.7946     0.448     0.7946  0.7876  0.7911
 7           (A AND B) OR (C AND F)         4    0.350      0.7955     0.440     0.7955  0.7743  0.7848
 8       (A AND B) OR (C AND NOT E)         4    0.354      0.7832     0.452     0.7832  0.7832  0.7832
 9           (A AND B) OR (C AND E)         4    0.344      0.8037     0.428     0.8037  0.7611  0.7818
10 (A AND B) OR (NOT D AND NOT E)           4    0.350      0.7883     0.444     0.7883  0.7743  0.7812

 DEEP DIVE — best rule (truth table + SOP/POS + simplification)
  IF  (A AND B) OR (C AND NOT D)  THEN  Y = 1
  Support=0.4320  Confidence=0.9686  Coverage=0.4460  F1=0.9621

  SOP (Sum-of-Products) : Σm(2, 6, 10, 12, 13, 14, 15)
  POS (Product-of-Sums) : ΠM(0, 1, 3, 4, 5, 7, 8, 9, 11)

  Simplification (Boolean identities + Quine-McCluskey):
      before : (A AND B) OR (C AND NOT D)   (4 literals)
      Minterms (F=1): 2(0010), 6(0110), 10(1010), 12(1100), 13(1101), 14(1110), 15(1111)
      Round 1: 0010 + 0110 -> 0-10;  0010 + 1010 -> -010;  0110 + 1110 -> -110;  ...
      Round 2: 0-10 + 1-10 -> --10;  -010 + -110 -> --10;  110- + 111- -> 11--;  ...
      Round 3: no further combinations — remaining implicants are prime
      after  : C·D' + A·B   (4 literals)  ->  already minimal

  Gate-level evaluation trace (real data rows):
    Row 0:  {'A': 0, 'B': 1, 'C': 1, 'D': 0, ...}   ->   F = 1
  Step 1 | Gate: AND | inputs: A=0, B=1                     | output: 0
  Step 2 | Gate: NOT | inputs: D=0                          | output: 1
  Step 3 | Gate: AND | inputs: C=1, NOT D=1                 | output: 1
  Step 4 | Gate: OR  | inputs: (A AND B)=0, (C AND NOT D)=1 | output: 1

[6/6] Charts saved to screenshots/ :
      - 01_top_rules_bar.png
      - 02_coverage_vs_confidence.png
      - 03_truth_table_heatmap.png
```

**Result:** the hidden rule `(A AND B) OR (C AND NOT D)` is recovered at
**rank #1** (F1 = 0.962, confidence = 0.969 ≈ the 1 − noise = 0.95 ceiling —
slightly above it by chance of the 5% flips).

## 8 · Folder structure

```
boolean-feature-rule-miner/
├── main.py                  # CLI: full pipeline + deep dive + 3 charts
├── boolean_logic.py         # gates, parser, evaluator, truth tables, SOP/POS, Q-M
├── rule_miner.py            # CSV loader/validator, candidate generator, metrics, ranking
├── data_generator.py        # reproducible synthetic data (seed=42, 5% noise)
├── app.py                   # optional Streamlit UI
├── make_screenshots.py      # generates the 4 PNGs in screenshots/
├── requirements.txt         # pinned versions
├── README.md                # this file
├── demo_script.md           # 3–5 min video narration script
├── data/
│   └── synthetic.csv        # generated, 500 rows × 7 columns
├── notebooks/
│   ├── demo.ipynb           # full step-by-step demo (executed outputs saved)
│   └── build_notebook.py    # rebuilds demo.ipynb from source
├── tests/
│   └── test_cases.py        # 15 test cases (10 normal + 5 edge/fault)
├── report/
│   └── report.md            # 800–1000 word project report + references
└── screenshots/
    ├── 01_top_rules_bar.png
    ├── 02_coverage_vs_confidence.png
    ├── 03_truth_table_heatmap.png
    └── 04_simplification_before_after.png
```

## 9 · Test summary (actual run: **15/15 PASS**)

| # | Test | Type | Result |
|---|------|------|--------|
| 1 | Default dataset, target Y, `max_literals=2` | normal | PASS |
| 2 | `max_literals=3` recovers hidden rule in Top-5 (F1 > 0.9) | normal | PASS |
| 3 | Target E (random feature) → weak rules only | normal | PASS |
| 4 | `min_support=0.2` → fewer rules, all Coverage ≥ 0.2 | normal | PASS |
| 5 | `min_confidence=0.5` → at least as many rules | normal | PASS |
| 6 | `top_n=5` vs `top_n=20` display correctly | normal | PASS |
| 7 | Seed 123: byte-identical data + identical Top-10 twice | normal | PASS |
| 8 | 1000-row dataset scales (≪ 60 s), hidden rule still Top-5 | normal | PASS |
| 9 | `(A AND B) OR (A AND NOT B)` simplifies to `A` | normal | PASS |
| 10 | Truth table of `(A AND NOT B)`: 4 rows verified | normal | PASS |
| 11 | Empty CSV → graceful `ValueError` | edge | PASS |
| 12 | All-zeros target → no rules + message (constant FALSE) | edge | PASS |
| 13 | All-ones target → best rule is constant `TRUE` (conf = 1.0) | edge | PASS |
| 14 | NaN / `2` / `"yes"` → validator error listing faults | edge | PASS |
| 15 | Contradictory rows (same X, different Y) → low-confidence warning | edge | PASS |

## 10 · Future scope

* **Larger k** — Espresso/SETM or BDD-based exact minimizers for >4 variables.
* **Multi-class targets** — one-vs-rest rule sets or decision-list output.
* **Rule lists** — greedy sequential covering (like CN2/PRISM) instead of a
  single best rule.
* **Real datasets** — e.g. Pima Indians diabetes (binarized), with proper
  train/test splits and cross-validation of the mined rules.
* **Interactive K-map** — draw the 4×4 K-map of any mined rule in the UI.
* **Explanation export** — LaTeX/PDF report of the top rules for the project
  file.

## 11 · Definition of done — checklist

- [x] `pip install -r requirements.txt && python data_generator.py && python main.py`
      runs without error and prints Top-10 rules
- [x] Hidden rule `(A AND B) OR (C AND NOT D)` found in Top-5 (**rank #1**)
- [x] Best rule shows full truth table + SOP/POS + simplified form
- [x] Charts saved as PNG in `screenshots/`
- [x] All 15 tests pass with clear PASS/FAIL messages
- [x] Notebook runs top-to-bottom in < 2 min (measured: **~4 s** of compute)
