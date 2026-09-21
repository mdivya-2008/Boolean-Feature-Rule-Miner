# Boolean Feature Rule Miner
## Project Report — EC2201 Unit I: Digital Fundamentals

**Type:** Software prototype (Python) · **Scope:** offline, reproducible, no paid APIs

---

### Abstract

This project implements a rule-mining tool that discovers human-readable
IF–THEN Boolean rules from binary datasets. The tool is built around a
**digital-logic core written from scratch**: AND/OR/NOT gates operating on
0/1 values, a Boolean expression parser and evaluator, a 2^k truth-table
generator, SOP/POS conversion, and a Quine–McCluskey simplifier with
Petrick's method. On top of this core, a data layer generates a reproducible
synthetic dataset (500 rows, 6 binary features, 5% label noise) whose hidden
ground truth is the textbook rule **Y = (A AND B) OR (C AND NOT D)**. The
miner brute-forces candidate conjunctions (APRIORI-style min-support
pruning), scores each rule with the logic gates plus
Support/Confidence/Coverage/Precision/Recall/F1, and ranks by
F1 → Confidence → Support → simplicity. The exact hidden rule is recovered
at **rank #1** (F1 = 0.962, confidence 0.969 ≈ the 1 − noise ceiling); a
15-case test suite passes 15/15, and the pipeline is demoed in a CLI, an
executed Jupyter notebook and an optional Streamlit app.

### 1. Introduction

Rule mining seeks compact, interpretable rules such as
`IF (A AND B) OR (C AND NOT D) THEN Y=1`. Most modern tools hide the
machinery inside black boxes; for a Digital Fundamentals course it is far
more instructive when the "intelligence" is just Boolean algebra: a rule is
a combinational logic circuit, a data row is an input vector, and rule
quality is measured by simple counting. This project therefore builds the
digital-logic model first, adds the data/AI layer on top, and keeps every
intermediate state (gate traces, truth tables, prime implicants) visible —
which also makes it a strong viva demonstration of Unit I concepts.

### 2. Digital logic theory used

* **Gates.** AND (product), OR (sum), NOT (complement), each implemented with
  truth-table docstrings and supporting scalars, NumPy arrays and Pandas
  Series element-wise.
* **Boolean algebra & canonical forms.** From the truth table of a rule we
  read its SOP (OR of minterms where F=1, written Σm(…)) and POS (AND of
  maxterms where F=0, ΠM(…)).
* **Quine–McCluskey method.** Minterms are written as k-bit strings; pairs
  differing in exactly one bit are combined into an implicant with a dash;
  the process repeats until no further combination exists. Implicants never
  combined are **prime implicants**; **Petrick's method** then selects the
  minimum cover (essential prime implicants first, then exhaustive
  product-of-sums expansion — trivial for k ≤ 4).
* **Boolean identities** (X·X′=0, X+X′=1, X+X·Y=X, …) are applied as an AST
  rewrite pass as an educational quick check alongside Q-M.

### 3. Methodology

1. **Data.** `data_generator.py` draws uniform binary features with a seeded
   NumPy RNG, computes the hidden rule with gate operations, flips a `noise`
   fraction of labels (default 5%) and writes `data/synthetic.csv`. E, F are
   pure noise so good rules must ignore them; seed 42 ⇒ bit-identical runs.
2. **Validation.** The loader rejects missing cells, non-numeric strings
   (e.g. `yes`) and out-of-range numbers (e.g. 2) with per-column errors.
3. **Candidate generation.** DFS over literals `{A, ¬A, …, F, ¬F}` grows
   conjunctions to `max_literals` (≤4), pruning a branch as soon as its
   coverage drops below `min_support` (downward closure). When
   `max_literals ≥ 3`, OR-pairs of strong (≥ min_confidence) 1/2-literal
   conjunctions are added so multi-term SOP rules are reachable;
   contradictory/absorbed pairs are skipped; the constant TRUE completes the
   space.
4. **Metrics.** Each rule's 0/1 vector is computed once with the gates; then
   Support = #(R∧Y=1)/N, Confidence = #(R∧Y=1)/#(R), Coverage = #(R)/N,
   Precision = Confidence, Recall = #(R∧Y=1)/#(Y=1), F1 = 2PR/(P+R).
5. **Ranking.** F1 desc → Confidence desc → Support desc → literal count
   asc (simplicity on ties); display Top-N.
6. **Explanation.** The best rule gets a truth table (k ≤ 4), SOP/POS,
   Q-M before/after with combination steps, and gate traces on two real
   rows; a detector warns when identical feature rows carry different
   targets (deterministic logic cannot fit both).

### 4. Implementation

Python 3.10+ with NumPy, Pandas, Matplotlib (and optional Streamlit). The
modules are small and separately testable: `boolean_logic.py` (≈550 lines,
no data dependencies beyond Pandas), `rule_miner.py` (loader, generator,
metrics), `main.py` (CLI + charts), `app.py` (UI). Everything runs locally;
typical run time on 500 rows is ~2 s, on 1000 rows ~3 s.

### 5. Results

**Table 1 — Top-5 mined rules (seed 42, 500 rows, default thresholds).**
The hidden rule is recovered exactly at rank 1; its confidence 0.969 ≈
1 − 0.05 noise, i.e. the best any rule can do on this data.

| Rank | Rule | Support | Confidence | Coverage | F1 |
|---:|---|---:|---:|---:|---:|
| 1 | (A AND B) OR (C AND NOT D)  ← hidden rule | 0.432 | 0.9686 | 0.446 | **0.9621** |
| 2 | (B) OR (C AND NOT D) | 0.440 | 0.7261 | 0.606 | 0.8318 |
| 3 | (A AND B) OR (C) | 0.442 | 0.7199 | 0.614 | 0.8293 |
| 4 | (A AND B) OR (NOT D) | 0.436 | 0.6877 | 0.634 | 0.8029 |
| 5 | (B AND NOT E) OR (C AND NOT D) | 0.350 | 0.8294 | 0.422 | 0.8009 |

**Table 2 — Simplification demo (Q-M) for `(A AND B) OR (A AND NOT B)`.**

| Step | Content |
|---|---|
| Minterms | 2(10), 3(11) |
| Round 1 | 10 + 11 → 1− |
| Prime implicants | {1−} = {A} |
| Result | **F = A** (4 literals → 1 literal; identical truth tables verified) |

For the mined best rule, Q-M confirms `C·D′ + A·B` is **already minimal**
(minterms {2,6,10,12,13,14,15} → prime implicants `11--` (A·B) and `--10`
(C·D′)).

**Table 3 — Test suite (15/15 PASS).** 10 normal cases (default run, hidden
rule recovery at `max_literals=3`, weak target E, high min_support, low
min_confidence, top_n variants, seed-123 reproducibility, 1000-row
scalability, simplification identity, 4-row truth table) and 5 edge/fault
cases (empty CSV, all-zero target → "constant FALSE" message, all-one target
→ rule TRUE, NaN/2/“yes” validation error, contradictory data →
low-confidence warning).

**Observations.** (i) Noise features E/F occasionally leak into lower-ranked
rules (e.g. rank 5 uses ¬E) but never beat the true rule on F1 — the ranking
does its job. (ii) Raising `min_support` to 0.2 shrinks the rule space as
predicted by downward closure; lowering `min_confidence` grows it. (iii) On
contradictory data the tool correctly reports that confidence is
mathematically capped below 1.0.

### 6. Conclusion

All acceptance criteria are met: the digital-logic core is implemented from
scratch and drives the pipeline; the hidden rule is recovered at rank 1 with
confidence at the noise ceiling; the best rule is fully explained (truth
table, SOP/POS, Q-M, gate trace); 15/15 tests pass; and the notebook
executes in seconds. The main limitations are the k ≤ 4 Q-M bound and the
single-best-rule output; Espresso-style minimization, rule lists and
real-world datasets are the natural next steps. Overall, the classic Unit I
toolset — gates, truth tables, canonical forms, Q-M — is a complete,
transparent and effective foundation for interpretable rule mining.

### 7. References

1. M. M. Mano, M. D. Ciletti, *Digital Design: Principles and Practices*, 6th ed., Pearson, 2021. (gates, K-maps, logic minimization)
2. L. L. Roth, L. D. Kinney, *Fundamentals of Logic Design*, 7th ed., Cengage, 2014. (SOP/POS, Quine–McCluskey, Petrick's method)
3. W. V. Quine, "The problem of simplifying truth functions," *American Mathematical Monthly*, vol. 59, no. 6, pp. 521–531, 1952.
4. E. J. McCluskey, "Introduction to the minimization of Boolean functions," *Bell System Technical Journal*, vol. 35, no. 8, pp. 866–884, 1956.
5. R. Agrawal, R. Srikant, "Fast algorithms for mining association rules," in *Proc. 20th VLDB*, 1994, pp. 487–499. (support/confidence, Apriori pruning)
6. J. Han, H. Pei, Y. Yin, R. Mao, "Mining frequent patterns without candidate generation," in *Proc. SIGMOD*, 2000, pp. 1–12.
7. T. M. Mitchell, *Machine Learning*, McGraw-Hill, 1997. (rule learning, evaluation metrics)
