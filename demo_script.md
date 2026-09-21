# Demo Script — Boolean Feature Rule Miner (≈ 4 minutes)

> Spoken narration for a 3–5 minute demo video.
> `[...]` marks what to show on screen. Pause where indicated.

---

## 1 · The problem (0:00 – 0:30)

"Many datasets in real life are made of pure 0-and-1 flags: symptoms
present or absent, a sensor triggered or not, an item bought or skipped.
The question is: can we find a *simple, human-readable rule* that explains
when the target becomes 1?

A classic answer from Digital Fundamentals is a Boolean expression —
something like: **IF A AND B, OR C AND NOT D, THEN the target is 1.**
That is exactly what this tool mines, automatically, from data — and every
step of it is built with the Unit-I concepts: logic gates, truth tables,
SOP/POS forms, and Quine-McCluskey simplification. No black boxes, no
deep learning — just Boolean algebra, implemented from scratch in Python."

`[Show: title slide / repository tree]`

---

## 2 · The demo (0:30 – 2:30)

**Dataset.**
"First, the data. We generate 500 rows with six binary features A to F and a
target Y. The generator hides a ground-truth rule — Y equals A AND B, OR C
AND NOT D — and then flips five percent of the labels to simulate real-world
noise. A, B, C, D matter; E and F are pure noise that a good rule must
ignore. Because the random seed is fixed, running this twice gives
bit-identical files."

`[Show: data_generator.py output + first rows of data/synthetic.csv]`

**Mining.**
"Now the miner. We open the CLI: `python main.py` with target Y, max
literals 3, min support 0.05, min confidence 0.6. It builds the candidate
space by brute force: every conjunction of up to three literals — that's 232
AND-terms here — and, because max literals is 3, also OR-pairs of strong
terms, so multi-term rules like our hidden rule are reachable. Growth is
pruned the Apriori way: as soon as an AND-term covers fewer rows than the
min support, we stop growing it. In total, 423 candidates."

`[Show: main.py running; the candidate-count lines]`

**Evaluation.**
"Each candidate is evaluated with actual digital logic — AND, OR and NOT
gates running on 0/1 columns — and scored. Support is the fraction of rows
where the rule and the target are both 1; confidence is, given the rule
fires, how often the target really is 1; coverage is how often the rule
fires at all. We also compute precision, recall and F1. Rules are ranked by
F1, then confidence, then support, then simplicity — fewer literals wins a
tie."

`[Show: Top-10 table scrolling into view]`

**The result.**
"And look at rank one: `(A AND B) OR (C AND NOT D)` — the exact hidden rule,
recovered from noisy data, with F1 of 0.96 and confidence 0.969, which is
right at the 1-minus-noise ceiling: no rule can do better on this data. The
noise features E and F do leak into some lower-ranked rules, but they never
outrank the truth — that's the ranking doing its job."

`[Pause — let the table be read for 5 seconds.]`

---

## 3 · Truth table + simplification (2:30 – 3:30)

"Rank one alone isn't proof, so the tool *explains* the best rule. Here is
its 16-row truth table over A, B, C, D — four variables, the classic K-map
size — with the output column computed by the gate evaluator. From the same
table we read the canonical forms: the SOP is the OR of minterms two, six,
ten, twelve, thirteen, fourteen and fifteen; the POS is the AND of the nine
maxterms where the output is zero.

Then we minimize. The module applies the Quine-McCluskey method: minterms
are paired when their bit patterns differ in exactly one position;
unpaired implicants are prime implicants; Petrick's method picks the minimum
cover. For the best rule it confirms the 4-literal form is already minimal:
A·B plus C·D prime.

And to prove the simplifier works, here's the classic textbook example:
`(A AND B) OR (A AND NOT B)`. Q-M minterms ten and eleven combine to the
pattern one-dash — prime implicant A. Before: four literals. After: one.
Same truth table, ten times simpler."

`[Show: notebook cells — truth table, SOP/POS, then the simplification cell]`

"Finally, the gate-level trace: for one real data row, the tool prints every
gate as it fires — NOT on D, AND on C, AND on A-B, OR at the top — exactly
like drawing the circuit on a whiteboard."

`[Show: gate trace output]`

---

## 4 · Results & close (3:30 – 4:15)

"To sum up: on the default dataset the exact hidden rule is mined at rank
one; a fifteen-case test suite — ten normal runs plus five fault cases like
an empty CSV, an all-zero target, invalid values, and contradictory rows —
passes fifteen out of fifteen; and everything is reproducible with one seed.
The notebook runs top to bottom in seconds, and there's an optional
Streamlit app where you can upload your own binary CSV, pick the target,
drag the thresholds, and watch the truth tables and charts update.

The future scope is straightforward: exact minimizers beyond four
variables, rule *lists* instead of a single best rule, and real-world
binarized datasets. But the core message stands: the Unit-I toolkit —
gates, truth tables, canonical forms, Quine-McCluskey — is a complete and
transparent foundation for interpretable rule mining. Thank you."

`[Show: screenshots folder + test summary 15/15]`

---

### Timing notes

| Section | Length | Cumulative |
|---|---|---|
| Problem | 0:30 | 0:30 |
| Demo (data → mining → result) | 2:00 | 2:30 |
| Truth table + simplification + trace | 1:00 | 3:30 |
| Results & close | 0:45 | 4:15 |

Total ≈ **4 minutes** (comfortably inside the 3–5 minute window; drop the
POS sentence in section 3 to reach ~3:45, or expand section 2 with the
weak-target-E experiment to reach ~4:45).
