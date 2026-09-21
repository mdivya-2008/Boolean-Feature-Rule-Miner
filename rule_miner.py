"""
rule_miner.py
=============
Data / AI layer of the Boolean Feature Rule Miner.

Pipeline
--------
    CSV loader + validator  ->  target selection  ->  candidate rule generation
    ->  gate-level evaluation  ->  Support/Confidence/Coverage (+P/R/F1)
    ->  ranking (F1 > Confidence > Support > simplicity)

Candidate rule space (brute force with APRIORI-style pruning)
-------------------------------------------------------------
* Literal      : a feature or its negation, e.g.  A  or  NOT A.
* Conjunction  : AND of 1..`max_literals` literals, e.g.  A AND NOT C.
* SOP candidate: OR of TWO strong conjunctions (generated when
  `max_literals >= 3`) so that multi-term rules such as
  (A AND B) OR (C AND NOT D) — the hidden ground-truth rule — can be mined.
* Constant     : the rule TRUE (needed for the all-ones-target edge case).

Pruning: while growing a conjunction we keep only literals whose combined
coverage is >= min_support.  This is safe because adding another literal to
an AND-term can only DECREASE its coverage (downward-closure property, the
same idea as the Apriori algorithm for association-rule mining).

Metrics for a rule R  ->  prediction "Y = 1"
--------------------------------------------
    Support    = count(R=1 AND Y=1) / N
    Confidence = count(R=1 AND Y=1) / count(R=1)      (a.k.a. Precision)
    Coverage   = count(R=1) / N
    Recall     = count(R=1 AND Y=1) / count(Y=1)
    F1         = 2 * Precision * Recall / (Precision + Recall)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from boolean_logic import AND, OR

# ---------------------------------------------------------------------------
# 1. CSV LOADER + VALIDATOR  (fault handling: only 0/1 allowed)
# ---------------------------------------------------------------------------

def load_and_validate_csv(path) -> pd.DataFrame:
    """Load a CSV and verify it is a valid binary (0/1) dataset.

    Fault cases handled with a clear error message:
        * file missing
        * completely empty file (no header)
        * header only (no data rows)
        * missing values (NaN)
        * non-numeric values such as "yes"
        * numeric values outside {0, 1} such as 2, -1, 0.5
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Input file not found: {path}")

    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty CSV file (no header and no rows): {path}")

    if df.shape[0] == 0:
        raise ValueError(f"CSV '{p.name}' has a header but no data rows — nothing to mine.")
    if df.shape[1] < 2:
        raise ValueError(f"CSV '{p.name}' needs at least 2 columns (features + target).")

    problems = []
    for col in df.columns:
        s = df[col]
        # 1) missing values
        n_missing = int(s.isna().sum())
        if n_missing:
            problems.append(f"column '{col}': {n_missing} missing value(s) — "
                            f"only 0/1 integers are allowed")
        # 2) invalid values
        if s.dtype == object:                      # any non-numeric cell -> object
            bad = sorted({str(v).strip() for v in s.dropna().unique()
                          if str(v).strip().lower() not in ("0", "1")})
            if bad:
                problems.append(f"column '{col}': invalid non-binary value(s) "
                                f"{bad[:5]} — only 0/1 integers are allowed")
        else:
            nums = pd.to_numeric(s, errors="coerce")
            bad = nums.dropna()
            bad = bad[(bad % 1 != 0) | (bad < 0) | (bad > 1)]
            if len(bad):
                problems.append(f"column '{col}': {len(bad)} value(s) outside "
                                f"{{0,1}} (e.g. {bad.head(3).tolist()}) — "
                                f"only 0/1 integers are allowed")

    if problems:
        raise ValueError("Invalid binary dataset — only 0/1 values are allowed. "
                         "Fault details:\n  - " + "\n  - ".join(problems))

    return df.astype(int)


def validate_binary(df: pd.DataFrame) -> list[str]:
    """Non-raising twin of the validator: returns a list of problems (empty = OK)."""
    problems = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        if n_missing:
            problems.append(f"column '{col}': {n_missing} missing value(s)")
        if s.dtype == object:
            bad = sorted({str(v).strip() for v in s.dropna().unique()
                          if str(v).strip().lower() not in ("0", "1")})
            if bad:
                problems.append(f"column '{col}': invalid value(s) {bad[:5]}")
        else:
            nums = pd.to_numeric(s, errors="coerce")
            bad = nums.dropna()
            bad = bad[(bad % 1 != 0) | (bad < 0) | (bad > 1)]
            if len(bad):
                problems.append(f"column '{col}': {len(bad)} value(s) outside {{0,1}}")
    return problems


# ---------------------------------------------------------------------------
# 2. TARGET SELECTOR
# ---------------------------------------------------------------------------

def choose_target(df: pd.DataFrame, target: str | None = None) -> str:
    """Pick the target column (explicit name or sensible default)."""
    if target is None:
        for cand in ("Y", "Target", "target", "label", "Label", df.columns[-1]):
            if cand in df.columns:
                target = cand
                break
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. "
                         f"Available columns: {list(df.columns)}")
    bad = validate_binary(df[[target]])
    if bad:
        raise ValueError(f"Target column '{target}' is not binary: " + "; ".join(bad))
    return target


# ---------------------------------------------------------------------------
# 3. CANDIDATE RULE GENERATOR (brute force + min_support pruning)
# ---------------------------------------------------------------------------

def _lit_str(lit) -> str:
    """(feature, polarity) -> display string, e.g. ('B', -1) -> 'NOT B'."""
    f, pol = lit
    return f if pol == 1 else f"NOT {f}"


def _term_str(lit_set) -> str:
    """Literal set -> conjunction string, e.g. {A+, B-} -> 'A AND NOT B'."""
    return " AND ".join(_lit_str(l) for l in sorted(lit_set, key=lambda x: (x[0], -x[1])))


def _terms_to_expr(terms) -> str:
    """List of literal-sets (product terms) -> canonical rule string."""
    if not terms:
        return "TRUE"
    tsorted = sorted(terms, key=_term_str)
    if len(tsorted) == 1:
        return _term_str(tsorted[0])
    return " OR ".join(f"({_term_str(t)})" for t in tsorted)


def _terms_contradict(t1, t2) -> bool:
    """True if two AND-terms can never be 1 together (contain X and NOT X)."""
    return any((f, -pol) in t2 for f, pol in t1)


def generate_candidates(df: pd.DataFrame,
                        target: str,
                        max_literals: int = 3,
                        min_support: float = 0.05,
                        min_confidence: float = 0.6,
                        verbose: bool = True):
    """Generate candidate rules and pre-evaluate them with the logic gates.

    Returns (candidates, stats) where each candidate is
        {'terms': [frozenset((feature, polarity), ...)],
         'expr':  canonical rule string,
         'series': Pandas Series of 0/1 over the whole dataset}
    """
    if max_literals > 4:
        raise ValueError("max_literals is capped at 4 (truth-table limit)")
    features = [c for c in df.columns if c != target]
    n = len(df)
    min_cnt = math.ceil(min_support * n)      # integer coverage threshold
    y = df[target].astype(int)
    y_pos = int(y.sum())

    # Pre-compute the 0/1 vector of every literal (feature and its negation).
    lit_cols = {}
    for f in features:
        lit_cols[(f, 1)] = df[f].astype(int)
        lit_cols[(f, -1)] = (1 - df[f]).astype(int)
    lit_counts = {lit: int(col.sum()) for lit, col in lit_cols.items()}
    # Order literals by decreasing coverage so the DFS prunes earlier.
    ordered = sorted(lit_cols.keys(), key=lambda l: (-lit_counts[l], l[0], l[1]))

    candidates: list[dict] = []
    conjunctions: list[dict] = []      # conjunctions with <= 2 literals
    n_conj = 0

    # ---- depth-first enumeration of conjunctions with min_support pruning ----
    def dfs(lit_set: frozenset, series: pd.Series, depth: int, start_idx: int):
        nonlocal n_conj
        n_conj += 1
        cnt = int(series.sum())
        if cnt < min_cnt:               # APRIORI downward closure: prune branch
            return
        cand = {"terms": [lit_set], "expr": _terms_to_expr([lit_set]), "series": series}
        candidates.append(cand)
        if len(lit_set) <= 2:
            conjunctions.append(cand)
        if depth < max_literals:
            for i in range(start_idx + 1, len(ordered)):
                lit = ordered[i]
                if any(p[0] == lit[0] for p in lit_set):
                    continue            # A AND NOT A would be the constant 0
                dfs(lit_set | {lit}, AND(series, lit_cols[lit]), depth + 1, i)

    for i, lit in enumerate(ordered):
        if lit_counts[lit] >= min_cnt:
            dfs(frozenset([lit]), lit_cols[lit].copy(), 1, i)

    # ---- SOP candidates: OR of two strong 1/2-literal conjunctions ----
    n_sop = 0
    if max_literals >= 3:
        pool = list(conjunctions)
        if y_pos > 0:                   # keep only "strong" terms -> small pool
            def conf(c):
                cov = int(c["series"].sum())
                return int((c["series"] & y).sum()) / cov if cov else 0.0
            pool = [c for c in pool if conf(c) >= min_confidence]
        else:
            conf = lambda c: 0.0
        pool = sorted(pool, key=lambda c: (-conf(c), -int(c["series"].sum())))[:60]
        seen_expr: set[str] = set()
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                t1, t2 = pool[i]["terms"][0], pool[j]["terms"][0]
                if _terms_contradict(t1, t2):
                    continue            # t1 AND t2 = 0  ->  OR = t1 (redundant)
                if t1 <= t2 or t2 <= t1:
                    continue            # absorption: t1 OR (t1 AND x) = t1
                s = OR(pool[i]["series"], pool[j]["series"])
                if int(s.sum()) < min_cnt:
                    continue
                terms = [t1, t2]
                expr = _terms_to_expr(terms)
                if expr in seen_expr:
                    continue
                seen_expr.add(expr)
                candidates.append({"terms": terms, "expr": expr, "series": s})
                n_sop += 1

    # ---- constant candidate TRUE (needed for the all-ones target case) ----
    candidates.append({"terms": [], "expr": "TRUE",
                       "series": pd.Series(1, index=df.index)})

    stats = {"n_conjunctions": n_conj,
             "n_conjunctions_passed": sum(1 for c in candidates if c["terms"]
                                          and len(c["terms"]) == 1),
             "n_sop": n_sop,
             "n_total": len(candidates)}
    if verbose:
        print(f"  Conjunction candidates : {n_conj} generated, "
              f"{stats['n_conjunctions_passed']} passed min_support={min_support}")
        if max_literals >= 3:
            print(f"  SOP (2-term) candidates : {n_sop}")
        print(f"  Constant candidates     : 1 (TRUE)")
        print(f"  Total candidates        : {len(candidates)}")
    return candidates, stats


# ---------------------------------------------------------------------------
# 4. METRICS  (Support / Confidence / Coverage / Precision / Recall / F1)
# ---------------------------------------------------------------------------

def evaluate_candidates(candidates: list, df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute all metrics for every candidate using the gate-evaluated series."""
    y = df[target].astype(int)
    n = len(df)
    pos = int(y.sum())
    rows = []
    for c in candidates:
        r = c["series"]
        cov_cnt = int(r.sum())
        if cov_cnt == 0:
            continue
        tp = int((r & y).sum())
        confidence = tp / cov_cnt                 # = precision of the rule
        support = tp / n
        coverage = cov_cnt / n
        recall = tp / pos if pos > 0 else 0.0
        f1 = (2 * confidence * recall / (confidence + recall)
              if (confidence + recall) > 0 else 0.0)
        rows.append({
            "Rule": c["expr"],
            "n_literals": sum(len(t) for t in c["terms"]),
            "Support": round(support, 4),
            "Confidence": round(confidence, 4),
            "Coverage": round(coverage, 4),
            "Precision": round(confidence, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "_series": r,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. CONTRADICTION (FAULT-CASE 15) CHECK
# ---------------------------------------------------------------------------

def contradiction_warning(df: pd.DataFrame, target: str, rule_series: pd.Series,
                          top_confidence: float, threshold: float = 0.90) -> str | None:
    """Warn when identical feature values carry different target values.

    A deterministic Boolean function maps each input row to exactly ONE
    output; if the data contains both outputs for the same input, the best
    achievable confidence is mathematically capped below 1.0.
    """
    if top_confidence >= threshold:
        return None
    feats = [c for c in df.columns if c != target]
    sel = df[rule_series.astype(bool)]
    mixed = 0
    for _, g in sel.groupby(feats):
        if g[target].nunique() > 1:
            mixed += len(g)
    if mixed > 0:
        return (f"WARNING: low confidence — {mixed} rows share identical feature "
                f"values but have DIFFERENT target values (contradictory data). "
                f"A deterministic Boolean function cannot fit both, so confidence "
                f"is mathematically capped below 1.0. Check the data quality.")
    return (f"WARNING: low confidence — the best rule has confidence "
            f"{top_confidence:.3f} < {threshold}; the target has a weak/noisy "
            f"Boolean dependency on the given features.")


# ---------------------------------------------------------------------------
# 6. FULL MINING PIPELINE
# ---------------------------------------------------------------------------

def mine_rules(df: pd.DataFrame,
               target: str | None = None,
               max_literals: int = 3,
               min_support: float = 0.05,
               min_confidence: float = 0.6,
               top_n: int = 10,
               verbose: bool = True) -> dict:
    """Run the whole pipeline.  Returns a dict with:

        target, features, n_rows,
        rules : full ranked rule table (after min_confidence filter),
        top   : top-N rows of `rules`,
        best  : dict(Rule, metrics..., series, terms) of the best rule or None,
        warning : low-confidence / contradiction warning or None,
        message : human message when no rule survived the filter,
        stats : candidate-generation statistics
    """
    target = choose_target(df, target)
    y_pos = int(df[target].sum())

    if verbose:
        print(f"  Target column : {target}  "
              f"(positive rows: {y_pos}/{len(df)} = {y_pos / len(df):.1%})")

    candidates, stats = generate_candidates(df, target, max_literals,
                                            min_support, min_confidence, verbose)
    table = evaluate_candidates(candidates, df, target)

    result = {"target": target,
              "features": [c for c in df.columns if c != target],
              "n_rows": len(df),
              "rules": pd.DataFrame(columns=["Rule", "n_literals", "Support",
                                             "Confidence", "Coverage",
                                             "Precision", "Recall", "F1"]),
              "top": pd.DataFrame(columns=["Rule", "n_literals", "Support",
                                           "Confidence", "Coverage",
                                           "Precision", "Recall", "F1"]),
              "best": None, "warning": None, "message": None, "stats": stats}

    if table.empty:
        result["message"] = "No candidate rule had any coverage — nothing to rank."
        return result

    filt = table[table["Confidence"] >= min_confidence - 1e-12]
    # Ranking: F1 first, then Confidence, then Support, then simplicity
    # (fewer literals = simpler = preferred on ties).
    filt = (filt.sort_values(["F1", "Confidence", "Support", "n_literals"],
                             ascending=[False, False, False, True])
            .reset_index(drop=True))
    top = filt.head(top_n)

    result["rules"] = filt.drop(columns=["_series"])
    result["top"] = top.drop(columns=["_series"])

    if not filt.empty:
        best_row = filt.iloc[0]
        result["best"] = {
            "Rule": best_row["Rule"],
            "Support": best_row["Support"],
            "Confidence": best_row["Confidence"],
            "Coverage": best_row["Coverage"],
            "Precision": best_row["Precision"],
            "Recall": best_row["Recall"],
            "F1": best_row["F1"],
            "n_literals": best_row["n_literals"],
            "series": best_row["_series"],
        }
        result["warning"] = contradiction_warning(df, target,
                                                  best_row["_series"],
                                                  best_row["Confidence"])
    else:
        if y_pos == 0:
            result["message"] = ("No rules found: the target column is ALWAYS 0. "
                                 "The output function is the constant FALSE (0) — "
                                 "there is nothing to mine.")
        else:
            result["message"] = (f"No rules satisfy min_confidence={min_confidence}. "
                                 f"Best candidate confidence is "
                                 f"{table['Confidence'].max():.3f} — the data does "
                                 f"not support a reliable Boolean rule (weak-signal case).")
    return result


# ---------------------------------------------------------------------------
# 7. DISPLAY HELPERS
# ---------------------------------------------------------------------------

DISPLAY_COLS = ["Rule", "n_literals", "Support", "Confidence",
                "Coverage", "Precision", "Recall", "F1"]


def format_rules_frame(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Rank-numbered display copy of a rule table."""
    if rules_df.empty:
        return rules_df
    out = rules_df.copy()
    out.insert(0, "#", range(1, len(out) + 1))
    return out[["#"] + DISPLAY_COLS]
