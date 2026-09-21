"""
test_cases.py
=============
15 test cases for the Boolean Feature Rule Miner
(10 normal + 5 edge/fault cases, per the project specification).

Run in either mode:

    python tests/test_cases.py          # standalone: prints PASS/FAIL per test
    pytest tests/test_cases.py -s       # pytest mode (same assertions)

Every test prints its intermediate output (datasets, tables, messages)
followed by PASS/FAIL.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

# make the project root importable no matter where we run from
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

import boolean_logic as bl
import rule_miner as rm
from data_generator import build_dataset

TMP = ROOT / "data" / "_test_tmp"

DEFAULTS = dict(max_literals=3, min_support=0.05, min_confidence=0.6, top_n=10)


def _check(cond: bool, msg: str):
    """Raise an AssertionError with a clear message (works for both runners)."""
    if not cond:
        raise AssertionError(msg)


def _load_default_csv() -> pd.DataFrame:
    """Load the default dataset (generating it first if it is missing)."""
    csv_path = ROOT / "data" / "synthetic.csv"
    if not csv_path.exists():
        from data_generator import PROJECT_ROOT
        build_dataset().to_csv(csv_path, index=False)
        print(f"  (generated {csv_path} first — it was missing)")
    return rm.load_and_validate_csv(csv_path)


# ---------------------------------------------------------------------------
# NORMAL TESTS (1-10)
# ---------------------------------------------------------------------------

def test_01_default_dataset_targetY_maxL2():
    """1. Default dataset, target Y, max_literals=2."""
    df = _load_default_csv()
    print(f"  dataset: {len(df)} rows, columns {list(df.columns)}")
    res = rm.mine_rules(df, "Y", max_literals=2, top_n=10,
                        verbose=False)
    print("  Top-3 rules (max_literals=2):")
    print(res["top"].head(3).to_string(index=False))
    _check(not res["top"].empty, "expected at least one rule")
    _check((res["top"]["Confidence"] >= 0.6 - 1e-9).all(),
           "min_confidence=0.6 was violated")
    _check((res["top"]["n_literals"] <= 2).all(),
           "max_literals=2 was violated")


def test_02_maxL3_recovers_hidden_rule():
    """2. max_literals=3 recovers the hidden rule (A AND B) OR (C AND NOT D)."""
    df = build_dataset()
    res = rm.mine_rules(df, "Y", top_n=10, verbose=False)
    hidden = bl.ast_to_expr(bl.parse_expression("(A AND B) OR (C AND NOT D)"))
    top5 = res["top"].head(5)["Rule"].tolist()
    print(f"  hidden rule expected: {hidden}")
    print("  Top-5 mined rules:")
    for i, r in enumerate(top5, 1):
        print(f"    {i}. {r}")
    _check(hidden in top5, f"hidden rule {hidden!r} NOT in top-5: {top5}")
    f1 = res["top"].head(5).loc[res["top"].head(5)["Rule"] == hidden, "F1"].iloc[0]
    print(f"  F1 of recovered hidden rule: {f1:.4f}")
    _check(f1 > 0.9, f"hidden rule F1 too low ({f1:.3f})")


def test_03_change_target_to_E_weak_rule():
    """3. Change target to E (random feature) -> only weak rules, no crash."""
    df = build_dataset()
    res = rm.mine_rules(df, "E", top_n=5, verbose=False)
    if res["top"].empty:
        print("  No rules survived min_confidence=0.6 — expected for a random target.")
        print(f"  message: {res['message']}")
    else:
        print("  Top-3 rules for target E (all should be weak):")
        print(res["top"].head(3).to_string(index=False))
        _check(res["top"]["Confidence"].max() < 0.8,
               "target E is independent of features — confidence should be weak (<0.8)")


def test_04_high_min_support_02():
    """4. High min_support=0.2 -> strictly fewer rules, all Coverage >= 0.2."""
    df = build_dataset()
    base = rm.mine_rules(df, "Y", min_support=0.05, top_n=100, verbose=False)
    high = rm.mine_rules(df, "Y", min_support=0.2, top_n=100, verbose=False)
    print(f"  rules with min_support=0.05 : {len(base['rules'])}")
    print(f"  rules with min_support=0.20 : {len(high['rules'])}")
    _check(len(high["rules"]) <= len(base["rules"]),
           "raising min_support must not increase the rule count")
    _check((high["rules"]["Coverage"] >= 0.2 - 1e-9).all(),
           "a rule with Coverage < 0.2 survived min_support=0.2")


def test_05_low_min_confidence_05():
    """5. Low min_confidence=0.5 -> at least as many rules as the default."""
    df = build_dataset()
    base = rm.mine_rules(df, "Y", min_confidence=0.6, top_n=200, verbose=False)
    low = rm.mine_rules(df, "Y", min_confidence=0.5, top_n=200, verbose=False)
    print(f"  rules with min_confidence=0.6 : {len(base['rules'])}")
    print(f"  rules with min_confidence=0.5 : {len(low['rules'])}")
    _check(len(low["rules"]) >= len(base["rules"]),
           "lowering min_confidence must not decrease the rule count")


def test_06_top_n_5_vs_20():
    """6. top_n=5 vs top_n=20 display correctly."""
    df = build_dataset()
    r5 = rm.mine_rules(df, "Y", top_n=5, verbose=False)
    r20 = rm.mine_rules(df, "Y", top_n=20, verbose=False)
    total = len(r20["rules"])
    print(f"  total rules after filter : {total}")
    print(f"  top_n=5  -> displayed {len(r5['top'])}")
    print(f"  top_n=20 -> displayed {len(r20['top'])}")
    _check(len(r5["top"]) == min(5, total), "top_n=5 returned wrong number of rules")
    _check(len(r20["top"]) == min(20, total), "top_n=20 returned wrong number of rules")
    _check(list(r5["top"]["Rule"]) == list(r20["top"]["Rule"])[:5],
           "top-5 of the two runs must be identical")


def test_07_custom_seed_123_reproducibility():
    """7. Custom seed=123: two runs give byte-identical data and identical Top-10."""
    d1, d2 = build_dataset(seed=123), build_dataset(seed=123)
    s1 = d1.to_csv(index=False)
    s2 = d2.to_csv(index=False)
    print(f"  dataset bytes identical: {s1 == s2}  (pos rate = {d1['Y'].mean():.3f})")
    _check(d1.equals(d2) and s1 == s2, "seed=123 datasets differ between runs")
    r1 = rm.mine_rules(d1, "Y", verbose=False)
    r2 = rm.mine_rules(d2, "Y", verbose=False)
    print("  Top-3 rules (run 1):", list(r1["top"]["Rule"].head(3)))
    print("  Top-3 rules (run 2):", list(r2["top"]["Rule"].head(3)))
    _check(list(r1["top"]["Rule"]) == list(r2["top"]["Rule"]),
           "Top-10 rules differ between two identical-seed runs")


def test_08_1000_row_scalability():
    """8. 1000-row dataset mines quickly and still finds the hidden rule."""
    t0 = time.time()
    df = build_dataset(n_rows=1000)
    res = rm.mine_rules(df, "Y", verbose=False)
    dt = time.time() - t0
    hidden = bl.ast_to_expr(bl.parse_expression("(A AND B) OR (C AND NOT D)"))
    print(f"  1000 rows mined in {dt:.2f}s; "
          f"candidates={res['stats']['n_total']}; top rule: {res['top']['Rule'].iloc[0]}")
    _check(dt < 60, f"mining 1000 rows took too long ({dt:.1f}s)")
    _check(not res["top"].empty, "no rules for the 1000-row dataset")
    _check(hidden in list(res["top"]["Rule"].head(5)),
           "hidden rule missing from top-5 on 1000 rows")


def test_09_simplification_AB_or_A_notB():
    """9. (A AND B) OR (A AND NOT B) must simplify to A."""
    expr = "(A AND B) OR (A AND NOT B)"
    s = bl.simplify_expression(expr)
    print(f"  before : {s['original']}")
    for st in s["steps"]:
        print(f"    {st}")
    print(f"  after  : {s['simplified_algebraic']}  ({s['simplified_plain']})")
    _check(s["simplified_algebraic"] == "A",
           f"expected simplified form 'A', got '{s['simplified_algebraic']}'")
    # equivalence check via truth tables (same variable set!)
    tt1 = bl.generate_truth_table(expr, ["A", "B"])
    tt2 = bl.generate_truth_table("A", ["A", "B"])
    _check(list(tt1["Y"]) == list(tt2["Y"]), "simplified form is not equivalent")


def test_10_truth_table_A_and_NOT_B():
    """10. Truth table of (A AND NOT B): exactly 4 rows with correct values."""
    tt = bl.generate_truth_table("A AND NOT B")
    print(tt.to_string(index=False))
    _check(len(tt) == 4, f"expected 4 rows (2^2), got {len(tt)}")
    expected = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 0}
    for i in range(4):
        key = (int(tt.loc[i, "A"]), int(tt.loc[i, "B"]))
        _check(int(tt.loc[i, "Y"]) == expected[key],
               f"row {key}: expected Y={expected[key]}, got {int(tt.loc[i, 'Y'])}")


# ---------------------------------------------------------------------------
# EDGE / FAULT TESTS (11-15)
# ---------------------------------------------------------------------------

def test_11_empty_csv_graceful_error():
    """11. Empty CSV -> graceful, informative ValueError (no traceback crash)."""
    TMP.mkdir(parents=True, exist_ok=True)
    p = TMP / "empty.csv"
    p.write_text("")
    try:
        rm.load_and_validate_csv(p)
        _check(False, "empty CSV should raise ValueError")
    except ValueError as e:
        print(f"  empty file  -> ValueError: {e}")
    p2 = TMP / "header_only.csv"
    p2.write_text("A,B,Y\n")
    try:
        rm.load_and_validate_csv(p2)
        _check(False, "header-only CSV should raise ValueError")
    except ValueError as e:
        print(f"  header only -> ValueError: {e}")
        _check("no data rows" in str(e).lower(), "error message should mention missing rows")


def test_12_all_zeros_target_no_rules():
    """12. All-zero target -> no rules, informative message."""
    df = build_dataset()
    df["Y"] = 0
    res = rm.mine_rules(df, "Y", verbose=False)
    print(f"  rules found: {len(res['top'])}")
    print(f"  message    : {res['message']}")
    _check(res["top"].empty, "all-zero target should produce no rules")
    _check(res["message"] and "ALWAYS 0" in res["message"],
           "expected the 'target always 0' message")


def test_13_all_ones_target_rule_true():
    """13. All-ones target -> the best rule must be the constant TRUE."""
    df = build_dataset()
    df["Y"] = 1
    res = rm.mine_rules(df, "Y", verbose=False)
    best = res["best"]
    print(f"  best rule: {best['Rule']}  (conf={best['Confidence']}, F1={best['F1']})")
    _check(best is not None, "expected at least one rule for all-ones target")
    _check(best["Rule"] == "TRUE", f"expected rule 'TRUE', got '{best['Rule']}'")
    _check(best["Confidence"] == 1.0 and best["F1"] == 1.0,
           "TRUE rule must have confidence = F1 = 1.0")


def test_14_missing_invalid_values_validator():
    """14. CSV with NaN / 2 / 'yes' -> validator error listing the faults."""
    TMP.mkdir(parents=True, exist_ok=True)
    bad = pd.DataFrame({
        "A": [1, 0, 1, None],
        "B": [1, "yes", 0, 1],
        "C": [1, 2, 0, 1],
        "Y": [1, 0, 1, 0],
    })
    p = TMP / "bad.csv"
    bad.to_csv(p, index=False)
    print(bad.to_string(index=False))
    try:
        rm.load_and_validate_csv(p)
        _check(False, "invalid CSV should raise ValueError")
    except ValueError as e:
        print(f"  ValueError: {e}")
        msg = str(e)
        _check("missing" in msg, "error should mention missing values (column A)")
        _check("yes" in msg, "error should mention the invalid value 'yes' (column B)")
        _check("2" in msg, "error should mention the out-of-range value 2 (column C)")


def test_15_contradictory_data_low_confidence_warning():
    """15. Same feature values with different Y -> low-confidence warning."""
    dfc = pd.DataFrame({
        "A": [1, 1, 1, 1, 0, 0, 0, 0],
        "B": [1, 1, 0, 0, 1, 1, 0, 0],
        "Y": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    # every (A,B) group contains BOTH Y values -> max possible confidence ~ 0.5
    res = rm.mine_rules(dfc, "Y", max_literals=2, min_support=0.05,
                        min_confidence=0.3, top_n=5, verbose=False)
    print("  Top-3 rules:")
    print(res["top"].head(3).to_string(index=False))
    print(f"  warning: {res['warning']}")
    _check(not res["top"].empty, "a low-confidence rule should survive min_confidence=0.3")
    _check(res["warning"] is not None, "expected a low-confidence/contradiction warning")
    _check("contradict" in res["warning"].lower(),
           "warning should mention contradictory data")
    _check(res["top"]["Confidence"].max() < 0.6,
           "confidence must be capped low for contradictory data")


# ---------------------------------------------------------------------------
# STANDALONE RUNNER
# ---------------------------------------------------------------------------

TESTS = [
    ("01 default dataset, target Y, max_literals=2", test_01_default_dataset_targetY_maxL2),
    ("02 max_literals=3 recovers hidden rule", test_02_maxL3_recovers_hidden_rule),
    ("03 target E -> weak rules", test_03_change_target_to_E_weak_rule),
    ("04 min_support=0.2 -> fewer rules", test_04_high_min_support_02),
    ("05 min_confidence=0.5 -> more rules", test_05_low_min_confidence_05),
    ("06 top_n=5 vs 20", test_06_top_n_5_vs_20),
    ("07 seed=123 reproducibility", test_07_custom_seed_123_reproducibility),
    ("08 1000-row scalability", test_08_1000_row_scalability),
    ("09 simplification (A&B)|(A&~B) => A", test_09_simplification_AB_or_A_notB),
    ("10 truth table of (A AND NOT B)", test_10_truth_table_A_and_NOT_B),
    ("11 empty CSV -> graceful error", test_11_empty_csv_graceful_error),
    ("12 all-zeros target -> no rules", test_12_all_zeros_target_no_rules),
    ("13 all-ones target -> rule TRUE", test_13_all_ones_target_rule_true),
    ("14 NaN/2/'yes' -> validator error", test_14_missing_invalid_values_validator),
    ("15 contradictory data -> warning", test_15_contradictory_data_low_confidence_warning),
]


def main() -> int:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True, exist_ok=True)
    results = []
    print("=" * 70)
    print(" Boolean Feature Rule Miner — 15 test cases")
    print("=" * 70)
    for name, fn in TESTS:
        print(f"\n--- TEST {name}")
        try:
            fn()
            results.append((name, True, ""))
            print(f"    PASS")
        except Exception as exc:              # noqa: BLE001
            results.append((name, False, str(exc)))
            print(f"    FAIL: {exc}")
    print("\n" + "=" * 70)
    n_pass = sum(ok for _, ok, _ in results)
    print(f" SUMMARY: {n_pass}/{len(TESTS)} tests passed")
    for name, ok, msg in results:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{msg}]" if msg else ""))
    print("=" * 70)
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if n_pass == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
