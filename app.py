"""
app.py
======
Optional Streamlit web UI for the Boolean Feature Rule Miner.

Run:
    streamlit run app.py

Features:
    * upload your own binary CSV (0/1 only) or load the built-in demo dataset
    * select the target column
    * sliders: max_literals, min_support, min_confidence, top_n
    * ranked rule table + low-confidence / contradiction warning
    * expandable deep-dive per rule: truth table, SOP/POS,
      Q-M simplification before/after, gate-level evaluation trace
    * charts: Support vs Confidence bar chart, Coverage vs Confidence scatter,
      truth-table heatmap
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import streamlit as st

import boolean_logic as bl
import rule_miner as rm
from data_generator import build_dataset

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Boolean Feature Rule Miner",
                   page_icon="🔢", layout="wide")
st.title("🔢 Boolean Feature Rule Miner")
st.caption("EC2201 Unit I — Digital Fundamentals: mine human-readable "
           "IF-THEN Boolean rules (AND/OR/NOT) from a binary dataset, "
           "ranked by Support / Confidence / F1, with truth tables, "
           "SOP/POS and Quine-McCluskey simplification.")

# ---------------------------------------------------------------------------
# DATA SOURCE
# ---------------------------------------------------------------------------
st.subheader("1 · Data source")
col_a, col_b = st.columns([3, 1])
with col_a:
    uploaded = st.file_uploader("Upload a binary CSV (only 0/1 values)", type=["csv"])
with col_b:
    if st.button("Load demo dataset (seed=42)"):
        st.session_state["df"] = build_dataset()

df = None
if uploaded is not None:
    try:
        df = rm.load_and_validate_csv(StringIO(uploaded.getvalue().decode("utf-8")))
        st.session_state["df"] = df
        st.success(f"Uploaded file validated: {len(df)} rows × {len(df.columns)} binary columns.")
    except ValueError as e:
        st.error(f"❌ Invalid binary dataset:\n\n{e}")
elif "df" in st.session_state:
    df = st.session_state["df"]

if df is None:
    st.info("👈 Upload a CSV or click **Load demo dataset** to get started.")
    st.stop()

st.dataframe(df.head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# MINER PARAMETERS
# ---------------------------------------------------------------------------
st.subheader("2 · Target & thresholds")
c1, c2, c3, c4 = st.columns(4)
target = c1.selectbox("Target column", list(df.columns),
                      index=list(df.columns).index("Y") if "Y" in df.columns else -1)
max_literals = c2.slider("Max literals per AND-term", 1, 4, 3)
min_support = c3.slider("Min support P(R & Y=1)", 0.01, 0.50, 0.05, 0.01)
min_confidence = c4.slider("Min confidence P(Y=1|R)", 0.0, 1.0, 0.60, 0.05)
top_n = st.slider("Top-N rules to display", 1, 30, 10)

st.subheader("3 · Ranked rules")
res = rm.mine_rules(df, target, max_literals, min_support, min_confidence, top_n, verbose=False)

if res["top"].empty:
    st.warning(f"No rules satisfy the thresholds — {res['message']}")
else:
    st.success(f"{len(res['rules'])} rules passed the filters. "
               f"Best: **IF {res['best']['Rule']} THEN Y=1** "
               f"(F1={res['best']['F1']:.3f}, Conf={res['best']['Confidence']:.3f})")
    st.dataframe(rm.format_rules_frame(res["top"]), use_container_width=True)

if res.get("warning"):
    st.warning(res["warning"])

# ---------------------------------------------------------------------------
# DEEP DIVE (expandable per top rule)
# ---------------------------------------------------------------------------
st.subheader("4 · Deep dive (truth table · SOP/POS · simplification · gate trace)")
if not res["top"].empty:
    pick = st.selectbox("Rule to inspect",
                        list(res["top"]["Rule"]),
                        format_func=lambda r: f"{r}")
    expr = pick
    series = res["best"]["series"] if expr == res["best"]["Rule"] else None
    if series is None:                     # re-evaluate the chosen rule
        series = bl.evaluate(expr, df)
    var_order = bl.expression_variables(expr) if expr not in ("TRUE", "FALSE") else \
        [c for c in df.columns if c != target][:4]

    t1, t2 = st.columns(2)
    with t1:
        tt = bl.generate_truth_table(expr, var_order)
        st.markdown(f"**Truth table** (k={len(tt.columns) - 1}, {len(tt)} rows)")
        st.dataframe(tt, use_container_width=True)
    with t2:
        sop = bl.to_sop_canonical(expr, var_order or None)
        pos = bl.to_pos_canonical(expr, var_order or None)
        st.markdown(f"**SOP**: `{sop['sum_notation']}`\n\n"
                    f"`F = {sop['algebraic'][:140]}`\n\n"
                    f"**POS**: `{pos['product_notation']}`\n\n"
                    f"`F = {pos['algebraic'][:140]}`")

    s = bl.simplify_expression(expr, var_order or None)
    st.markdown(f"**Simplification (Quine-McCluskey)**: "
                f"`{s['original']}`  →  **`{s['simplified_algebraic']}`** "
                f"({s['literal_before']} → {s['literal_after']} literals)")
    with st.expander("Q-M combination steps"):
        for line in s["steps"]:
            st.code(line)

    if series is not None and int(series.sum()) > 0:
        idx = int(series[series == 1].index[0])
        row = {f: int(df.loc[idx, f]) for f in df.columns if f != target}
        val, trace = bl.evaluate_row(expr, row)
        st.markdown(f"**Gate-level trace** — row {idx}: `{row}` → F={val}")
        st.code(bl.format_trace(trace))

# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
st.subheader("5 · Visualisation")
if not res["top"].empty:
    import main as m
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(m.plot_top_rules_bar(res["top"]))
    with c2:
        st.pyplot(m.plot_coverage_scatter(res["top"]))
    var_order = bl.expression_variables(res["best"]["Rule"]) if res["best"] else None
    st.pyplot(m.plot_truth_table_heatmap(res["best"]["Rule"], var_order=var_order or None))

st.caption("Boolean Feature Rule Miner — EC2201 Unit I project prototype. "
           "All logic implemented from scratch (no sklearn).")
