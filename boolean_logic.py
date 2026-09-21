"""
boolean_logic.py
================
Core Digital-Logic module for the **Boolean Feature Rule Miner**.

EC2201 Unit I (Digital Fundamentals) concepts implemented from scratch:

    * Basic logic gates          ->  AND(), OR(), NOT()   on 0/1, NumPy, Pandas
    * Boolean expression parser  ->  "(A AND B) OR (C AND NOT D)"  (recursive descent)
    * Expression evaluator       ->  evaluate() on a DataFrame row or whole dataset
    * Truth-table generator      ->  generate_truth_table()   (k <= 4 variables)
    * SOP / POS conversion       ->  to_sop_canonical(), to_pos_canonical()
    * Boolean simplification     ->  simplify_expression()
                                     (Boolean identities + Quine-McCluskey, k <= 4)
    * Gate-level evaluation trace->  evaluate_row() + format_trace()

Boolean conventions (same as the EC2201 textbook):
    * 1 = TRUE,  0 = FALSE
    * AND  = multiplication  (a·b)
    * OR   = addition        (a + b)
    * NOT  = complement      (a')

Everything is written with NumPy / Pandas only — NO sklearn, NO external
Boolean-algebra libraries.  All functions accept plain integers, NumPy
arrays OR Pandas Series, so the same gate can be demoed on one row (viva)
or evaluated on the whole dataset (mining).
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. BASIC LOGIC GATES  (Section 1 of EC2201 Unit I)
# ---------------------------------------------------------------------------

def AND(a, b):
    """2-input AND gate.   out = a · b  (1 only when BOTH inputs are 1).

    Truth table:
         a   b | a AND b
         0   0 |    0
         0   1 |    0
         1   0 |    0
         1   1 |    1

    Works on: Python int/bool, NumPy arrays, Pandas Series (element-wise).
    """
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            return (a.astype(bool) & b.astype(bool)).astype(int)
        arr, other = (a, b) if isinstance(a, pd.Series) else (b, a)
        return (arr.astype(bool) & bool(int(other))).astype(int)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        arr, other = (a, b) if isinstance(a, np.ndarray) else (b, a)
        return (arr.astype(bool) & bool(int(other))).astype(int)
    return int(bool(a) and bool(b))


def OR(a, b):
    """2-input OR gate.   out = a + b  (1 when AT LEAST ONE input is 1).

    Truth table:
         a   b | a OR b
         0   0 |    0
         0   1 |    1
         1   0 |    1
         1   1 |    1

    Works on: Python int/bool, NumPy arrays, Pandas Series (element-wise).
    """
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            return (a.astype(bool) | b.astype(bool)).astype(int)
        arr, other = (a, b) if isinstance(a, pd.Series) else (b, a)
        return (arr.astype(bool) | bool(int(other))).astype(int)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        arr, other = (a, b) if isinstance(a, np.ndarray) else (b, a)
        return (arr.astype(bool) | bool(int(other))).astype(int)
    return int(bool(a) or bool(b))


def NOT(a):
    """NOT gate (inverter).   out = a'  (1 when input is 0, else 0).

    Truth table:
         a | a NOT
         0 |   1
         1 |   0

    Works on: Python int/bool, NumPy arrays, Pandas Series (element-wise).
    """
    if isinstance(a, pd.Series):
        return (1 - a.astype(int)).astype(int)
    if isinstance(a, np.ndarray):
        return (1 - a.astype(int)).astype(int)
    return int(not bool(a))


# ---------------------------------------------------------------------------
# 2. EXPRESSION PARSER  (grammar, recursive-descent)
# ---------------------------------------------------------------------------
#
# Grammar (standard precedence: NOT > AND > OR):
#
#     expr    := term  (OR term)*
#     term    := factor (AND factor)*
#     factor  := NOT factor | primary
#     primary := '(' expr ')' | literal
#     literal := variable | TRUE | FALSE | 0 | 1
#
# Accepted spellings:  AND | &        OR | |        NOT | ~ | !

def _tokenize(expr: str) -> list:
    """Split the rule string into raw tokens (symbols + words)."""
    tokens = []
    i, s = 0, expr.strip()
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
        elif ch in "()&|~!":
            tokens.append(("sym", ch))
            i += 1
        elif ch.isalpha() or ch == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(("word", s[i:j]))
            i = j
        elif ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(("word", s[i:j]))
            i = j
        else:
            raise ValueError(f"Invalid character '{ch}' in rule expression: {expr!r}")
    if not tokens:
        raise ValueError("Empty rule expression")
    return tokens


def _normalize(tokens: list) -> list:
    """Convert raw tokens to a unified token stream used by the parser.

    Operators become the strings 'AND' / 'OR' / 'NOT';
    values become tuples ('const', 0|1) and variables become ('var', name).
    """
    out = []
    for kind, val in tokens:
        if kind == "sym":
            if val == "(":
                out.append("(")
            elif val == ")":
                out.append(")")
            elif val == "&":
                out.append("AND")
            elif val == "|":
                out.append("OR")
            elif val in ("~", "!"):
                out.append("NOT")
        else:
            w = val.upper()
            if w == "AND":
                out.append("AND")
            elif w == "OR":
                out.append("OR")
            elif w == "NOT":
                out.append("NOT")
            elif w == "TRUE":
                out.append(("const", 1))
            elif w == "FALSE":
                out.append(("const", 0))
            elif w in ("0", "1"):
                out.append(("const", int(val)))
            else:
                out.append(("var", val))
    return out


class _Parser:
    """Tiny recursive-descent parser for Boolean expressions."""

    def __init__(self, toks: list):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self):
        node = self._or()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token {self.peek()!r} in expression")
        return node

    def _or(self):
        node = self._and()
        children = [node]
        while self.peek() == "OR":
            self.next()
            children.append(self._and())
        return children[0] if len(children) == 1 else ("or", tuple(children))

    def _and(self):
        node = self._factor()
        children = [node]
        while self.peek() == "AND":
            self.next()
            children.append(self._factor())
        return children[0] if len(children) == 1 else ("and", tuple(children))

    def _factor(self):
        if self.peek() == "NOT":
            self.next()
            return ("not", self._factor())
        return self._primary()

    def _primary(self):
        tok = self.next()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if tok == "(":
            node = self._or()
            if self.next() != ")":
                raise ValueError("Missing closing ')' in expression")
            return node
        if isinstance(tok, tuple):
            return tok            # ('const', v) or ('var', name)
        raise ValueError(f"Unexpected token {tok!r} in expression")


def parse_expression(expr: str):
    """Parse a rule string into an AST.

    AST nodes:  ('const', 0|1) | ('var', name) | ('not', node)
                | ('and', (node, ...)) | ('or', (node, ...))
    """
    return _Parser(_normalize(_tokenize(expr))).parse()


def expression_variables(expr: str) -> List[str]:
    """Return the variable names in an expression (order of appearance)."""
    tree = parse_expression(expr)
    order: List[str] = []

    def walk(node):
        if node[0] == "var":
            if node[1] not in order:
                order.append(node[1])
        elif node[0] in ("and", "or"):
            for c in node[1]:
                walk(c)
        elif node[0] == "not":
            walk(node[1])

    walk(tree)
    return order


def ast_to_expr(node) -> str:
    """Convert an AST back to a readable rule string, e.g.
    '(A AND B) OR (C AND NOT D)'.  Used for canonical display / dedup."""
    t = node[0]
    if t == "const":
        return "TRUE" if node[1] else "FALSE"
    if t == "var":
        return node[1]
    if t == "not":
        inner = ast_to_expr(node[1])
        if node[1][0] in ("var", "const"):          # simple operand: NOT A
            return "NOT " + inner
        return "NOT (" + inner + ")"               # compound: NOT (A AND B)
    op = " AND " if t == "and" else " OR "
    parts = []
    for c in node[1]:
        s = ast_to_expr(c)
        if c[0] in ("and", "or") and c[0] != t:     # wrap other operator
            s = f"({s})"
        parts.append(s)
    return op.join(parts)


# ---------------------------------------------------------------------------
# 3. EXPRESSION EVALUATOR  (+ optional gate-level trace)
# ---------------------------------------------------------------------------

def _eval_node(node, lookup, trace=None):
    """Recursively evaluate an AST node.

    lookup(name) must return the current 0/1 value of the variable — either
    an int (single row) or a Pandas Series (whole dataset).

    Returns (value, label).  If `trace` (a list) is given, every binary
    gate application is appended as (gate, [(label, value), ...], result).
    """
    t = node[0]
    if t == "const":
        return node[1], ("TRUE" if node[1] else "FALSE")
    if t == "var":
        return lookup(node[1]), node[1]
    if t == "not":
        val, lab = _eval_node(node[1], lookup, trace)
        out = NOT(val)
        if trace is not None:
            trace.append(("NOT", [(lab, val)], out))
        return out, (f"NOT {lab}" if node[1][0] not in ("and", "or")
                     else f"NOT ({lab})")
    gate = "AND" if t == "and" else "OR"
    acc = None
    acc_lab = None
    for c in node[1]:
        val, lab = _eval_node(c, lookup, trace)
        if acc is None:
            acc, acc_lab = val, lab
        else:
            prev = acc                        # value entering THIS gate
            acc = AND(acc, val) if t == "and" else OR(acc, val)
            if trace is not None:
                trace.append((gate, [(acc_lab, prev), (lab, val)], acc))
            acc_lab = f"({acc_lab} {gate} {lab})"
    return acc, acc_lab


def evaluate(expression, data):
    """Evaluate a rule expression.

    data = pandas.DataFrame  -> returns a Pandas Series of 0/1 (one per row)
    data = dict (one row)    -> returns int 0/1

    Example:
        evaluate("(A AND B) OR (C AND NOT D)", df)
    """
    tree = parse_expression(expression)
    if isinstance(data, pd.DataFrame):
        missing = set(expression_variables(expression)) - set(data.columns)
        if missing:
            raise KeyError(f"Expression references columns not in DataFrame: {sorted(missing)}")
        return _eval_node(tree, lambda name: data[name].astype(int))[0]
    if isinstance(data, dict):
        missing = set(expression_variables(expression)) - set(data.keys())
        if missing:
            raise KeyError(f"Expression references variables not in row: {sorted(missing)}")
        return _eval_node(tree, lambda name: int(data[name]))[0]
    raise TypeError("data must be a pandas.DataFrame or a dict of 0/1 values")


def evaluate_row(expression, row: dict, trace: list | None = None):
    """Evaluate a rule on ONE row and (optionally) record the gate trace.

    Returns (output 0/1, trace).  Each trace step is
        (gate_name, [(input_label, input_value), ...], output_value)
    which is exactly what a student would draw on a gate-level timing diagram.
    """
    if trace is None:
        trace = []
    tree = parse_expression(expression)
    missing = set(expression_variables(expression)) - set(row.keys())
    if missing:
        raise KeyError(f"Row {row} is missing variables: {sorted(missing)}")
    val, _ = _eval_node(tree, lambda name: int(row[name]), trace)
    return val, trace


def format_trace(trace: list) -> str:
    """Pretty-print a gate-level trace produced by evaluate_row()."""
    lines = []
    for i, (gate, inputs, out) in enumerate(trace, 1):
        ins = ", ".join(f"{lab}={int(v)}" for lab, v in inputs)
        lines.append(f"  Step {i} | Gate: {gate:<3} | inputs: {ins:<34} | output: {int(out)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. TRUTH-TABLE GENERATOR  (k <= 4 variables)
# ---------------------------------------------------------------------------

def generate_truth_table(expression: str, var_order: list | None = None,
                         max_vars: int = 4) -> pd.DataFrame:
    """Generate the full 2^k truth table of a rule as a DataFrame.

    Columns = the k input variables (in `var_order` if given) + output column 'Y'.
    Rows are the 2^k input combinations in binary counting order.
    """
    if var_order is None:
        var_order = expression_variables(expression)
    k = len(var_order)
    if k > max_vars:
        raise ValueError(f"Truth table supports up to {max_vars} variables, got {k}")
    rows = []
    for combo in itertools.product([0, 1], repeat=k):
        row = dict(zip(var_order, combo))
        y = evaluate(expression, row)
        rec = {v: row[v] for v in var_order}
        rec["Y"] = y
        rows.append(rec)
    cols = list(var_order) + ["Y"]
    if k == 0:                        # constant rule (TRUE / FALSE)
        return pd.DataFrame([{"Y": evaluate(expression, {})}])
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# 5. SOP / POS CONVERSION  (canonical forms from the truth table)
# ---------------------------------------------------------------------------

def _minterms_of(expression: str, var_order: list) -> Tuple[list, int]:
    tt = generate_truth_table(expression, var_order)
    minterms = []
    for i in range(len(tt)):
        if int(tt.loc[i, "Y"]) == 1:
            bits = "".join(str(int(tt.loc[i, v])) for v in var_order)
            minterms.append(int(bits, 2))
    return sorted(minterms), len(var_order)


def _minterm_algebraic(m: int, vars_: list) -> str:
    """One minterm as an algebraic product term, e.g. A'·B·C·D'."""
    bits = f"{m:0{len(vars_)}b}"
    return "·".join(v if b == "1" else v + "'" for v, b in zip(vars_, bits))


def _minterm_plain(m: int, vars_: list) -> str:
    """One minterm in AND/OR/NOT words, e.g. NOT A AND B AND NOT C AND D."""
    bits = f"{m:0{len(vars_)}b}"
    return " AND ".join(v if b == "1" else f"NOT {v}" for v, b in zip(vars_, bits))


def to_sop_canonical(expression: str, var_order: list | None = None) -> dict:
    """Convert a rule to canonical Sum-of-Products (minterm) form.

    Returns a dict with:
        minterms        : list of minterm indices, e.g. [2, 6, 10, 12]
        sum_notation    : 'Σm(2, 6, 10, 12)'
        algebraic       : "A'·B'·C·D' + A'·B·C·D' + ..."
        plain           : "(NOT A AND NOT B AND C AND NOT D) OR ..."
    (or the constant 0 / FALSE when the rule is never 1)
    """
    if var_order is None:
        var_order = expression_variables(expression)
    minterms, k = _minterms_of(expression, var_order)
    if not minterms:
        return {"minterms": [], "sum_notation": "Σm( )",
                "algebraic": "0", "plain": "FALSE"}
    terms = [_minterm_algebraic(m, var_order) for m in minterms]
    plains = [_minterm_plain(m, var_order) for m in minterms]
    return {
        "minterms": minterms,
        "sum_notation": "Σm(" + ", ".join(map(str, minterms)) + ")",
        "algebraic": " + ".join(terms),
        "plain": " OR ".join(f"({p})" for p in plains),
    }


def to_pos_canonical(expression: str, var_order: list | None = None) -> dict:
    """Convert a rule to canonical Product-of-Sums (maxterm) form.

    Maxterm for a row where F=0: OR of (variable if bit is 0, variable' if 1).
    Returns a dict with:
        maxterms        : list of maxterm indices, e.g. [0, 1, 3]
        product_notation: 'ΠM(0, 1, 3)'
        algebraic       : "(A + B + C' + D)·(A + B + C + D)·..."
        plain           : "(A OR NOT B OR C OR D) AND ..."
    (or the constant 1 / TRUE when the rule is always 1)
    """
    if var_order is None:
        var_order = expression_variables(expression)
    minterms, k = _minterms_of(expression, var_order)
    maxterms = [m for m in range(2 ** k) if m not in minterms]
    if not maxterms:
        return {"maxterms": [], "product_notation": "ΠM( )",
                "algebraic": "1", "plain": "TRUE"}
    terms, plains = [], []
    for m in maxterms:
        bits = f"{m:0{k}b}"
        terms.append("(" + " + ".join(v if b == "0" else v + "'" for v, b in zip(var_order, bits)) + ")")
        plains.append("(" + " OR ".join(v if b == "0" else f"NOT {v}" for v, b in zip(var_order, bits)) + ")")
    return {
        "maxterms": maxterms,
        "product_notation": "ΠM(" + ", ".join(map(str, maxterms)) + ")",
        "algebraic": "·".join(terms),
        "plain": " AND ".join(plains),
    }


# ---------------------------------------------------------------------------
# 6. BOOLEAN SIMPLIFICATION
#    6a. quick pass with Boolean identities (AST rewrite)
#    6b. exact minimization with the Quine-McCluskey method (k <= 4)
# ---------------------------------------------------------------------------

def _normalize_ast(node):
    """Apply basic Boolean identities to an AST (one pass):

        X + X = X ,  X·X = X           (idempotent law)
        X + X' = 1 ,  X·X' = 0         (complement law)
        0 + X = X ,  0·X = 0           (domination / annulment)
        1·X = X ,  1 + X = 1
        X + X·Y = X                    (absorption law)
    """
    t = node[0]
    if t not in ("and", "or"):
        return node
    kids = [_normalize_ast(c) for c in node[1]]
    # idempotent law: drop duplicate sub-expressions
    seen, uniq = set(), []
    for kk in kids:
        key = ast_to_expr(kk)
        if key not in seen:
            seen.add(key)
            uniq.append(kk)
    kids = uniq
    if t == "and":
        # complement law X·X' = 0  (or the same literal twice — already deduped)
        sigs = set()
        for kk in kids:
            lit = kk[1] if kk[0] == "not" else kk
            if lit[0] == "var":
                if lit[1] in sigs:
                    return ("const", 0)
                sigs.add(lit[1])
        # 0·X = 0,  1·X = X
        if any(kk[0] == "const" and kk[1] == 0 for kk in kids):
            return ("const", 0)
        kids = [kk for kk in kids if not (kk[0] == "const" and kk[1] == 1)]
        return kids[0] if len(kids) == 1 else (("and", tuple(kids)) if kids else ("const", 1))
    # t == "or"
    sigs = set()
    for kk in kids:
        lit = kk[1] if kk[0] == "not" else kk
        if lit[0] == "var":
            if lit[1] in sigs:
                return ("const", 1)          # X + X' = 1
            sigs.add(lit[1])
    if any(kk[0] == "const" and kk[1] == 1 for kk in kids):
        return ("const", 1)                  # 1 + X = 1
    kids = [kk for kk in kids if not (kk[0] == "const" and kk[1] == 0)]
    # absorption law:  X + X·Y = X  (drop AND-terms that contain literal X)
    pos_vars = {kk[1] for kk in kids if kk[0] == "var"}
    if pos_vars:
        kids = [kk for kk in kids
                if not (kk[0] == "and"
                        and {c[1] for c in kk[1] if c[0] == "var"} & pos_vars)]
    return kids[0] if len(kids) == 1 else (("or", tuple(kids)) if kids else ("const", 0))


def simplify_by_identities(expression: str) -> dict:
    """Fast simplification using textbook Boolean identities only.

    Returns {original, simplified, changed, applied_rules}.
    This is an educational quick pass — the authoritative minimal form comes
    from Quine-McCluskey (simplify_expression).
    """
    tree = parse_expression(expression)
    original = ast_to_expr(tree)
    node = tree
    for _ in range(10):                      # fixed-point (few rules needed)
        new_node = _normalize_ast(node)
        if ast_to_expr(new_node) == ast_to_expr(node):
            node = new_node
            break
        node = new_node
    simplified = ast_to_expr(node)
    changed = simplified != original
    return {
        "original": original,
        "simplified": simplified,
        "changed": changed,
        "applied_rules": ["idempotent / complement / dominance / absorption"] if changed
                         else ["no identity applied (form already simple)"],
    }


def _quine_mccluskey(minterms: list, n: int) -> Tuple[list, list]:
    """Quine-McCluskey minimization (up to 4 variables).

    Steps (as in the EC2201 textbook):
      1. Write each minterm as an n-bit binary string.
      2. Combine strings that differ in exactly ONE bit position
         (same dash positions) -> replace that bit by '-'.
      3. Repeat until no further combination is possible.
      4. Implicants never combined in a round are PRIME IMPlicants.

    Returns (prime_implicants as bit-patterns, steps as human-readable lines).
    """
    steps = []
    current = [f"{m:0{n}b}" for m in sorted(set(minterms))]
    primes = []
    rnd = 0
    while True:
        rnd += 1
        combined: dict = {}
        used: set = set()
        combos = []
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                a, b = current[i], current[j]
                if a.count("-") != b.count("-"):
                    continue
                if [p for p, c in enumerate(a) if c == "-"] != [p for p, c in enumerate(b) if c == "-"]:
                    continue
                diffs = [p for p in range(n) if a[p] != b[p]]
                if len(diffs) == 1:
                    pos = diffs[0]
                    pat = a[:pos] + "-" + a[pos + 1:]
                    combined[pat] = True
                    used.add(a)
                    used.add(b)
                    combos.append(f"{a} + {b} -> {pat}")
        if combined:
            steps.append(f"Round {rnd}: " + ";  ".join(combos))
        else:
            steps.append(f"Round {rnd}: no further combinations — remaining implicants are prime")
        for s in current:
            if s not in used:
                primes.append(s)
        if not combined:
            break
        current = list(combined.keys())
    return primes, steps


def _pattern_covers(pattern: str, m: int, n: int) -> bool:
    """Does a bit pattern (with '-') cover minterm m?"""
    bits = f"{m:0{n}b}"
    return all(c == "-" or c == bits[i] for i, c in enumerate(pattern))


def _select_cover(minterms: list, primes: list, n: int) -> list:
    """Choose the minimal set of prime implicants (Petrick's method).

    1. Prime implicants covering a single minterm are ESSENTIAL.
    2. For the minterms still uncovered, Petrick's product-of-sums enumerates
       every possible cover; we pick the one with the fewest (and fewest
       literal) implicants.  For n <= 4 this enumeration is tiny.
    Falls back to a greedy cover if the enumeration ever gets large.
    """
    def covers(pat, m):
        return _pattern_covers(pat, m, n)

    table = {m: [i for i, p in enumerate(primes) if covers(p, m)] for m in minterms}
    essential: set = set()
    for m, pis in table.items():
        if len(pis) == 1:
            essential.add(pis[0])
    covered = set()
    for e in essential:
        covered |= {m for m in minterms if covers(primes[e], m)}
    remaining = [m for m in minterms if m not in covered]
    if remaining:
        clauses = [tuple(i for i, p in enumerate(primes) if covers(p, m)) for m in remaining]
        combos = [()]
        for cl in clauses:
            combos = [c + (i,) for c in combos for i in cl]
            if len(combos) > 100000:          # safety net (cannot happen for n<=4)
                chosen = set(essential)
                rem = set(remaining)
                while rem:
                    best_i = max(range(len(primes)),
                                 key=lambda i: sum(1 for m in rem if covers(primes[i], m)))
                    chosen.add(best_i)
                    rem = {m for m in rem if not covers(primes[best_i], m)}
                return sorted(chosen)
        best = None
        for c in combos:
            s = frozenset(c)
            if best is None or len(s) < len(best) or (len(s) == len(best) and sorted(s) < sorted(best)):
                best = s
        essential |= best
    return sorted(essential)


def _pattern_to_term(pattern: str, var_order: list) -> str:
    """Bit pattern -> algebraic product term, e.g. '1-01' -> A·C'·D'."""
    parts = []
    for v, c in zip(var_order, pattern):
        if c == "1":
            parts.append(v)
        elif c == "0":
            parts.append(v + "'")
    return "·".join(parts) if parts else "1"


def _pattern_to_plain(pattern: str, var_order: list) -> str:
    """Bit pattern -> product term in AND/NOT words, e.g. '1-01' -> A AND NOT C AND D'."""
    parts = []
    for v, c in zip(var_order, pattern):
        if c == "1":
            parts.append(v)
        elif c == "0":
            parts.append(f"NOT {v}")
    return " AND ".join(parts) if parts else "TRUE"


def _count_literals(node) -> int:
    """Count literal occurrences in an AST (used for 'before/after' summary)."""
    t = node[0]
    if t == "var":
        return 1
    if t == "const":
        return 0
    if t == "not":
        return _count_literals(node[1])
    return sum(_count_literals(c) for c in node[1])


def simplify_expression(expression: str, var_order: list | None = None) -> dict:
    """Simplify a rule using Boolean identities + Quine-McCluskey (k <= 4).

    Returns a dict:
        original            : canonical rule string (before)
        minterms            : minterm indices (truth table of F=1)
        prime_implicants    : bit patterns, e.g. ['11--', '--10']
        essential           : selected (minimal) prime implicants
        simplified_algebraic: minimal SOP in algebraic notation, e.g. "A·B + C·D'"
        simplified_plain    : minimal SOP in AND/OR/NOT words
        steps               : human-readable Q-M combination steps
        literal_before / literal_after : literal counts (show the reduction)
        already_minimal     : True if the rule was already in minimal SOP form
    """
    tree = parse_expression(expression)
    original = ast_to_expr(tree)
    if var_order is None:
        var_order = expression_variables(expression)
    k = len(var_order)
    if k > 4:
        raise ValueError("Quine-McCluskey here is implemented for up to 4 variables")

    minterms, _ = _minterms_of(expression, var_order)
    if not minterms:
        return {"original": original, "minterms": [], "prime_implicants": [],
                "essential": [], "simplified_algebraic": "0",
                "simplified_plain": "FALSE", "steps": ["F is FALSE for every input combination"],
                "literal_before": _count_literals(tree), "literal_after": 0,
                "already_minimal": True}
    if len(minterms) == 2 ** k:
        return {"original": original, "minterms": minterms,
                "prime_implicants": ["-" * k], "essential": [0],
                "simplified_algebraic": "1", "simplified_plain": "TRUE",
                "steps": ["F is TRUE for every input combination"],
                "literal_before": _count_literals(tree), "literal_after": 0,
                "already_minimal": True}

    primes, steps = _quine_mccluskey(minterms, k)
    steps.insert(0, "Minterms (F=1): " + ", ".join(
        f"{m}({f'{m:0{k}b}'})" for m in minterms))
    essential = _select_cover(minterms, primes, k)
    sel = [primes[i] for i in essential]
    terms = [_pattern_to_term(p, var_order) for p in sel]
    plains = [_pattern_to_plain(p, var_order) for p in sel]
    simplified_alg = " + ".join(terms)
    simplified_plain = " OR ".join(f"({p})" if " AND " in p else p for p in plains)
    lit_before = _count_literals(tree)
    lit_after = sum(len(p) - p.count("-") for p in sel)
    return {
        "original": original,
        "minterms": minterms,
        "prime_implicants": primes,
        "essential": sel,
        "simplified_algebraic": simplified_alg,
        "simplified_plain": simplified_plain,
        "steps": steps,
        "literal_before": lit_before,
        "literal_after": lit_after,
        "already_minimal": lit_after >= lit_before,
    }
