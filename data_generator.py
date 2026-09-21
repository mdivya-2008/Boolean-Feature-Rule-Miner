"""
data_generator.py
=================
Reproducible synthetic binary dataset generator.

Ground truth (hidden rule), deliberately chosen from EC2201 Unit I concepts:

        Y = (A AND B) OR (C AND NOT D)

* A, B, C, D are the "meaningful" features (4 bits -> 16-combination truth
  table, exactly the classic K-map / Quine-McCluskey case study).
* E, F (if present) are pure noise features — they must NOT appear in a good
  rule, which makes the miner's ranking meaningful.
* `noise` fraction of Y labels are randomly flipped (label noise), so the
  mined rules can never be perfect — the miner must still RECOVER the hidden
  rule at the top of the ranking.

CLI usage:
    python data_generator.py                                  # 500 rows, seed 42
    python data_generator.py --n_rows 1000 --n_features 6 \
        --noise 0.05 --seed 123 --output data/synthetic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

# Maximum 26 binary features so we can name them A, B, C, ...
MAX_FEATURES = 26


def build_dataset(n_rows: int = 500,
                  n_features: int = 6,
                  noise: float = 0.05,
                  seed: int = 42) -> pd.DataFrame:
    """Build the synthetic binary dataset in memory (reproducible via seed).

    Parameters
    ----------
    n_rows     : number of data rows (default 500)
    n_features : number of binary features A..(A+n-1) (default 6, max 26)
    noise      : fraction of target labels randomly flipped (default 0.05 = 5%)
    seed       : RNG seed for full reproducibility (default 42)

    Returns
    -------
    pandas.DataFrame with binary columns [A, B, ..., F] + target column 'Y'.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if not (1 <= n_features <= MAX_FEATURES):
        raise ValueError(f"n_features must be in 1..{MAX_FEATURES}, got {n_features}")
    if not (0.0 <= noise < 1.0):
        raise ValueError(f"noise must be in [0, 1), got {noise}")

    feature_names = [chr(ord("A") + i) for i in range(n_features)]
    rng = np.random.default_rng(seed)

    # 1) uniform random binary features (each 0/1 with p = 0.5)
    X = rng.integers(0, 2, size=(n_rows, n_features))

    # 2) hidden ground-truth rule (digital logic!):
    #        Y = (A AND B) OR (C AND NOT D)
    #    written directly with gate operations on 0/1 numpy arrays
    A, B, C, D = (X[:, i] for i in range(4))
    Y = ((A & B) | (C & (1 - D))).astype(int)

    # 3) label noise: flip `noise` fraction of the Y values at random
    if noise > 0:
        flip = rng.random(n_rows) < noise
        Y = np.where(flip, 1 - Y, Y).astype(int)

    df = pd.DataFrame(X, columns=feature_names)
    df["Y"] = Y
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a synthetic binary dataset with hidden rule "
                    "Y = (A AND B) OR (C AND NOT D) plus label noise.")
    p.add_argument("--n_rows", type=int, default=500, help="number of rows (default 500)")
    p.add_argument("--n_features", type=int, default=6,
                   help="number of binary features A..F (default 6, max 26)")
    p.add_argument("--noise", type=float, default=0.05,
                   help="label-noise fraction 0..1 (default 0.05 = 5%%)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    p.add_argument("--output", type=str,
                   default=str(PROJECT_ROOT / "data" / "synthetic.csv"),
                   help="output CSV path (default data/synthetic.csv)")
    args = p.parse_args()

    df = build_dataset(args.n_rows, args.n_features, args.noise, args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    pos = int(df["Y"].sum())
    print("=" * 62)
    print(" Synthetic binary dataset generated (reproducible)")
    print("=" * 62)
    print(f"  Rows              : {len(df)}")
    print(f"  Feature columns   : {', '.join(df.columns[:-1])}  (+ target 'Y')")
    print(f"  Hidden rule       : Y = (A AND B) OR (C AND NOT D)")
    print(f"  Label noise       : {args.noise:.0%}  (seed = {args.seed})")
    print(f"  Positive targets  : {pos}/{len(df)}  ({pos / len(df):.1%})")
    print(f"  Saved to          : {out}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
