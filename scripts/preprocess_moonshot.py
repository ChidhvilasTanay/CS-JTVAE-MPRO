"""Standalone preprocessor for the raw COVID Moonshot dump.

Reads ``covid_moonshot_raw.csv`` (the public PostEra release) and writes
``covid_moonshot_processed.csv`` with two columns suitable for the
training pipeline:

* ``SMILES`` -- the raw SMILES string
* ``standard_value`` -- IC50 in nanomolar (NaN where not measured)

Run this once before any of the training scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running this script directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charaka.data import canonicalise  # noqa: E402


SMILES_CANDIDATES = ("SMILES", "smiles", "Canonical_SMILES")
IC50_CANDIDATES = (
    "standard_value", "IC50_nM", "IC50 (uM)", "IC50_uM", "fluorescence_IC50",
)


def _find_column(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def main(raw_path: str, out_path: str) -> int:
    df = pd.read_csv(raw_path)
    smi_col = _find_column(df, SMILES_CANDIDATES)
    ic_col = _find_column(df, IC50_CANDIDATES)

    df["SMILES"] = df[smi_col].apply(canonicalise)
    df = df.dropna(subset=["SMILES"]).copy()

    # Heuristic: column names with "uM" are micromolar; convert to nM.
    if "uM" in ic_col:
        df["standard_value"] = df[ic_col] * 1000
    else:
        df["standard_value"] = df[ic_col]

    df = df.drop_duplicates("SMILES", keep="first").reset_index(drop=True)
    df = df[["SMILES", "standard_value"]]
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/covid_moonshot_raw.csv")
    parser.add_argument("--out", default="data/covid_moonshot_processed.csv")
    args = parser.parse_args()
    raise SystemExit(main(args.raw, args.out))
