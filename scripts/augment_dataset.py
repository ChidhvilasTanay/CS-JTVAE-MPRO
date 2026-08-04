"""Assemble the v3 augmented training dataset.

ADDITIVE and v3-ONLY. Output = frozen Moonshot (read-only) + the selected covalent
set, written to data/covid_moonshot_augmented_v3.csv in the exact (SMILES,
standard_value) schema the JT-VAE trainer reads. The covalent rows carry no label
(standard_value = NaN), so they augment only the encoder/decoder/vocab training, the
labelled bioactivity subset is unchanged.

Usage:  python scripts/augment_dataset.py            # peptidomimetic-only (default)
        python scripts/augment_dataset.py --mode all # all standardised covalent
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Input files this stage must never overwrite.
_PROTECTED = {"covid_moonshot_processed.csv", "covid_moonshot_raw.csv"}


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m else None


def main(moonshot, covalent, out, mode):
    out = Path(out)
    if out.name in _PROTECTED:
        raise SystemExit(f"refusing to overwrite input file: {out.name}")
    out.parent.mkdir(parents=True, exist_ok=True)

    ms = pd.read_csv(moonshot)
    base_cols = ["SMILES", "standard_value"]
    if list(ms.columns[:2]) != base_cols and not set(base_cols).issubset(ms.columns):
        raise SystemExit(f"unexpected Moonshot columns: {list(ms.columns)}")
    print(f"Moonshot base: {len(ms)} rows  (labelled: {ms['standard_value'].notna().sum()})")

    cov = pd.read_csv(covalent)
    if mode == "peptidomimetic":
        cov = cov[cov["peptidomimetic"] == True]  # noqa: E712
    print(f"covalent to add ({mode}): {len(cov)}")

    # Preserve Moonshot exactly (rows, labels, and its own dups).
    # Dedup only the covalent additions: internally, and against Moonshot's
    # canonical set, so we never drop or alter a Moonshot row or its label.
    ms_canon = set(ms["SMILES"].apply(canon).dropna())
    cov["_c"] = cov["smiles_std"].apply(canon)
    cov = cov[cov["_c"].notna()].copy()
    n_pre = len(cov)
    cov = cov.drop_duplicates("_c")
    cov = cov[~cov["_c"].isin(ms_canon)]
    print(f"covalent after dedup (self + vs Moonshot): {len(cov)}  (removed {n_pre - len(cov)})")

    add = pd.DataFrame({"SMILES": cov["smiles_std"].values, "standard_value": np.nan})
    merged = pd.concat([ms[base_cols], add], ignore_index=True)
    merged.to_csv(out, index=False)

    print(f"\naugmented dataset: {len(merged)} rows -> {out}")
    print(f"  Moonshot rows kept unchanged: {len(ms)}  | labelled: {merged['standard_value'].notna().sum()}")
    print(f"  new covalent structures added: {len(cov)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--moonshot", default="data/covid_moonshot_processed.csv")
    ap.add_argument("--covalent", default="data/chembl_covalent_standardized_v3.csv")
    ap.add_argument("--out", default="data/covid_moonshot_augmented_v3.csv")
    ap.add_argument("--mode", choices=["peptidomimetic", "all"], default="peptidomimetic")
    a = ap.parse_args()
    raise SystemExit(main(a.moonshot, a.covalent, a.out, a.mode))
