"""Data loading and canonicalisation utilities."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split


ZINC250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)


def canonicalise(smiles: Optional[str]) -> Optional[str]:
    """Return RDKit's canonical isomeric SMILES, or ``None`` if unparseable."""
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def load_moonshot(
    csv_path: str,
    smiles_column: str = "SMILES",
    value_column: str = "standard_value",
) -> pd.DataFrame:
    """Load and canonicalise the COVID Moonshot dataset.

    ``standard_value`` is assumed to be IC50 in nanomolar; it is
    converted to ``log10(IC50_nM)`` in a new ``value`` column.
    Duplicate canonical SMILES are deduplicated to the most-potent row.
    """
    df = pd.read_csv(csv_path)
    df["canonical"] = df[smiles_column].apply(canonicalise)
    df = df.dropna(subset=["canonical"]).copy()
    df["value"] = df[value_column].apply(
        lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan
    )
    df = df.sort_values("value", na_position="last")
    df = df.drop_duplicates("canonical", keep="first").reset_index(drop=True)
    return df


def load_zinc250k(csv_path: str = "zinc250k.csv") -> pd.DataFrame:
    """Download (once) and canonicalise the ZINC-250K SMILES set."""
    path = Path(csv_path)
    if not path.exists():
        urllib.request.urlretrieve(ZINC250K_URL, str(path))
    df = pd.read_csv(path)
    df["canonical"] = df["smiles"].astype(str).str.strip().apply(canonicalise)
    return df.dropna(subset=["canonical"]).reset_index(drop=True)


def split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/val split. Always split *before* any augmentation."""
    train, val = train_test_split(df, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True)
