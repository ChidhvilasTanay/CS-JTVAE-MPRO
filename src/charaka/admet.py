"""ADMET-AI safety panel wrapper.

Predicts the standard 41-endpoint panel for both candidate analogs and
their parent anchors. For each candidate we compute an
*endpoint-by-endpoint* "safer than parent" flag on the eight clinically
critical endpoints (hERG, AMES, DILI, and five CYP isoforms) so the
downstream code can summarise majority-safer / universally-safer
candidates.
"""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd


SAFETY_ENDPOINTS = (
    "hERG", "AMES", "DILI",
    "CYP3A4_Veith", "CYP2D6_Veith", "CYP2C9_Veith",
    "CYP1A2_Veith", "CYP2C19_Veith",
)


def predict_admet(smiles: Sequence[str]) -> pd.DataFrame:
    """Run ADMET-AI on ``smiles`` and return its raw prediction frame.

    Importing ``admet_ai`` is deferred so the rest of the package does
    not require it.
    """
    from admet_ai import ADMETModel

    model = ADMETModel()
    return model.predict(smiles=list(smiles))


def annotate_safety(
    df: pd.DataFrame,
    cand_column: str = "cand_smi",
    orig_column: str = "orig_smi",
    endpoints: Sequence[str] = SAFETY_ENDPOINTS,
) -> pd.DataFrame:
    """Append ADMET predictions and per-endpoint "safer than orig" flags.

    Adds columns ``admet_<endpoint>``, ``admet_<endpoint>_orig`` and
    ``admet_<endpoint>_safer`` for every endpoint available in the
    panel, plus the summary columns ``admet_safer_count``,
    ``admet_safer_total`` and ``admet_safer_frac``.
    """
    cand_smiles = df[cand_column].tolist()
    cand_pred = predict_admet(cand_smiles)
    for col in cand_pred.columns:
        df[f"admet_{col}"] = cand_pred[col].values

    orig_smiles = df[orig_column].unique().tolist()
    orig_pred = predict_admet(orig_smiles)
    orig_pred.index = orig_smiles

    available: List[str] = [e for e in endpoints if e in cand_pred.columns]
    safer_columns: List[str] = []
    for ep in available:
        orig_map = orig_pred[ep].to_dict()
        df[f"admet_{ep}_orig"] = df[orig_column].map(orig_map)
        flag = f"admet_{ep}_safer"
        df[flag] = (df[f"admet_{ep}"] < df[f"admet_{ep}_orig"]).astype(int)
        safer_columns.append(flag)

    if safer_columns:
        df["admet_safer_count"] = df[safer_columns].sum(axis=1)
        df["admet_safer_total"] = len(safer_columns)
        df["admet_safer_frac"] = df["admet_safer_count"] / len(safer_columns)
    return df
