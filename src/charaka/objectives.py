"""Scoring functions used as NSGA-II objectives.

Four scalars per candidate:

* :func:`qed_no_alerts` -- QED without the ALERTS term (which penalises
  covalent warheads).
* :func:`sa_norm` -- normalised inverse synthetic-accessibility score.
* :func:`warhead_score` -- fraction of the parent's covalent warheads
  preserved in the candidate.
* :func:`potency_norm` -- logistic squash of the bioactivity surrogate's
  ``log10(IC50)`` prediction.

The post-hoc ranking composite is::

    C = 0.50 * qed_na + 0.15 * sa_norm + 0.20 * warhead + 0.15 * potency
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Set

from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import QED as _QED
from rdkit.Chem import RDConfig

from . import catalogues as _catalogues


_SA_AVAILABLE = False
try:
    _sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if _sa_path not in sys.path:
        sys.path.append(_sa_path)
    import sascorer  # noqa: E402
    _SA_AVAILABLE = True
except Exception:
    sascorer = None  # type: ignore[assignment]


def _qed_weights_without_alerts():
    weights = list(_QED.WEIGHT_MEAN)
    keep = [w for i, w in enumerate(weights) if i != 7]
    return keep, sum(keep)


_QED_W_KEEP, _QED_W_SUM = _qed_weights_without_alerts()
_QED_DROP_IDX = 7


WARHEAD_SMARTS = (
    ("michael_acrylate", "[CX3]=[CX3]C(=O)O"),
    ("michael_acrylamide", "C=CC(=O)N"),
    ("michael_enone", "[CX3]=[CX3]C(=O)[#6]"),
    ("vinyl_sulfone", "C=C[SX4](=O)(=O)"),
    ("nitrile", "[CX2]#N"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("cf3_amide", "C(=O)C(F)(F)F"),
    ("boronic_acid", "B(O)O"),
    ("epoxide", "C1OC1"),
    ("chloroacetamide", "ClCC(=O)N"),
)

_WARHEAD_PATTERNS = [
    (name, Chem.MolFromSmarts(smarts)) for name, smarts in WARHEAD_SMARTS
]


# --------------------------------------------------------------------------
# Selectable catalogues (additive; the published one stays the default).
#
# The catalogue is used inside the search loop, by the warhead-retention
# objective, the warhead floor g1, and the reversibility objective, so changing
# it changes the search. "published" is therefore frozen: with no call to
# ``set_catalogue`` every result reproduces exactly as before. See
# ``charaka.catalogues`` for what "v2" corrects and why.
# --------------------------------------------------------------------------
_CATALOGUES = {
    # name -> (patterns, nitrile-activation SMARTS or None)
    "published": (WARHEAD_SMARTS, None),
    "v2":        (_catalogues.WARHEAD_SMARTS_V2, _catalogues.NITRILE_ACTIVATED),
}
_COMPILED_CATALOGUES = {
    name: [(n, Chem.MolFromSmarts(s)) for n, s in pats]
    for name, (pats, _) in _CATALOGUES.items()
}
_NITRILE_PATTERNS = {
    name: (Chem.MolFromSmarts(rule) if rule else None)
    for name, (_, rule) in _CATALOGUES.items()
}
_ACTIVE_CATALOGUE = "published"


def set_catalogue(name: str) -> str:
    """Select the warhead catalogue for this process. Returns the new name.

    Affects every consumer of :func:`detect_warheads`, including the search
    objectives and :func:`genotox_score`. Process-global by design so that a
    caller sets it once at start-up rather than threading it through pymoo.
    """
    global _ACTIVE_CATALOGUE
    if name not in _CATALOGUES:
        raise KeyError(f"unknown catalogue {name!r}; have {sorted(_CATALOGUES)}")
    _ACTIVE_CATALOGUE = name
    return _ACTIVE_CATALOGUE


def active_catalogue() -> str:
    """Name of the catalogue currently in force."""
    return _ACTIVE_CATALOGUE


def qed_no_alerts(mol: Chem.Mol) -> float:
    """Geometric mean of QED's seven non-ALERTS desirability functions."""
    if mol is None:
        return 0.0
    try:
        props = _QED.properties(mol)
        d = [
            _QED.ads(v, _QED.adsParameters[name])
            for name, v in props._asdict().items()
        ]
        d_keep = [v for i, v in enumerate(d) if i != _QED_DROP_IDX]
        s = sum(w * math.log(max(v, 1e-9)) for v, w in zip(d_keep, _QED_W_KEEP))
        return math.exp(s / _QED_W_SUM)
    except Exception:
        return 0.0


def sa_raw(mol: Chem.Mol) -> float:
    """Raw Ertl-Schuffenhauer synthetic-accessibility score.

    Returns 4.0 if RDKit's contrib ``sascorer`` is unavailable.
    """
    if not _SA_AVAILABLE or mol is None:
        return 4.0
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 4.0


def sa_norm(mol: Chem.Mol) -> float:
    """SA score normalised to ``[0, 1]`` (higher = more synthesisable)."""
    if not _SA_AVAILABLE or mol is None:
        return 0.5
    return max(0.0, min(1.0, (10.0 - sa_raw(mol)) / 9.0))


def detect_warheads(mol: Chem.Mol, catalogue: Optional[str] = None) -> Set[str]:
    """Return the set of warhead patterns matched in ``mol``.

    ``catalogue`` overrides the process-wide selection for one call; by default
    the active catalogue is used, which is ``"published"`` unless
    :func:`set_catalogue` was called.
    """
    if mol is None:
        return set()
    name = catalogue or _ACTIVE_CATALOGUE
    names = {
        n
        for n, pattern in _COMPILED_CATALOGUES[name]
        if pattern is not None and mol.HasSubstructMatch(pattern)
    }
    # A nitrile is only a covalent warhead when it is activated; catalogues that
    # define a rule drop the ones that are merely decoration.
    rule = _NITRILE_PATTERNS[name]
    if rule is not None and "nitrile" in names and not mol.HasSubstructMatch(rule):
        names.discard("nitrile")
    return names


def mol_logp(mol: Chem.Mol) -> float:
    """Raw Crippen logP (atom-contribution formula; fixed, non-learnable)."""
    if mol is None:
        return 0.0
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return 0.0


def logp_score(mol: Chem.Mol) -> float:
    """Lipophilicity desirability in [0, 1]: high for low/safe logP, low for high.

    Uses Crippen logP -- a fixed atom-contribution formula, NOT a learned model,
    so it cannot be reward-hacked the way a learned ADMET predictor can (the
    documented reason DeepDL was dropped). Sigmoid centred at logP = 3, the
    inflection where ADMET safety drops sharply in our data (safer_frac falls
    from ~0.57 below logP 3 to ~0.41 above logP 4).
    """
    if mol is None:
        return 0.0
    try:
        lp = Crippen.MolLogP(mol)
    except Exception:
        return 0.0
    return 1.0 / (1.0 + math.exp(1.5 * (lp - 3.0)))


def logp_relative_score(mol: Chem.Mol, anchor_logp: float) -> float:
    """Aggressive relative lipophilicity desirability in [0, 1]. Unlike
    logp_score (a fixed sigmoid at logP=3 that plateaus once a molecule is
    "lean enough"), this centres the sigmoid on the ANCHOR's own logP, so the
    objective keeps rewarding leaner-than-parent no matter how lean the anchor
    already is. cand==anchor -> 0.5, leaner -> ->1, greasier -> ->0. Directly
    targets the per-parent safety metric (hERG/CYP liability tracks
    lipophilicity relative to the source inhibitor)."""
    if mol is None:
        return 0.0
    try:
        lp = Crippen.MolLogP(mol)
    except Exception:
        return 0.0
    # steepness 2.5 (aggressive): a sharp reward for going below the anchor's
    # logP, to fight the greasy pull of QED and the genotoxicity objective.
    return 1.0 / (1.0 + math.exp(2.5 * (lp - anchor_logp)))


def warhead_score(orig_warheads: Set[str], cand_warheads: Set[str]) -> float:
    """Fraction of ``orig_warheads`` preserved in ``cand_warheads``."""
    if not orig_warheads:
        return 1.0
    return len(orig_warheads & cand_warheads) / len(orig_warheads)


# Per-warhead druggability/safety weights for the genotoxicity-weighted objective
# (v3.1). Grounded in measured ADMET-AI AMES per warhead class plus reversibility /
# clinical precedent: reversible warheads (nitrile, boronic acid) score high,
# irreversible genotoxic ones (chloroacetamide, Michael acceptors) score low.
WARHEAD_SAFETY = {
    "nitrile": 1.00,
    "cf3_amide": 0.90,
    "boronic_acid": 0.90,
    "michael_acrylamide": 0.50,
    "vinyl_sulfone": 0.50,
    "michael_acrylate": 0.45,
    "michael_enone": 0.45,
    "aldehyde": 0.35,
    "epoxide": 0.35,
    "chloroacetamide": 0.20,
}

# Safety weights per catalogue. v2 adds entries for the classes it can see that
# the published catalogue cannot; the published ten keep their published values.
_SAFETY_TABLES = {
    "published": WARHEAD_SAFETY,
    "v2":        _catalogues.WARHEAD_SAFETY_V2,
}


def active_warhead_safety() -> dict:
    """Safety weights matching the active catalogue."""
    return _SAFETY_TABLES[_ACTIVE_CATALOGUE]


def warhead_score_weighted(orig_warheads: Set[str], cand_warheads: Set[str]) -> float:
    """Genotoxicity-weighted warhead desirability (v3.1).

    Non-covalent anchor -> 1.0 (warheads neither rewarded nor required). Covalent
    anchor -> the max safety weight over the candidate's warheads. This rewards
    covalency, steers toward reversible/cleaner warheads, and allows warhead-hopping
    (e.g. a chloroacetamide anchor onto a nitrile analog, 0.20 -> 1.00). Returns
    0.0 if covalency is lost entirely.
    """
    if not orig_warheads:
        return 1.0
    if not cand_warheads:
        return 0.0
    return max(active_warhead_safety().get(w, 0.30) for w in cand_warheads)


def genotox_score(mol: Chem.Mol) -> float:
    """Reversibility / genotoxicity desirability in [0, 1] for use as a soft
    objective (NOT a hard filter). 1.0 = fully clean (nitrile, boronic acid,
    or no warhead at all); low = irreversible genotoxic (chloroacetamide 0.20).
    The WORST warhead present sets the score (min), so a dirty warhead cannot
    hide behind a clean one -- unlike warhead_score_weighted, which uses max.
    """
    if mol is None:
        return 0.0
    whs = detect_warheads(mol)
    if not whs:
        return 1.0
    return min(active_warhead_safety().get(w, 0.30) for w in whs)


@dataclass
class CompositeWeights:
    qed: float = 0.50
    sa: float = 0.15
    warhead: float = 0.20
    potency: float = 0.15


def composite(
    qed_na: float,
    sa: float,
    warhead: float,
    potency: Optional[float],
    weights: CompositeWeights = CompositeWeights(),
) -> float:
    """Linear composite ranking score."""
    pot_term = weights.potency * (potency if potency is not None else 0.0)
    return (
        weights.qed * qed_na
        + weights.sa * sa
        + weights.warhead * warhead
        + pot_term
    )
