"""Alternative covalent-warhead catalogues.

Kept in its own module, importing nothing from charaka, so that the change to
:mod:`charaka.objectives` stays small and the published catalogue there is
untouched. ``objectives.WARHEAD_SMARTS`` remains the default; nothing here is
used unless a caller opts in with ``objectives.set_catalogue``.

v2 is the published catalogue plus two corrections, both measured on the
reported analog set before being proposed:

1. Four warhead classes the published catalogue has no pattern for at all, so
   molecules carrying them were scored as having no covalent mechanism.
   Boceprevir's alpha-ketoamide and GC376's bisulfite-masked aldehyde are the
   two that matter for this project, since both are anchors. The sulfonyl
   fluoride was invented by the search itself and scored as warhead-free.

2. A nitrile only behaves as a covalent warhead when it sits alpha to an amide
   nitrogen, which withdraws enough electron density for the catalytic cysteine
   to add reversibly. That is nirmatrelvir's P1 warhead. A nitrile on an
   aromatic ring, or on a quaternary carbon with no adjacent nitrogen, is an
   inert substituent. ``[CX2]#N`` matches all of them.

   Measured consequences of (2) on the published set: eight of the twenty
   anchors are credited with a covalent warhead they do not have, which switches
   on the warhead floor ``g1`` for them and forces every analog to carry a
   nitrile it never needed; and 84 of the 122 molecules in the per-warhead AMES
   breakdown's "nitrile" class are in fact non-covalent.

Deliberately NOT adopted here: the full SMARTS-RX catalogue staged in
``objectives_curated.py``. It is a much larger surface area for the same effect
on this project's chemistry, and it silently drops the trifluoroacetyl amide,
which would erase nirmatrelvir's second electrophile.
"""

from __future__ import annotations

from typing import Dict, Tuple

# The published ten, repeated verbatim so v2 is readable as a standalone list.
# Keep in sync with objectives.WARHEAD_SMARTS; a test asserts they agree.
_PUBLISHED_TEN: Tuple[Tuple[str, str], ...] = (
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

# Classes the published catalogue cannot see at all.
_ADDITIONS: Tuple[Tuple[str, str], ...] = (
    # boceprevir, telaprevir: reversible, forms a hemiketal with Cys
    ("alpha_ketoamide", "[#6]C(=O)C(=O)[NX3]"),
    # GC376: a bisulfite adduct that releases the aldehyde in situ
    ("masked_aldehyde_bisulfite", "[CX4]([OX2H])[SX4](=O)(=O)[OX2H,OX1-]"),
    # irreversible; the search produced one of these and scored it warhead-free
    ("sulfonyl_fluoride", "[SX4](=O)(=O)[F]"),
    # irreversible alkylator, the chloroacetamide's ketone cousin
    ("halomethyl_ketone", "[#6]C(=O)[CH2][F,Cl,Br,I]"),
)

WARHEAD_SMARTS_V2: Tuple[Tuple[str, str], ...] = _PUBLISHED_TEN + _ADDITIONS

# A nitrile counts as a warhead only when this also matches.
NITRILE_ACTIVATED = "[NX3][CX4][CX2]#[NX1]"

# Looser alternatives, for sensitivity analysis rather than production use.
NITRILE_RULES: Dict[str, str] = {
    "strict": NITRILE_ACTIVATED,          # alpha to an amide N (nirmatrelvir)
    "medium": "[CX4][CX2]#[NX1]",         # any sp3-attached nitrile
    "loose":  "[!a][CX2]#[NX1]",          # any non-aromatic nitrile
}

# Per-class safety weights for the reversibility objective. The published ten
# keep their published values; the additions are placed by reversibility and
# clinical precedent, matching the reasoning in the paper's Table 3.
WARHEAD_SAFETY_V2: Dict[str, float] = {
    "nitrile": 1.00,
    "cf3_amide": 0.90,
    "boronic_acid": 0.90,
    "alpha_ketoamide": 0.80,            # reversible, two marketed HCV drugs
    "masked_aldehyde_bisulfite": 0.55,  # releases an aldehyde, so inherits some risk
    "michael_acrylamide": 0.50,
    "vinyl_sulfone": 0.50,
    "michael_acrylate": 0.45,
    "michael_enone": 0.45,
    "sulfonyl_fluoride": 0.45,
    "aldehyde": 0.35,
    "epoxide": 0.35,
    "halomethyl_ketone": 0.25,
    "chloroacetamide": 0.20,
}
