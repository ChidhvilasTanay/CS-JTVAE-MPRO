r"""Latent-space multi-objective search with anchor-relative uncertainty cap.

The search runs in the box ``mu_anchor +/- radius`` around each anchor's
encoder mean. Objectives are maximised (negated for pymoo's minimisation
convention): the four core objectives are QED-no-alerts, SA-normalised,
warhead retention, and predicted potency, with optional lipophilicity and
warhead-reversibility objectives added for the reported six-objective run.
Five inequality constraints gate the feasible region:

==================  ====================================================
``g1``              ``0.5 - warhead_score`` (only if anchor has warheads)
``g2``              ``sa_raw - sa_ceiling``
``g3``              ``pot_floor - potency``
``g4``              ``tanimoto_floor - tanimoto``
``g5``              ``ensemble_std - uncertainty_cap``
==================  ====================================================

The uncertainty cap is anchor-relative
(:math:`\max(U_{global}, \gamma \cdot s(\boldsymbol{\mu}_a))`); a global
percentile cap rejects warhead-bearing anchors by construction because
those anchors are themselves out-of-distribution. The optimiser (NSGA-II or
NSGA-III) is selected by the caller; the reported run uses NSGA-III.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize

import re

from .objectives import (
    detect_warheads,
    genotox_score,
    logp_relative_score,
    logp_score,
    mol_logp,
    qed_no_alerts,
    sa_norm,
    sa_raw,
    warhead_score,
    warhead_score_weighted,
)

# Irreversible-warhead names (same definition as the post-hoc reversible filter:
# Michael acceptors, chloroacetamide, aldehyde, epoxide). Used only by the
# optional reversible_only constraint.
_IRREV_RE = re.compile(r"chloro|michael|acryl|enone|aldehyde|epoxide")


def _has_irreversible(warheads) -> bool:
    return any(_IRREV_RE.search(w) for w in warheads)


@dataclass
class NSGAConfig:
    pop_size: int = 60
    generations: int = 30
    radius: float = 0.5
    sbx_eta: int = 15
    pm_eta: int = 20
    pot_floor_abs: float = 0.40
    pot_floor_rel: float = 0.65
    sa_floor_base: float = 4.0
    sa_floor_rel_delta: float = 0.3
    tanimoto_min: float = 0.20
    uncertainty_global: float = 0.50
    uncertainty_margin: float = 1.30
    use_logp: bool = False  # add a 5th objective: lipophilicity desirability (non-gameable)
    logp_relative: bool = False  # use aggressive leaner-than-parent logP desirability (vs fixed sigmoid at 3)
    use_genotox: bool = False  # add a 6th objective: reversibility/genotoxicity desirability
    algorithm: str = "NSGA2"  # optimiser: NSGA2 (<=4-5 obj) or NSGA3 (many-objective)
    logp_constraint: bool = False  # hard constraint: logP(analog) <= logP(parent) + margin
    logp_margin: float = 0.0
    reversible_only: bool = False  # hard constraint: reject candidates carrying an irreversible warhead
    warhead_weighting: bool = False  # v3.1: genotoxicity-weighted warhead term (steer to reversible)
    warhead_floor: float = 0.5  # min warhead score for a covalent anchor's analog to be feasible


@dataclass
class Anchor:
    """All anchor-specific invariants needed during search."""
    smi: str
    mu: np.ndarray
    fingerprint: object
    warheads: Set[str]
    qed_na: float
    sa_norm: float
    sa_raw: float
    potency: Optional[float]
    pred_log_ic50: Optional[float]
    pred_std: float
    logp: float = 0.0

    @property
    def has_warheads(self) -> bool:
        return bool(self.warheads)


@dataclass
class CandidateEval:
    valid: bool
    cand_smi: Optional[str] = None
    cand_mol: Optional[Chem.Mol] = None
    qed_na: float = 0.0
    sa_norm: float = 0.0
    sa_raw: float = 4.0
    warhead: float = 0.0
    potency: Optional[float] = None
    pred_log_ic50: Optional[float] = None
    pred_std: float = 0.0
    tanimoto: float = 0.0
    x: Optional[np.ndarray] = None


# Type for the candidate decode function. Takes (z_np, anchor) and
# returns ``CandidateEval``. Callers compose this from their model and
# bioactivity surrogate.
DecodeFn = Callable[[np.ndarray, Anchor], CandidateEval]


def build_anchor(
    smi: str,
    mu: np.ndarray,
    pred_log_ic50: Optional[float],
    potency: Optional[float],
    pred_std: float,
    fp_radius: int = 2,
    fp_bits: int = 1024,
) -> Anchor:
    """Compute scalar properties for ``smi`` and bundle them in an ``Anchor``."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Cannot parse anchor SMILES: {smi}")
    return Anchor(
        smi=smi,
        mu=mu,
        fingerprint=GetMorganFingerprintAsBitVect(mol, fp_radius, fp_bits),
        warheads=detect_warheads(mol),
        qed_na=qed_no_alerts(mol),
        sa_norm=sa_norm(mol),
        sa_raw=sa_raw(mol),
        potency=potency,
        pred_log_ic50=pred_log_ic50,
        pred_std=pred_std,
        logp=mol_logp(mol),
    )


def sa_ceiling(anchor: Anchor, cfg: NSGAConfig) -> float:
    return max(cfg.sa_floor_base, anchor.sa_raw + cfg.sa_floor_rel_delta)


def potency_floor(anchor: Anchor, cfg: NSGAConfig) -> float:
    if anchor.potency is None:
        return cfg.pot_floor_abs
    return max(cfg.pot_floor_abs, cfg.pot_floor_rel * anchor.potency)


def uncertainty_cap(anchor: Anchor, cfg: NSGAConfig) -> float:
    return max(cfg.uncertainty_global, anchor.pred_std * cfg.uncertainty_margin)


class LatentProblem(Problem):
    """pymoo problem wrapping latent-space decoding + scoring."""

    def __init__(
        self,
        anchor: Anchor,
        decode_fn: DecodeFn,
        cfg: NSGAConfig,
    ):
        self.anchor = anchor
        self.decode_fn = decode_fn
        self.cfg = cfg
        self.sa_ceiling = sa_ceiling(anchor, cfg)
        self.pot_floor = potency_floor(anchor, cfg)
        self.uncertainty_cap = uncertainty_cap(anchor, cfg)
        self.feasible: List[CandidateEval] = []
        self.n_uncertainty_dropped = 0

        n_var = len(anchor.mu)
        super().__init__(
            n_var=n_var,
            n_obj=4 + (1 if cfg.use_logp else 0) + (1 if cfg.use_genotox else 0),
            n_ieq_constr=5 + (1 if cfg.logp_constraint else 0)
                           + (1 if cfg.reversible_only else 0),
            xl=anchor.mu - cfg.radius,
            xu=anchor.mu + cfg.radius,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        f_rows: List[List[float]] = []
        g_rows: List[List[float]] = []
        for x in X:
            n_g = (5 + (1 if self.cfg.logp_constraint else 0)
                     + (1 if self.cfg.reversible_only else 0))
            n_obj = 4 + (1 if self.cfg.use_logp else 0) + (1 if self.cfg.use_genotox else 0)
            r = self.decode_fn(x, self.anchor)
            if not r.valid:
                f_rows.append([0.0] * n_obj)
                g_rows.append([1.0] * n_g)
                continue

            pot = r.potency if r.potency is not None else 0.0
            obj = [-r.qed_na, -r.sa_norm, -r.warhead, -pot]
            if self.cfg.use_logp:
                ls = (logp_relative_score(r.cand_mol, self.anchor.logp)
                      if self.cfg.logp_relative else logp_score(r.cand_mol))
                obj.append(-ls)
            if self.cfg.use_genotox:
                obj.append(-genotox_score(r.cand_mol))
            f_rows.append(obj)

            g_wh = (self.cfg.warhead_floor - r.warhead) if self.anchor.has_warheads else -1.0
            g_sa = r.sa_raw - self.sa_ceiling
            g_pot = (
                self.pot_floor - pot
                if self.anchor.potency is not None
                else -1.0
            )
            g_tan = self.cfg.tanimoto_min - r.tanimoto
            g_unc = r.pred_std - self.uncertainty_cap
            g = [g_wh, g_sa, g_pot, g_tan, g_unc]
            # logP constraint: analog must not be greasier than parent (+margin)
            g_logp = (mol_logp(r.cand_mol) - (self.anchor.logp + self.cfg.logp_margin)
                      if self.cfg.logp_constraint else None)
            if g_logp is not None:
                g.append(g_logp)
            # reversibility constraint: reject any candidate carrying an
            # irreversible warhead (Michael/chloroacetamide/aldehyde/epoxide)
            g_rev = None
            if self.cfg.reversible_only:
                g_rev = 1.0 if _has_irreversible(detect_warheads(r.cand_mol)) else -1.0
                g.append(g_rev)
            g_rows.append(g)

            within_others = (
                g_wh <= 0 and g_sa <= 0 and g_pot <= 0 and g_tan <= 0
                and (g_logp is None or g_logp <= 0)
                and (g_rev is None or g_rev <= 0)
            )
            if within_others and g_unc > 0:
                self.n_uncertainty_dropped += 1
            if within_others and g_unc <= 0:
                r.x = x.copy()
                self.feasible.append(r)

        out["F"] = np.array(f_rows)
        out["G"] = np.array(g_rows)


def make_decoder_evaluator(
    model,
    sequence_to_tree,
    assemble_tree,
    bioactivity_predictor: Optional[Callable] = None,
    device: torch.device = torch.device("cpu"),
    temperature: float = 0.8,
    fp_radius: int = 2,
    fp_bits: int = 1024,
    warhead_weighting: bool = False,
) -> DecodeFn:
    """Compose a :data:`DecodeFn` from the JT-VAE model + scoring callables.

    ``model`` must expose a ``decoder.generate(z)`` method.
    ``sequence_to_tree`` and ``assemble_tree`` are the corresponding
    callables from :mod:`charaka.tokenize` and :mod:`charaka.assemble`
    (or :mod:`charaka.scorer`) pre-bound to the vocabulary.
    ``bioactivity_predictor(smi) -> (pred, norm, std)`` predicts potency
    from a candidate SMILES; if ``None``, potency is reported as zero.
    """

    def _decode(x: np.ndarray, anchor: Anchor) -> CandidateEval:
        z = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            tok = (
                model.decoder.generate(z, temperature=temperature)
                .cpu()
                .numpy()[0]
            )
        try:
            cs, edges = sequence_to_tree(tok)
            cand_smi = assemble_tree(cs, edges, z=z[0].detach().cpu())
        except Exception:
            return CandidateEval(valid=False)
        if cand_smi is None or cand_smi == anchor.smi:
            return CandidateEval(valid=False)
        cand_mol = Chem.MolFromSmiles(cand_smi)
        if cand_mol is None:
            return CandidateEval(valid=False)

        cand_fp = GetMorganFingerprintAsBitVect(cand_mol, fp_radius, fp_bits)
        tanimoto = TanimotoSimilarity(anchor.fingerprint, cand_fp)
        cand_warheads = detect_warheads(cand_mol)
        wh = (warhead_score_weighted(anchor.warheads, cand_warheads)
              if warhead_weighting
              else warhead_score(anchor.warheads, cand_warheads))

        if bioactivity_predictor is not None:
            pred_v, pot, std = bioactivity_predictor(cand_smi)
        else:
            pred_v, pot, std = None, None, 0.0

        return CandidateEval(
            valid=True,
            cand_smi=cand_smi,
            cand_mol=cand_mol,
            qed_na=qed_no_alerts(cand_mol),
            sa_norm=sa_norm(cand_mol),
            sa_raw=sa_raw(cand_mol),
            warhead=wh,
            potency=pot,
            pred_log_ic50=pred_v,
            pred_std=std,
            tanimoto=tanimoto,
        )

    return _decode
