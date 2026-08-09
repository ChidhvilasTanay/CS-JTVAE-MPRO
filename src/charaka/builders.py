"""Runtime pipeline builders for the latent-space search.

Two factory functions shared by the generation and MOO-catalogue scripts:

* :func:`_build_potency_predictor` wraps the frozen encoder and the
  bioactivity ensemble into a ``predict(smi) -> (log_ic50, norm, std)`` closure.
* :func:`_build_assembler` wraps the attachment scorer and the fallback tree
  assembler into an ``assemble(cs, edges, z) -> smiles`` closure.
"""

from __future__ import annotations

import torch
from rdkit import Chem

from .assemble import assemble_tree
from .bioactivity import POTENCY_CENTRE, POTENCY_SLOPE
from .scorer import assemble_with_scorer
from .tokenize import tree_to_sequence
from .vocab import smiles_to_tree


def _build_potency_predictor(encoder, ensemble, vocab, max_len, device):
    def predict(smi):
        result = smiles_to_tree(smi)
        if result is None:
            return None, 0.5, 0.0
        cs, edges, _ = result
        seq = torch.tensor(
            tree_to_sequence(cs, edges, vocab, max_len),
            dtype=torch.long, device=device,
        ).unsqueeze(0)
        encoder.eval()
        with torch.no_grad():
            _, mu, _ = encoder(seq)
            pred, std = ensemble(mu)
        pred_val = float(pred.item())
        norm = float(torch.sigmoid(-(pred - POTENCY_CENTRE) * POTENCY_SLOPE).item())
        return pred_val, norm, float(std.item())

    return predict


def _build_assembler(scorer, vocab, device, max_attempts, use_scorer=True):
    def _finalise(rwmol):
        try:
            Chem.SanitizeMol(rwmol)
            smi = Chem.MolToSmiles(rwmol, canonical=True, isomericSmiles=True)
            return smi if Chem.MolFromSmiles(smi) is not None else None
        except Exception:
            return None

    def assemble(cs, edges, z=None):
        if use_scorer and z is not None:
            out = assemble_with_scorer(
                cs, edges, scorer, z, device,
                max_attempts=max_attempts, finaliser=_finalise,
            )
            if out is not None:
                return out
        return assemble_tree(cs, edges)

    return assemble
