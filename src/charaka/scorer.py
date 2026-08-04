"""z-conditioned GINEConv attachment scorer.

Given a partial assembly and a candidate (parent, child) alignment, the
scorer produces a scalar score conditioned on the source molecule's
latent code. Candidate alignments enumerated per tree edge are ranked
by score; assembly proceeds in score-descending order with backtracking.

The scorer reads features over the fused candidate subgraph through a
three-layer GINEConv stack with dual-pool readout (global mean +
junction-atom mean) concatenated with the latent code.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool


SYMBOLS = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P")
ATOM_FEATURE_DIM = 17
BOND_FEATURE_DIM = 4

_HYBRIDISATION = {
    Chem.HybridizationType.SP: 1.0,
    Chem.HybridizationType.SP2: 1.0,
    Chem.HybridizationType.SP3: 0.5,
}

_BOND_TYPE_INDEX = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}


def atom_features(atom: Chem.Atom, is_junction: bool) -> List[float]:
    """17-dimensional atom feature vector used by the scorer GNN."""
    one_hot = [1.0 if atom.GetSymbol() == s else 0.0 for s in SYMBOLS]
    one_hot.append(1.0 if atom.GetSymbol() not in SYMBOLS else 0.0)
    return one_hot + [
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        atom.GetDegree() / 4.0,
        (atom.GetFormalCharge() + 2.0) / 4.0,
        atom.GetTotalNumHs() / 4.0,
        _HYBRIDISATION.get(atom.GetHybridization(), 0.0),
        1.0 if is_junction else 0.0,
    ]


def bond_features(bond: Chem.Bond) -> List[float]:
    """4-dimensional one-hot bond-type feature vector."""
    vec = [0.0] * BOND_FEATURE_DIM
    vec[_BOND_TYPE_INDEX.get(bond.GetBondType(), 0)] = 1.0
    return vec


class GINEAttachmentScorer(nn.Module):
    """3-layer GINEConv attachment scorer with z-conditioned MLP head."""

    def __init__(
        self,
        latent_dim: int,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_embed = nn.Linear(atom_dim, hidden)
        self.edge_embed = nn.Linear(bond_dim, hidden)
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
            )
            for _ in range(n_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        junction_mask: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, e))

        h_mean = global_mean_pool(h, batch)
        jm = junction_mask.unsqueeze(-1)
        h_j = global_add_pool(h * jm, batch) / global_add_pool(jm, batch).clamp(min=1.0)
        pooled = torch.cat([h_mean, h_j, z], dim=-1)
        return self.head(pooled).squeeze(-1)


def candidate_alignments(
    parent_mol: Chem.Mol, child_mol: Chem.Mol,
) -> List[dict]:
    """Enumerate single-atom and edge-fused (parent, child) alignments."""
    cands: List[dict] = []
    for p in range(parent_mol.GetNumAtoms()):
        for c in range(child_mol.GetNumAtoms()):
            if parent_mol.GetAtomWithIdx(p).GetSymbol() == child_mol.GetAtomWithIdx(c).GetSymbol():
                cands.append({
                    "kind": "single", "p_atoms": (p,), "c_atoms": (c,),
                    "bond_type": None,
                })

    for p_bond in parent_mol.GetBonds():
        p_i, p_j = p_bond.GetBeginAtomIdx(), p_bond.GetEndAtomIdx()
        sp_i = parent_mol.GetAtomWithIdx(p_i).GetSymbol()
        sp_j = parent_mol.GetAtomWithIdx(p_j).GetSymbol()
        for c_bond in child_mol.GetBonds():
            c_i, c_j = c_bond.GetBeginAtomIdx(), c_bond.GetEndAtomIdx()
            sc_i = child_mol.GetAtomWithIdx(c_i).GetSymbol()
            sc_j = child_mol.GetAtomWithIdx(c_j).GetSymbol()
            if sc_i == sp_i and sc_j == sp_j:
                cands.append({
                    "kind": "edge", "p_atoms": (p_i, p_j),
                    "c_atoms": (c_i, c_j), "bond_type": p_bond.GetBondType(),
                })
            if sc_i == sp_j and sc_j == sp_i:
                cands.append({
                    "kind": "edge", "p_atoms": (p_i, p_j),
                    "c_atoms": (c_j, c_i), "bond_type": p_bond.GetBondType(),
                })
    return cands


def apply_alignment(
    rwmol: Chem.RWMol,
    child_mol: Chem.Mol,
    cand: dict,
) -> Optional[dict]:
    """Merge ``child_mol`` into ``rwmol`` using ``cand``. Returns atom map."""
    p_atoms = cand["p_atoms"]
    c_atoms = cand["c_atoms"]
    new_indices = {c_atoms[0]: p_atoms[0]}
    if cand["kind"] == "edge":
        new_indices[c_atoms[1]] = p_atoms[1]

    for ci in range(child_mol.GetNumAtoms()):
        if ci in new_indices:
            continue
        src = child_mol.GetAtomWithIdx(ci)
        ni = rwmol.AddAtom(Chem.Atom(src.GetSymbol()))
        dst = rwmol.GetAtomWithIdx(ni)
        dst.SetFormalCharge(src.GetFormalCharge())
        dst.SetIsAromatic(src.GetIsAromatic())
        dst.SetChiralTag(src.GetChiralTag())
        new_indices[ci] = ni

    shared = (
        frozenset((p_atoms[0], p_atoms[1])) if cand["kind"] == "edge" else None
    )
    for bond in child_mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ni, nj = new_indices[a], new_indices[b]
        if ni == nj:
            continue
        if shared is not None and frozenset((ni, nj)) == shared:
            continue
        if rwmol.GetBondBetweenAtoms(ni, nj) is not None:
            continue
        try:
            rwmol.AddBond(ni, nj, bond.GetBondType())
        except Exception:
            return None
    return new_indices


def candidate_graph(
    rwmol: Chem.RWMol,
    child_mol: Chem.Mol,
    cand: dict,
) -> Optional[Data]:
    """Build a torch_geometric ``Data`` for the candidate fused graph."""
    rw = copy.deepcopy(rwmol)
    new_indices = apply_alignment(rw, child_mol, cand)
    if new_indices is None:
        return None
    try:
        rw.UpdatePropertyCache(strict=False)
    except Exception:
        return None

    junction_set = set(cand["p_atoms"])
    feats: List[List[float]] = []
    junc_mask: List[float] = []
    for ai in range(rw.GetNumAtoms()):
        feats.append(atom_features(rw.GetAtomWithIdx(ai), ai in junction_set))
        junc_mask.append(1.0 if ai in junction_set else 0.0)

    edge_index: List[Tuple[int, int]] = []
    edge_attr: List[List[float]] = []
    for b in rw.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index.extend([(i, j), (j, i)])
        edge_attr.extend([bf, bf])

    if not edge_index:
        ei = torch.empty((2, 0), dtype=torch.long)
        ea = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
    else:
        ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        ea = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=ei,
        edge_attr=ea,
    )
    data.junction_mask = torch.tensor(junc_mask, dtype=torch.float32)
    return data


def score_candidates(
    scorer: GINEAttachmentScorer,
    rwmol: Chem.RWMol,
    child_mol: Chem.Mol,
    candidates: Sequence[dict],
    z: torch.Tensor,
    device: torch.device,
) -> List[Tuple[float, dict]]:
    """Score every candidate and return ``[(score, candidate), ...]`` sorted descending."""
    graphs: List[Optional[Data]] = [
        candidate_graph(rwmol, child_mol, c) for c in candidates
    ]
    valid = [(i, g) for i, g in enumerate(graphs) if g is not None]
    if not valid:
        return []

    batch = Batch.from_data_list([g for _, g in valid]).to(device)
    z_repeat = z.to(device).unsqueeze(0).expand(len(valid), -1).contiguous()
    with torch.no_grad():
        scores = scorer(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.junction_mask.to(device),
            z_repeat,
        ).cpu().numpy()

    ranked = [(float(scores[k]), candidates[i]) for k, (i, _) in enumerate(valid)]
    ranked.sort(key=lambda p: -p[0])
    return ranked


def assemble_with_scorer(
    cluster_smis: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    scorer: GINEAttachmentScorer,
    z: torch.Tensor,
    device: torch.device,
    max_attempts: int = 2000,
    finaliser=None,
) -> Optional[str]:
    """Assemble a molecule with the scorer ranking each alignment."""
    if not cluster_smis:
        return None
    mols: List[Chem.Mol] = []
    for cs in cluster_smis:
        m = Chem.MolFromSmiles(cs)
        if m is None:
            return None
        mols.append(m)

    adjacency: defaultdict = defaultdict(list)
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    order: List[Tuple[int, Optional[int]]] = []
    visited = {0}
    queue: List[Tuple[int, Optional[int]]] = [(0, None)]
    while queue:
        node, par = queue.pop(0)
        order.append((node, par))
        for nbr in adjacency[node]:
            if nbr not in visited and nbr < len(mols):
                visited.add(nbr)
                queue.append((nbr, node))

    counter = [0]

    def recurse(rwmol: Chem.RWMol, atom_map: dict, idx: int) -> Optional[str]:
        if counter[0] >= max_attempts:
            return None
        if idx >= len(order):
            return finaliser(rwmol) if finaliser is not None else None
        node, par = order[idx]
        if par is None or node >= len(mols):
            return recurse(rwmol, atom_map, idx + 1)

        child_mol = mols[node]
        parent_mol = mols[par]
        cands = candidate_alignments(parent_mol, child_mol)
        par_atoms = atom_map[par]
        translated: List[dict] = []
        for c in cands:
            if max(c["p_atoms"]) >= len(par_atoms):
                continue
            t = dict(c)
            t["p_atoms"] = tuple(par_atoms[pi] for pi in c["p_atoms"])
            translated.append(t)
        if not translated:
            return None

        ranked = score_candidates(scorer, rwmol, child_mol, translated, z, device)
        for _, cand in ranked:
            counter[0] += 1
            if counter[0] >= max_attempts:
                return None
            rwmol_try = copy.deepcopy(rwmol)
            new_indices = apply_alignment(rwmol_try, child_mol, cand)
            if new_indices is None:
                continue
            map_try = {k: list(v) for k, v in atom_map.items()}
            map_try[node] = [
                new_indices[ci] for ci in range(child_mol.GetNumAtoms())
            ]
            result = recurse(rwmol_try, map_try, idx + 1)
            if result is not None:
                return result
        return None

    root = Chem.RWMol(mols[0])
    init_map = {0: list(range(root.GetNumAtoms()))}
    return recurse(root, init_map, 1)
