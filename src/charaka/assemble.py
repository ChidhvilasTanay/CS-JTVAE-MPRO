"""Molecular graph assembly from a junction tree.

The exhaustive assembler enumerates both single-atom and edge-fused
alignments per tree edge and backtracks until a sanitisable molecule
emerges. The learned attachment scorer (:mod:`charaka.scorer`) can be
plugged in to rank candidate alignments instead of using priority
heuristics.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple

from rdkit import Chem


MAX_BACKTRACK_ATTEMPTS = 10_000

_MAX_VALENCE = {
    "C": 4, "N": 3, "O": 2, "S": 6, "P": 5,
    "F": 1, "Cl": 1, "Br": 1, "I": 1, "H": 1,
}


def _spare_valence(rwmol: Chem.RWMol, atom_idx: int) -> float:
    atom = rwmol.GetAtomWithIdx(atom_idx)
    used = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
    return _MAX_VALENCE.get(atom.GetSymbol(), 4) - used


def _attach_single(
    rwmol: Chem.RWMol,
    child_mol: Chem.Mol,
    parent_atom: int,
    child_atom: int,
) -> dict:
    new_indices = {child_atom: parent_atom}
    for ci in range(child_mol.GetNumAtoms()):
        if ci == child_atom:
            continue
        src = child_mol.GetAtomWithIdx(ci)
        ni = rwmol.AddAtom(Chem.Atom(src.GetSymbol()))
        dst = rwmol.GetAtomWithIdx(ni)
        dst.SetFormalCharge(src.GetFormalCharge())
        dst.SetIsAromatic(src.GetIsAromatic())
        dst.SetChiralTag(src.GetChiralTag())
        new_indices[ci] = ni

    for bond in child_mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ni, nj = new_indices[a], new_indices[b]
        if ni == nj or rwmol.GetBondBetweenAtoms(ni, nj) is not None:
            continue
        try:
            bidx = rwmol.AddBond(ni, nj, bond.GetBondType()) - 1
            stereo = bond.GetStereo()
            if stereo != Chem.BondStereo.STEREONONE:
                new_bond = rwmol.GetBondWithIdx(bidx)
                new_bond.SetStereo(stereo)
                ref_atoms = bond.GetStereoAtoms()
                if (len(ref_atoms) == 2
                        and ref_atoms[0] in new_indices
                        and ref_atoms[1] in new_indices):
                    new_bond.SetStereoAtoms(
                        new_indices[ref_atoms[0]],
                        new_indices[ref_atoms[1]],
                    )
        except Exception:
            continue
    return new_indices


def _attach_edge(
    rwmol: Chem.RWMol,
    child_mol: Chem.Mol,
    parent_pair: Tuple[int, int],
    child_pair: Tuple[int, int],
) -> Optional[dict]:
    p_i, p_j = parent_pair
    c_i, c_j = child_pair
    new_indices = {c_i: p_i, c_j: p_j}
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

    shared = frozenset(parent_pair)
    for bond in child_mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ni, nj = new_indices[a], new_indices[b]
        if ni == nj:
            continue
        if frozenset((ni, nj)) == shared:
            continue
        if rwmol.GetBondBetweenAtoms(ni, nj) is not None:
            continue
        try:
            rwmol.AddBond(ni, nj, bond.GetBondType())
        except Exception:
            return None
    return new_indices


def _enumerate_alignments(
    rwmol: Chem.RWMol,
    child_mol: Chem.Mol,
    parent_atoms: Sequence[int],
    allow_edge: bool = True,
) -> List[Tuple]:
    """Return candidate alignments sorted by priority (lower = tried first).

    The priority ordering interleaves single-atom and edge-fused matches
    so that single-atom alignments compatible with parent valence are
    attempted before more exotic edge-fused alignments.
    """
    cands: List[Tuple[int, Tuple]] = []
    parent_rev = list(reversed(parent_atoms))
    parent_set = set(parent_atoms)

    for rank, p_idx in enumerate(parent_rev):
        if _spare_valence(rwmol, p_idx) <= 0:
            continue
        p_sym = rwmol.GetAtomWithIdx(p_idx).GetSymbol()
        for c_idx in range(child_mol.GetNumAtoms()):
            if child_mol.GetAtomWithIdx(c_idx).GetSymbol() == p_sym:
                cands.append((rank, ("single", p_idx, c_idx)))

    for rank, p_idx in enumerate(parent_rev):
        if _spare_valence(rwmol, p_idx) > 0:
            cands.append((100 + rank, ("single", p_idx, 0)))

    if allow_edge:
        for p_bond in rwmol.GetBonds():
            p_i, p_j = p_bond.GetBeginAtomIdx(), p_bond.GetEndAtomIdx()
            if p_i not in parent_set or p_j not in parent_set:
                continue
            a_pi = rwmol.GetAtomWithIdx(p_i)
            a_pj = rwmol.GetAtomWithIdx(p_j)
            p_arom = a_pi.GetIsAromatic() and a_pj.GetIsAromatic()
            p_bt = p_bond.GetBondType()
            for c_bond in child_mol.GetBonds():
                c_i, c_j = c_bond.GetBeginAtomIdx(), c_bond.GetEndAtomIdx()
                a_ci = child_mol.GetAtomWithIdx(c_i)
                a_cj = child_mol.GetAtomWithIdx(c_j)
                c_arom = a_ci.GetIsAromatic() and a_cj.GetIsAromatic()
                if c_bond.GetBondType() != p_bt and not (p_arom and c_arom):
                    continue
                pr_base = 200 if (p_arom and c_arom) else 300
                if (a_ci.GetSymbol() == a_pi.GetSymbol()
                        and a_cj.GetSymbol() == a_pj.GetSymbol()):
                    cands.append((pr_base, ("edge", (p_i, p_j), (c_i, c_j))))
                if (a_ci.GetSymbol() == a_pj.GetSymbol()
                        and a_cj.GetSymbol() == a_pi.GetSymbol()):
                    cands.append((pr_base, ("edge", (p_i, p_j), (c_j, c_i))))

    for rank, p_idx in enumerate(parent_rev):
        cands.append((400 + rank, ("single", p_idx, 0)))

    seen: set = set()
    out: List[Tuple] = []
    for _, align in sorted(cands, key=lambda x: x[0]):
        if align not in seen:
            seen.add(align)
            out.append(align)
    return out


def _finalise(
    rwmol: Chem.RWMol,
    post_process: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """Sanitise ``rwmol`` and return canonical isomeric SMILES.

    Tries three increasingly permissive sanitisation modes; any returned
    string is validated by re-parsing through ``Chem.MolFromSmiles`` so
    callers can rely on the result being parseable.
    """

    def _wrap(smi: Optional[str]) -> Optional[str]:
        if smi is None or Chem.MolFromSmiles(smi) is None:
            return None
        return post_process(smi) if post_process is not None else smi

    try:
        Chem.SanitizeMol(rwmol)
        Chem.AssignStereochemistry(rwmol, cleanIt=True, force=True)
        out = _wrap(Chem.MolToSmiles(rwmol, canonical=True, isomericSmiles=True))
        if out is not None:
            return out
    except Exception:
        pass
    try:
        Chem.SanitizeMol(
            rwmol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE,
        )
        Chem.AssignStereochemistry(rwmol, cleanIt=True, force=True)
        out = _wrap(Chem.MolToSmiles(rwmol, canonical=True, isomericSmiles=True))
        if out is not None:
            return out
    except Exception:
        pass
    try:
        return _wrap(
            Chem.MolToSmiles(rwmol.GetMol(), canonical=False, isomericSmiles=True),
        )
    except Exception:
        return None


def assemble_tree(
    cluster_smis: Sequence[Optional[str]],
    edges: Sequence[Tuple[int, int]],
    max_attempts: int = MAX_BACKTRACK_ATTEMPTS,
    allow_edge: bool = True,
    post_process: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """Assemble a molecule from cluster SMILES + tree edges.

    Returns canonical isomeric SMILES on success, ``None`` otherwise.
    """
    if not cluster_smis:
        return None

    mols: List[Chem.Mol] = []
    for cs in cluster_smis:
        if cs is None:
            return None
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
            return _finalise(rwmol, post_process=post_process)
        node, par = order[idx]
        if par is None or node >= len(mols):
            return recurse(rwmol, atom_map, idx + 1)

        child_mol = mols[node]
        parent_atoms = atom_map.get(par, [])
        alignments = _enumerate_alignments(
            rwmol, child_mol, parent_atoms, allow_edge=allow_edge,
        )
        if not alignments:
            return None

        for align in alignments:
            counter[0] += 1
            if counter[0] >= max_attempts:
                return None
            rwmol_try = copy.deepcopy(rwmol)
            map_try = {k: list(v) for k, v in atom_map.items()}

            if align[0] == "single":
                _, p_idx, c_idx = align
                new_indices = _attach_single(rwmol_try, child_mol, p_idx, c_idx)
            else:
                _, p_pair, c_pair = align
                new_indices = _attach_edge(rwmol_try, child_mol, p_pair, c_pair)
                if new_indices is None:
                    continue

            map_try[node] = list(new_indices.values())
            result = recurse(rwmol_try, map_try, idx + 1)
            if result is not None:
                return result
        return None

    root = Chem.RWMol(mols[0])
    init_map = {0: list(range(root.GetNumAtoms()))}
    return recurse(root, init_map, 1)
