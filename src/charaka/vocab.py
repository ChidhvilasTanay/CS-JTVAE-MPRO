"""Junction-tree decomposition and cluster-vocabulary construction.

A molecule is decomposed into a junction tree whose nodes are
*clusters* (rings or non-ring bonds) and whose edges connect clusters
sharing at least one atom. Rings sharing more than two atoms (bridge
fusion) are merged into a composite cluster. Edge-fused rings (sharing
exactly two atoms) stay separate and are reassembled at decoding time
by the attachment scorer.
"""

import contextlib
import os
import sys
from collections import Counter, defaultdict
from typing import Iterable, List, Optional, Tuple

from rdkit import Chem, RDLogger, rdBase

RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.*")


@contextlib.contextmanager
def _silence_native_stderr():
    """Redirect OS-level stderr to NUL for the duration of the block.

    RDKit's C++ kekulisation occasionally emits pre-condition violations
    that bypass Python's logger. Suppressing them only matters during
    fragment canonicalisation of malformed substructures; the fallback
    path handles the actual failure.
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


SPECIAL_TOKENS = (
    "<PAD>", "<SOS>", "<EOS>", "<UNK>", "<START_CHILDREN>", "<END_CHILDREN>",
)


def get_clusters(mol: Chem.Mol) -> List[Tuple[int, ...]]:
    """Return the list of clusters covering ``mol``.

    Each cluster is a tuple of atom indices. Clusters are either rings
    detected by RDKit's SSSR (with bridge-fused rings merged) or non-ring
    bonds.
    """
    if mol.GetNumAtoms() == 1:
        return [(0,)]

    rings = [list(r) for r in Chem.GetSymmSSSR(mol)]
    rings = _merge_bridge_fused_rings(rings)

    clusters = [tuple(sorted(r)) for r in rings]
    ring_atoms = {a for r in rings for a in r}

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if not (a1 in ring_atoms and a2 in ring_atoms):
            clusters.append(tuple(sorted((a1, a2))))
        elif not bond.IsInRing():
            clusters.append(tuple(sorted((a1, a2))))

    return list(dict.fromkeys(clusters))


def _merge_bridge_fused_rings(
    rings: List[List[int]],
) -> List[List[int]]:
    """Iteratively merge rings sharing more than two atoms."""
    changed = True
    while changed:
        changed = False
        new_rings: List[List[int]] = []
        used = [False] * len(rings)
        for i in range(len(rings)):
            if used[i]:
                continue
            cur = set(rings[i])
            used[i] = True
            for j in range(i + 1, len(rings)):
                if used[j]:
                    continue
                if len(cur & set(rings[j])) > 2:
                    cur |= set(rings[j])
                    used[j] = True
                    changed = True
            new_rings.append(list(cur))
        rings = new_rings
    return rings


def cluster_smiles(
    mol: Chem.Mol, atom_indices: Iterable[int],
) -> Optional[str]:
    """Canonical Kekule SMILES for the subgraph induced by ``atom_indices``.

    Kekule form is used so fragments containing aromatic atoms but not
    the full aromatic ring system can still be parsed back by RDKit.
    Aromaticity is re-perceived on the assembled molecule.
    """
    atom_indices = list(atom_indices)
    if len(atom_indices) == 1:
        return f"[{mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()}]"

    with _silence_native_stderr():
        try:
            kekulised = Chem.Mol(mol)
            Chem.Kekulize(kekulised, clearAromaticFlags=True)
            atom_set = set(atom_indices)
            bonds_inside = [
                b.GetIdx()
                for b in kekulised.GetBonds()
                if b.GetBeginAtomIdx() in atom_set and b.GetEndAtomIdx() in atom_set
            ]
            smi = Chem.MolFragmentToSmiles(
                kekulised,
                atomsToUse=atom_indices,
                bondsToUse=bonds_inside,
                kekuleSmiles=True,
                canonical=True,
                isomericSmiles=True,
            )
            roundtripped = Chem.MolFromSmiles(smi)
            if roundtripped is not None:
                Chem.AssignStereochemistry(roundtripped, cleanIt=True, force=True)
                return Chem.MolToSmiles(
                    roundtripped, canonical=True, isomericSmiles=True,
                )
            return smi
        except Exception:
            return _cluster_smiles_fallback(mol, atom_indices)


def _cluster_smiles_fallback(
    mol: Chem.Mol, atom_indices: List[int],
) -> Optional[str]:
    with _silence_native_stderr():
        try:
            atom_set = set(atom_indices)
            bonds_inside = [
                b.GetIdx()
                for b in mol.GetBonds()
                if b.GetBeginAtomIdx() in atom_set and b.GetEndAtomIdx() in atom_set
            ]
            smi = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=atom_indices,
                bondsToUse=bonds_inside,
                canonical=True,
                isomericSmiles=True,
            )
            roundtripped = Chem.MolFromSmiles(smi)
            if roundtripped is not None:
                Chem.AssignStereochemistry(roundtripped, cleanIt=True, force=True)
                return Chem.MolToSmiles(
                    roundtripped, canonical=True, isomericSmiles=True,
                )
            return smi
        except Exception:
            return None


def build_junction_tree_edges(
    clusters: List[Tuple[int, ...]],
) -> List[Tuple[int, int]]:
    """Return the spanning-tree edges of the cluster-overlap graph."""
    n = len(clusters)
    if n == 0:
        return []

    adjacency: defaultdict = defaultdict(list)
    for i in range(n):
        si = set(clusters[i])
        for j in range(i + 1, n):
            if si & set(clusters[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited = {0}
    queue = [0]
    edges: List[Tuple[int, int]] = []
    while queue:
        u = queue.pop(0)
        for v in adjacency[u]:
            if v not in visited:
                visited.add(v)
                edges.append((u, v))
                queue.append(v)
    return edges


def smiles_to_tree(
    smi: str,
) -> Optional[Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, ...]]]]:
    """Decompose ``smi`` into ``(cluster_smiles, tree_edges, atom_indices)``.

    Returns ``None`` if RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    clusters = get_clusters(mol)
    if not clusters:
        return None
    cluster_smis = [cluster_smiles(mol, list(c)) for c in clusters]
    edges = build_junction_tree_edges(clusters)
    return cluster_smis, edges, clusters


class Vocabulary:
    """Mapping between cluster SMILES strings and integer token ids."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.token_to_idx = {t: i for i, t in enumerate(tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(tokens)}
        self.pad = self.token_to_idx["<PAD>"]
        self.sos = self.token_to_idx["<SOS>"]
        self.eos = self.token_to_idx["<EOS>"]
        self.unk = self.token_to_idx["<UNK>"]
        self.start_children = self.token_to_idx["<START_CHILDREN>"]
        self.end_children = self.token_to_idx["<END_CHILDREN>"]
        self.n_specials = len(SPECIAL_TOKENS)

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_smiles(
        cls,
        smiles_iter: Iterable[str],
        min_freq: int = 2,
        verbose: bool = False,
    ) -> "Vocabulary":
        """Build a vocabulary by decomposing every molecule in ``smiles_iter``.

        Clusters with frequency below ``min_freq`` are dropped to avoid
        long-tail OOV explosion.
        """
        counts: Counter = Counter()
        for i, smi in enumerate(smiles_iter):
            if verbose and i and i % 5000 == 0:
                print(f"  processed {i}  vocab so far={len(counts)}")
            result = smiles_to_tree(smi)
            if result is None:
                continue
            cs, _, _ = result
            for c in cs:
                if c is not None:
                    counts[c] += 1
        kept = sorted(c for c, n in counts.items() if n >= min_freq)
        tokens = list(SPECIAL_TOKENS) + kept
        return cls(tokens)

    def encode_cluster(self, smi: Optional[str]) -> int:
        return self.token_to_idx.get(smi, self.unk) if smi is not None else self.unk

    def decode_cluster(self, idx: int) -> Optional[str]:
        token = self.idx_to_token.get(int(idx))
        if token is None or token in SPECIAL_TOKENS:
            return None
        return token

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for token in self.tokens:
                f.write(f"{token}\n")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            tokens = [line.rstrip("\n") for line in f if line.strip()]
        return cls(tokens)
