"""Tree-to-sequence serialisation for the JT-VAE token model.

Sequence format::

    <SOS> root <START_CHILDREN> child1 <START_CHILDREN> ... <END_CHILDREN>
                                  child2 ... <END_CHILDREN> <EOS>

Every cluster token is followed by a child-bracketed block; leaves emit
an empty ``<START_CHILDREN><END_CHILDREN>`` pair. Token sequences are
padded with ``<PAD>`` up to ``max_len``.
"""

from collections import defaultdict
from typing import List, Sequence, Tuple

from .vocab import Vocabulary

MAX_TREE_LEN = 100


def tree_to_sequence(
    cluster_smis: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    vocab: Vocabulary,
    max_len: int = MAX_TREE_LEN,
) -> List[int]:
    """Serialise a junction tree as a depth-first token sequence."""
    if not cluster_smis:
        return [vocab.sos, vocab.eos] + [vocab.pad] * (max_len - 2)

    adjacency: defaultdict = defaultdict(list)
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    seq: List[int] = [vocab.sos]
    visited: set = set()

    def dfs(node: int) -> None:
        visited.add(node)
        seq.append(vocab.encode_cluster(cluster_smis[node]))
        seq.append(vocab.start_children)
        children = sorted(
            (n for n in adjacency[node] if n not in visited),
            key=lambda n: cluster_smis[n] or "",
        )
        for child in children:
            if len(seq) >= max_len - 2:
                break
            dfs(child)
        seq.append(vocab.end_children)

    dfs(0)
    seq.append(vocab.eos)

    if len(seq) > max_len:
        seq = seq[: max_len - 1] + [vocab.eos]
    while len(seq) < max_len:
        seq.append(vocab.pad)
    return seq


def sequence_to_tree(
    token_ids: Sequence[int],
    vocab: Vocabulary,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Inverse of :func:`tree_to_sequence`. Robust to malformed sequences."""
    cluster_smis: List[str] = []
    edges: List[Tuple[int, int]] = []
    stack: List[int] = []

    for tok in token_ids:
        tok = int(tok)
        if tok == vocab.sos:
            continue
        if tok in (vocab.eos, vocab.pad):
            break
        if tok == vocab.start_children:
            if cluster_smis:
                stack.append(len(cluster_smis) - 1)
            continue
        if tok == vocab.end_children:
            if stack:
                stack.pop()
            continue
        if tok < vocab.n_specials:
            continue

        cs = vocab.decode_cluster(tok)
        if cs is None:
            continue
        new_idx = len(cluster_smis)
        cluster_smis.append(cs)
        if stack:
            edges.append((stack[-1], new_idx))

    return cluster_smis, edges
