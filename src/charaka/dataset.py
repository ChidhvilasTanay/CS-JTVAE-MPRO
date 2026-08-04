"""PyTorch ``Dataset`` over tokenised junction-tree sequences."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenize import tree_to_sequence
from .vocab import Vocabulary, smiles_to_tree


class JTreeDataset(Dataset):
    """Maps a SMILES dataframe to ``(token_sequence, bioactivity)`` pairs.

    Molecules that cannot be decomposed (or that yield only special tokens)
    are silently skipped. ``df`` must have a ``canonical`` column; ``value``
    is optional and treated as ``0.0`` when missing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        max_len: int,
        smiles_column: str = "canonical",
        value_column: Optional[str] = "value",
    ):
        self.sequences = []
        self.values = []
        self.smiles = []
        n_skipped = 0
        has_values = value_column is not None and value_column in df.columns

        for _, row in df.iterrows():
            smi = row[smiles_column]
            result = smiles_to_tree(smi)
            if result is None:
                n_skipped += 1
                continue
            cs, edges, _ = result
            seq = tree_to_sequence(cs, edges, vocab, max_len=max_len)
            non_special = [t for t in seq if t >= vocab.n_specials]
            if not non_special:
                n_skipped += 1
                continue

            self.sequences.append(torch.tensor(seq, dtype=torch.long))
            self.smiles.append(smi)
            if has_values:
                v = row[value_column]
                self.values.append(float(v) if pd.notna(v) else 0.0)
            else:
                self.values.append(0.0)

        self.n_skipped = n_skipped

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], torch.tensor(self.values[idx], dtype=torch.float)


def collate(batch: Iterable):
    """Default collate: stack the padded sequence and value tensors."""
    seqs, values = zip(*batch)
    return torch.stack(list(seqs)), torch.stack(list(values))
