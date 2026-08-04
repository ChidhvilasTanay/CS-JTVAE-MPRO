"""Build the cluster vocabulary used by the trained JT-VAE checkpoint.

The published ``jtvae_best.pth`` was trained against a vocabulary built
from the COVID Moonshot training split combined with a 50k random
sample of ZINC-250K. Running this script with the default arguments
reproduces that vocabulary deterministically.

Output: ``checkpoints/vocab.txt`` (one token per line, special tokens
first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charaka.data import load_moonshot, load_zinc250k, split  # noqa: E402
from charaka.vocab import Vocabulary  # noqa: E402


def main(config_path: str, n_zinc: int) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = load_moonshot(cfg["dataset"]["csv_path"])
    train_df, _ = split(
        df, test_size=cfg["dataset"]["val_fraction"], seed=cfg["seed"],
    )
    smiles = train_df["canonical"].tolist()

    if n_zinc > 0:
        zinc_path = Path(cfg["dataset"].get("zinc_csv", "data/zinc250k.csv"))
        if zinc_path.exists():
            zinc = load_zinc250k(str(zinc_path))
            sample = zinc["canonical"].sample(
                n=min(n_zinc, len(zinc)), random_state=cfg["seed"],
            ).tolist()
            smiles += sample
            print(f"Including {len(sample)} ZINC samples in vocab build.")
        else:
            print(f"ZINC file {zinc_path} not found; building Moonshot-only vocab.")

    print(f"Building vocabulary from {len(smiles)} molecules...")
    vocab = Vocabulary.from_smiles(
        smiles, min_freq=cfg["vocab"]["min_freq"], verbose=True,
    )
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / "vocab.txt"
    vocab.save(out)
    print(f"Vocabulary size: {len(vocab)}  -> {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jtvae_v3_aug.yaml")
    parser.add_argument(
        "--n-zinc", type=int, default=50_000,
        help="ZINC samples to include (matches the published checkpoint).",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.n_zinc))
