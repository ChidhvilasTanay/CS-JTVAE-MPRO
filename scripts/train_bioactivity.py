"""Train the 5-member bioactivity surrogate ensemble.

Encodes every labelled training molecule to its encoder ``mu``, then
trains each ensemble member independently on the ``(mu, log10(IC50))``
pairs. Saves one ``.pth`` per member plus the val-set uncertainty
percentile suggestions used as the global floor in :mod:`charaka.nsga`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charaka.bioactivity import BioactivityEnsemble, BioactivityMLP  # noqa: E402
from charaka.data import load_moonshot, split  # noqa: E402
from charaka.model import JTreeEncoder  # noqa: E402
from charaka.tokenize import tree_to_sequence  # noqa: E402
from charaka.training import BioConfig, train_bioactivity_member  # noqa: E402
from charaka.vocab import Vocabulary, smiles_to_tree  # noqa: E402


def _encode(df, model, vocab, max_len, device):
    mus, values = [], []
    model.eval()
    for _, row in df.iterrows():
        result = smiles_to_tree(row["canonical"])
        if result is None:
            continue
        cs, edges, _ = result
        seq = torch.tensor(
            tree_to_sequence(cs, edges, vocab, max_len),
            dtype=torch.long, device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            _, mu, _ = model(seq)
        mus.append(mu.cpu())
        values.append(float(row["value"]))
    if not mus:
        return torch.zeros(0, 256), torch.zeros(0)
    return torch.cat(mus, dim=0), torch.tensor(values, dtype=torch.float)


def main(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    vocab = Vocabulary.load(ckpt_dir / "vocab.txt")

    encoder = JTreeEncoder(
        vocab_size=len(vocab), pad_idx=vocab.pad,
        latent_dim=cfg["model"]["latent_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        n_layers=cfg["model"]["encoder_layers"],
        n_heads=cfg["model"]["n_heads"],
        max_len=cfg["model"]["max_tree_len"],
    ).to(device)
    # Load the frozen JT-VAE encoder; the surrogate trains over its latent means.
    state = torch.load(ckpt_dir / "jtvae_best.pth", map_location=device)
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state["model"].items() if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    print("Loaded JT-VAE encoder from jtvae_best.pth")

    df = load_moonshot(cfg["dataset"]["csv_path"])
    train_df, val_df = split(
        df, test_size=cfg["dataset"]["val_fraction"], seed=cfg["seed"],
    )
    train_df = train_df[train_df["value"].notna()].reset_index(drop=True)
    val_df = val_df[val_df["value"].notna()].reset_index(drop=True)
    print(f"Labelled train: {len(train_df)}  val: {len(val_df)}")

    print("Encoding training set...")
    mu_train, y_train = _encode(
        train_df, encoder, vocab, cfg["model"]["max_tree_len"], device,
    )
    mu_val, y_val = _encode(
        val_df, encoder, vocab, cfg["model"]["max_tree_len"], device,
    )

    bio_cfg = BioConfig(
        learning_rate=cfg["bioactivity"]["learning_rate"],
        weight_decay=cfg["bioactivity"]["weight_decay"],
        epochs=cfg["bioactivity"]["epochs"],
        batch_size=cfg["bioactivity"]["batch_size"],
    )

    ensemble = BioactivityEnsemble(
        n_members=cfg["bioactivity"]["n_members"],
        latent_dim=cfg["model"]["latent_dim"],
        hidden=cfg["bioactivity"]["hidden_dim"],
        dropout=cfg["bioactivity"]["dropout"],
    )

    member_dir = ckpt_dir / "bioactivity"
    member_dir.mkdir(parents=True, exist_ok=True)
    maes = []
    for i in range(cfg["bioactivity"]["n_members"]):
        member = BioactivityMLP(
            latent_dim=cfg["model"]["latent_dim"],
            hidden=cfg["bioactivity"]["hidden_dim"],
            dropout=cfg["bioactivity"]["dropout"],
        )
        seed = cfg["seed"] + i
        state, mae = train_bioactivity_member(
            member, mu_train, y_train, mu_val, y_val, device, bio_cfg, seed,
        )
        torch.save(state, member_dir / f"member_{i}.pth")
        ensemble.members[i].load_state_dict(state)
        maes.append(mae)
        print(f"  member {i+1}/{cfg['bioactivity']['n_members']}  val MAE={mae:.3f}")

    ensemble.to(device).eval()
    with torch.no_grad():
        _, val_std = ensemble(mu_val.to(device))
    val_std_np = val_std.cpu().numpy()
    p90 = float(np.percentile(val_std_np, 90))
    p95 = float(np.percentile(val_std_np, 95))
    print(f"Ensemble val MAE = {np.mean(maes):.3f} +/- {np.std(maes):.3f}")
    print(f"Val ensemble-std 90th pctile = {p90:.3f}  95th = {p95:.3f}")

    summary_path = ckpt_dir / "bioactivity_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "val_mae_per_member": maes,
                "uncertainty_percentile_90": p90,
                "uncertainty_percentile_95": p95,
            },
            f,
        )
    print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bioactivity_v3_aug.yaml")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
