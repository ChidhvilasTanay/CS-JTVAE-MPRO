"""Train the Junction-Tree VAE on the COVID Moonshot dataset.

Usage::

    python scripts/train_jtvae.py --config configs/jtvae_v3_aug.yaml

Outputs:

* ``checkpoints/vocab.txt`` -- cluster vocabulary
* ``checkpoints/jtvae_best.pth`` -- best-validation encoder + decoder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charaka.data import load_moonshot, split  # noqa: E402
from charaka.dataset import JTreeDataset, collate  # noqa: E402
from charaka.model import JTreeDecoder, JTreeEncoder, JTreeVAE  # noqa: E402
from charaka.tokenize import MAX_TREE_LEN  # noqa: E402
from charaka.training import VAEConfig, eval_epoch, train_epoch  # noqa: E402
from charaka.vocab import Vocabulary  # noqa: E402


def main(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_moonshot(cfg["dataset"]["csv_path"])
    train_df, val_df = split(
        df, test_size=cfg["dataset"]["val_fraction"], seed=cfg["seed"],
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = ckpt_dir / "vocab.txt"

    if cfg["vocab"].get("use_prebuilt_vocab", False) and vocab_path.exists():
        # Use the build-vocab output (includes the ZINC sample). Rebuilding here
        # would silently drop ZINC and shrink the vocab -- the v3 confound.
        vocab = Vocabulary.load(vocab_path)
        print(f"Loaded prebuilt vocabulary: {len(vocab)} tokens <- {vocab_path}")
    else:
        print("Building vocabulary...")
        vocab = Vocabulary.from_smiles(
            train_df["canonical"].tolist(),
            min_freq=cfg["vocab"]["min_freq"],
        )
        print(f"Vocabulary size: {len(vocab)}")
        vocab.save(vocab_path)

    print("Building datasets...")
    max_len = cfg["model"]["max_tree_len"]
    train_ds = JTreeDataset(train_df, vocab, max_len)
    val_ds = JTreeDataset(val_df, vocab, max_len)
    print(f"Sequences: train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    encoder = JTreeEncoder(
        vocab_size=len(vocab),
        pad_idx=vocab.pad,
        latent_dim=cfg["model"]["latent_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        n_layers=cfg["model"]["encoder_layers"],
        n_heads=cfg["model"]["n_heads"],
        max_len=max_len,
    )
    decoder = JTreeDecoder(
        vocab_size=len(vocab),
        pad_idx=vocab.pad,
        sos_idx=vocab.sos,
        eos_idx=vocab.eos,
        latent_dim=cfg["model"]["latent_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        n_layers=cfg["model"]["decoder_layers"],
        n_heads=cfg["model"]["n_heads"],
        max_len=max_len,
    )
    model = JTreeVAE(encoder, decoder).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    vae_cfg = VAEConfig(
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_clip=cfg["training"]["grad_clip"],
        free_bits=cfg["training"]["free_bits"],
        warmup_epochs=cfg["training"]["warmup_epochs"],
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=vae_cfg.learning_rate,
        weight_decay=vae_cfg.weight_decay,
    )

    best_ce = float("inf")
    no_improve = 0
    best_path = ckpt_dir / "jtvae_best.pth"

    for epoch in range(1, vae_cfg.epochs + 1):
        beta = min(1.0, epoch / vae_cfg.warmup_epochs)
        t_loss, t_ce, t_kl = train_epoch(
            model, train_loader, optimizer, device, vocab.pad, beta, vae_cfg,
        )
        v_loss, v_ce, v_kl = eval_epoch(
            model, val_loader, device, vocab.pad, beta, vae_cfg,
        )
        print(
            f"[FT {epoch:02d}/{vae_cfg.epochs}] "
            f"TRAIN ce={t_ce:.4f} kl={t_kl:.4f} | "
            f"VAL ce={v_ce:.4f} kl={v_kl:.4f}  beta={beta:.2f}"
        )

        if epoch <= vae_cfg.warmup_epochs:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_ce": v_ce},
                best_path,
            )
            best_ce = min(best_ce, v_ce)
            continue
        if v_ce < best_ce - 1e-3:
            best_ce = v_ce
            no_improve = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_ce": v_ce},
                best_path,
            )
        else:
            no_improve += 1
            if no_improve >= vae_cfg.patience:
                print("Early stop.")
                break
    print(f"Best val CE: {best_ce:.4f}  -> {best_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jtvae_v3_aug.yaml")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
