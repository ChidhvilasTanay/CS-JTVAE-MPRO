"""Training loops for the JT-VAE, scorer joint training, and bioactivity ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import JTreeVAE


# ---------------------------------------------------------------------------
# JT-VAE training
# ---------------------------------------------------------------------------


@dataclass
class VAEConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    free_bits: float = 0.5
    warmup_epochs: int = 5
    epochs: int = 30
    patience: int = 8


def vae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pad_idx: int,
    beta: float,
    free_bits: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard JT-VAE loss: next-token CE + free-bits KL."""
    pred = logits[:, :-1, :].contiguous()
    tgt = target[:, 1:].contiguous()
    ce = F.cross_entropy(
        pred.view(-1, pred.size(-1)),
        tgt.view(-1),
        ignore_index=pad_idx,
    )
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = torch.mean(torch.clamp(kl_per_dim, min=free_bits))
    return ce + beta * kl, ce, kl


def train_epoch(
    model: JTreeVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_idx: int,
    beta: float,
    cfg: VAEConfig,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = total_ce = total_kl = 0.0
    n = 0
    for tokens, _ in loader:
        tokens = tokens.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(tokens)
        loss, ce, kl = vae_loss(
            logits, tokens, mu, logvar, pad_idx, beta, cfg.free_bits,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        total_ce += ce.item()
        total_kl += kl.item()
        n += 1
    return total_loss / n, total_ce / n, total_kl / n


@torch.no_grad()
def eval_epoch(
    model: JTreeVAE,
    loader: DataLoader,
    device: torch.device,
    pad_idx: int,
    beta: float,
    cfg: VAEConfig,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = total_ce = total_kl = 0.0
    n = 0
    for tokens, _ in loader:
        tokens = tokens.to(device)
        logits, mu, logvar = model(tokens)
        loss, ce, kl = vae_loss(
            logits, tokens, mu, logvar, pad_idx, beta, cfg.free_bits,
        )
        total_loss += loss.item()
        total_ce += ce.item()
        total_kl += kl.item()
        n += 1
    return total_loss / n, total_ce / n, total_kl / n


# ---------------------------------------------------------------------------
# Joint scorer + encoder training
# ---------------------------------------------------------------------------


@dataclass
class JointConfig:
    warmup_epochs: int = 5
    joint_epochs: int = 20
    scorer_lr: float = 1e-3
    encoder_lr: float = 1e-5
    lambda_recon: float = 1.0
    beta_kl: float = 0.1
    patience: int = 8


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _scorer_loss_for_example(
    scorer,
    example,
    encoder,
    device,
    pad_idx: int,
    train_encoder: bool,
    cfg: JointConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    from torch_geometric.data import Batch

    graphs, label, token_seq = example
    tokens = torch.tensor([token_seq], dtype=torch.long, device=device)
    if train_encoder:
        encoder.train()
    else:
        encoder.eval()
    z, mu, logvar = encoder(tokens)

    batch = Batch.from_data_list(graphs).to(device)
    junction_mask = batch.junction_mask.to(device)
    z_input = mu[0]
    z_repeat = z_input.unsqueeze(0).expand(len(graphs), -1).contiguous()
    scores = scorer(
        batch.x, batch.edge_index, batch.edge_attr,
        batch.batch, junction_mask, z_repeat,
    )
    target = torch.tensor([label], device=device)
    scorer_loss = F.cross_entropy(scores.unsqueeze(0), target)
    correct = int(scores.argmax().item() == label)
    return scorer_loss, mu, logvar, correct


def _scorer_loss_for_batch(
    scorer,
    examples: List,
    encoder,
    device,
    pad_idx: int,
    cfg: JointConfig,
) -> Tuple[torch.Tensor, int, int]:
    """Minibatched scorer loss over many examples (encoder frozen / eval).

    Batches the encoder token sequences and ALL candidate graphs across the
    examples in one scorer forward, while keeping each example's softmax over
    its own candidates isolated. Numerically equivalent to averaging
    ``_scorer_loss_for_example`` over the batch (validated in eval mode).
    Returns ``(mean_loss, correct_count, n_examples)``.
    """
    from torch_geometric.data import Batch

    encoder.eval()
    seqs = [ex[2] for ex in examples]
    lmax = max(len(s) for s in seqs)
    tokens = torch.full((len(seqs), lmax), pad_idx, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        tokens[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    _, mu, _ = encoder(tokens)                       # (B, latent)

    all_graphs: List = []
    ex_id: List[int] = []
    ks: List[int] = []
    labels: List[int] = []
    for i, (graphs, label, _) in enumerate(examples):
        all_graphs.extend(graphs)
        ex_id.extend([i] * len(graphs))
        ks.append(len(graphs))
        labels.append(label)

    batch = Batch.from_data_list(all_graphs).to(device)
    z_repeat = mu[torch.tensor(ex_id, dtype=torch.long, device=device)]  # (sum K_i, latent)
    scores = scorer(
        batch.x, batch.edge_index, batch.edge_attr,
        batch.batch, batch.junction_mask.to(device), z_repeat,
    )

    offsets = [0]
    for k in ks:
        offsets.append(offsets[-1] + k)
    losses, correct = [], 0
    for i in range(len(examples)):
        seg = scores[offsets[i]: offsets[i + 1]]
        losses.append(F.cross_entropy(seg.unsqueeze(0),
                                      torch.tensor([labels[i]], device=device)))
        correct += int(seg.argmax().item() == labels[i])
    return torch.stack(losses).mean(), correct, len(examples)


def joint_phase_batched(
    scorer,
    encoder: nn.Module,
    dataset: List,
    indices: List[int],
    device: torch.device,
    pad_idx: int,
    cfg: JointConfig,
    train: bool,
    scorer_opt: torch.optim.Optimizer,
    batch_size: int = 16,
) -> Tuple[float, float, float, float]:
    """Minibatched warm-up epoch (encoder frozen). Same signature shape as
    ``joint_phase`` (returns scorer/recon/kl/acc; recon/kl are 0 here)."""
    scorer.train() if train else scorer.eval()
    total_loss = total_correct = n = 0.0
    for b in tqdm(range(0, len(indices), batch_size), desc="train" if train else "val",
                  unit="batch", leave=False, mininterval=30.0):
        examples = [dataset[i] for i in indices[b: b + batch_size]]
        if train:
            scorer_opt.zero_grad()
            loss, correct, bn = _scorer_loss_for_batch(
                scorer, examples, encoder, device, pad_idx, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            scorer_opt.step()
        else:
            with torch.no_grad():
                loss, correct, bn = _scorer_loss_for_batch(
                    scorer, examples, encoder, device, pad_idx, cfg)
        total_loss += float(loss.item()) * bn
        total_correct += correct
        n += bn
    n = max(1.0, n)
    return total_loss / n, 0.0, 0.0, total_correct / n


def _recon_loss(
    decoder: nn.Module,
    encoder: nn.Module,
    token_seq,
    device,
    pad_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([token_seq], dtype=torch.long, device=device)
    z, mu, logvar = encoder(tokens)
    logits = decoder(z, tokens)
    target = tokens[:, 1:].reshape(-1)
    pred = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    nonpad = target != pad_idx
    return (
        F.cross_entropy(pred[nonpad], target[nonpad]),
        mu,
        logvar,
    )


def joint_phase(
    scorer,
    encoder: nn.Module,
    decoder: nn.Module,
    dataset: List,
    indices: List[int],
    device: torch.device,
    pad_idx: int,
    cfg: JointConfig,
    train: bool,
    train_encoder: bool,
    scorer_opt: torch.optim.Optimizer,
    encoder_opt: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float, float, float]:
    """One epoch over ``dataset`` indices in either warm-up or joint mode."""
    if train:
        scorer.train()
    else:
        scorer.eval()

    total_scorer = total_recon = total_kl = 0.0
    total_correct = 0
    n = 0

    _desc = "train" if train else "val"
    for idx in tqdm(indices, desc=_desc, unit="ex", leave=False,
                    mininterval=30.0, miniters=200):
        example = dataset[idx]
        if train:
            scorer_opt.zero_grad()
            if encoder_opt is not None:
                encoder_opt.zero_grad()
            scorer_loss, mu, logvar, correct = _scorer_loss_for_example(
                scorer, example, encoder, device, pad_idx, train_encoder, cfg,
            )
            if train_encoder:
                recon, _, _ = _recon_loss(
                    decoder, encoder, example[2], device, pad_idx,
                )
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                total = scorer_loss + cfg.lambda_recon * recon + cfg.beta_kl * kl
                total.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                if encoder_opt is not None:
                    encoder_opt.step()
                total_recon += float(recon.item())
                total_kl += float(kl.item())
            else:
                scorer_loss.backward()
            nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            scorer_opt.step()
        else:
            with torch.no_grad():
                scorer_loss, mu, logvar, correct = _scorer_loss_for_example(
                    scorer, example, encoder, device, pad_idx,
                    train_encoder=False, cfg=cfg,
                )
                if train_encoder:
                    recon, _, _ = _recon_loss(
                        decoder, encoder, example[2], device, pad_idx,
                    )
                    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                    total_recon += float(recon.item())
                    total_kl += float(kl.item())
        total_scorer += float(scorer_loss.item())
        total_correct += correct
        n += 1

    n = max(1, n)
    return (
        total_scorer / n,
        total_recon / n,
        total_kl / n,
        total_correct / n,
    )


# ---------------------------------------------------------------------------
# Bioactivity ensemble training
# ---------------------------------------------------------------------------


@dataclass
class BioConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 32


def train_bioactivity_member(
    member: nn.Module,
    mu_train: torch.Tensor,
    y_train: torch.Tensor,
    mu_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    cfg: BioConfig,
    seed: int = 42,
) -> Tuple[dict, float]:
    """Train one ensemble member; return ``(best_state_dict, best_val_mae)``."""
    torch.manual_seed(seed)
    member = member.to(device)
    optimizer = torch.optim.Adam(
        member.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    best_mae = float("inf")
    best_state: dict = {}
    n_train = len(mu_train)

    for _ in range(cfg.epochs):
        member.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i: i + cfg.batch_size]
            pred = member(mu_train[idx].to(device))
            loss = F.mse_loss(pred, y_train[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        member.eval()
        with torch.no_grad():
            if len(mu_val):
                pred_val = member(mu_val.to(device)).cpu()
                val_mae = (pred_val - y_val).abs().mean().item()
            else:
                val_mae = float("nan")

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in member.state_dict().items()}
    return best_state, best_mae
