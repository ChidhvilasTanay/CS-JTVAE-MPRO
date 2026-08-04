"""Junction-tree VAE: transformer encoder + autoregressive decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JTreeEncoder(nn.Module):
    """Transformer encoder pooled to ``(mu, logvar)``."""

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        latent_dim: int = 256,
        emb_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        max_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc_mu = nn.Linear(emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(emb_dim, latent_dim)

    def forward(self, tokens: torch.Tensor):
        b, t = tokens.shape
        positions = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        emb = self.token_embed(tokens) + self.pos_embed(positions)
        pad_mask = tokens == self.pad_idx
        h = self.transformer(emb, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled).clamp(-10, 2)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return z, mu, logvar


class JTreeDecoder(nn.Module):
    """Causal transformer decoder conditioned on the latent code."""

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        sos_idx: int,
        eos_idx: int,
        latent_dim: int = 256,
        emb_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, emb_dim)
        self.input_proj = nn.Linear(emb_dim + latent_dim, emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(emb_dim, vocab_size)

    @staticmethod
    def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(t, t, device=device), diagonal=1).bool()

    def forward(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, t = target.shape
        positions = torch.arange(t, device=target.device).unsqueeze(0).expand(b, t)
        emb = self.token_embed(target) + self.pos_embed(positions)
        z_exp = z.unsqueeze(1).expand(-1, t, -1)
        h = self.input_proj(torch.cat([emb, z_exp], dim=-1))
        h = self.transformer(
            h,
            mask=self._causal_mask(t, target.device),
            src_key_padding_mask=(target == self.pad_idx),
        )
        return self.out(h)

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_len: int = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively decode a token sequence from ``z``."""
        ml = max_len or self.max_len
        b = z.size(0)
        out = torch.full((b, 1), self.sos_idx, dtype=torch.long, device=z.device)
        for _ in range(1, ml):
            logits = self.forward(z, out)[:, -1, :]
            if temperature == 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            out = torch.cat([out, next_tok], dim=1)
            if (next_tok.squeeze(-1) == self.eos_idx).all():
                break
        return out


class JTreeVAE(nn.Module):
    """Wrapper combining encoder + decoder."""

    def __init__(self, encoder: JTreeEncoder, decoder: JTreeDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, tokens: torch.Tensor):
        z, mu, logvar = self.encoder(tokens)
        logits = self.decoder(z, tokens)
        return logits, mu, logvar

    @torch.no_grad()
    def reconstruct(
        self, tokens: torch.Tensor, max_len: int = None, temperature: float = 0,
    ) -> torch.Tensor:
        """Encode then deterministically decode (argmax) from ``mu``."""
        self.eval()
        _, mu, _ = self.encoder(tokens)
        return self.decoder.generate(mu, max_len=max_len, temperature=temperature)

    @torch.no_grad()
    def sample(
        self, n: int, latent_dim: int, device, temperature: float = 0.8,
    ) -> torch.Tensor:
        """Sample ``z ~ N(0, I)`` and decode."""
        self.eval()
        z = torch.randn(n, latent_dim, device=device)
        return self.decoder.generate(z, temperature=temperature)
