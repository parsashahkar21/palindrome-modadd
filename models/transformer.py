from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn


def _mask_to_bool(mask: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Normalize mask to boolean where True means "mask out".
    Accepts:
      - bool mask: (T,T), (B,T,T), (B,1,T,T)
      - float mask with -inf in masked positions: (T,T), (B,T,T), (B,1,T,T)
    Returns:
      - bool mask of shape (B,1,T,T) or (1,1,T,T) if mask had no batch.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # (1,T,T)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,T,T)
    if mask.dim() != 4:
        raise ValueError("mask must have shape (T,T), (B,T,T) or (B,1,T,T)")

    if mask.shape[-2:] != (seq_len, seq_len):
        raise ValueError(f"mask last dims must be ({seq_len},{seq_len})")

    if mask.dtype == torch.bool:
        return mask.to(device=device)
    # float mask: treat very negative values as masked
    return (mask.to(device=device) < -1e8)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(float(dropout))
        self.proj_drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must be (B,T,E)")
        B, T, E = x.shape
        if E != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {E}")

        qkv = self.qkv(x)  # (B,T,3E)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)

        if mask is not None:
            m = _mask_to_bool(mask, T, x.device)  # (B,1,T,T) or (1,1,T,T)
            if m.size(0) == 1 and B != 1:
                m = m.expand(B, -1, -1, -1)
            scores = scores.masked_fill(m, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,H,T,D)
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(int(embed_dim), int(mlp_dim))
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(mlp_dim), int(embed_dim))
        self.drop2 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        embed_dim = int(embed_dim)
        max_len = int(max_len)

        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1,max_len,E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must be (B,T,E)")
        T = x.size(1)
        return x + self.pe[:, :T, :].to(dtype=x.dtype, device=x.device)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(int(embed_dim))
        self.attn = MultiHeadAttention(int(embed_dim), int(num_heads), dropout=float(dropout))
        self.drop1 = nn.Dropout(float(dropout))

        self.norm2 = nn.LayerNorm(int(embed_dim))
        self.mlp = MLP(int(embed_dim), int(mlp_dim), float(dropout))
        self.drop2 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        pad_token: int,
        max_len: int,
        dropout: float,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.pad_token = int(pad_token)

        # per instructions: vocab_size excludes pad token
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embed_dim, padding_idx=self.pad_token)
        self.pos = PositionalEncoding(self.embed_dim, int(max_len))
        self.drop = nn.Dropout(float(dropout))

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(self.embed_dim, int(num_heads), int(mlp_dim), float(dropout)) for _ in range(int(num_layers))]
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("x must be (B,T)")
        if lengths.dim() != 1:
            raise ValueError("lengths must be (B,)")

        B, T = x.shape
        lengths = torch.clamp(lengths.to(device=x.device), min=0, max=T)

        # padding mask: True where PAD; used to mask keys for all queries
        pad_bt = (x == self.pad_token)  # (B,T)
        pad_attn = pad_bt.unsqueeze(1).unsqueeze(2).expand(B, 1, T, T)  # (B,1,T,T)

        if mask is not None:
            m = _mask_to_bool(mask, T, x.device)
            if m.size(0) == 1 and B != 1:
                m = m.expand(B, -1, -1, -1)
            attn_mask = pad_attn | m
        else:
            attn_mask = pad_attn

        h = self.embedding(x)
        h = self.pos(h)
        h = self.drop(h)

        for layer in self.layers:
            h = layer(h, mask=attn_mask)

        return h


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        pad_token: int,
        max_len: int,
        dropout: float,
    ):
        super().__init__()
        self.pad_token = int(pad_token)
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pad_token=pad_token,
            max_len=max_len,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(int(embed_dim))
        self.head = nn.Linear(int(embed_dim), int(num_classes))

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,  # ignored, for compatibility
        **kwargs,
    ) -> torch.Tensor:
        h = self.encoder(x, lengths, mask=mask)
        h = self.final_norm(h)

        # masked mean pooling
        nonpad = (x != self.pad_token).to(h.dtype)  # (B,T)
        denom = nonpad.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * nonpad.unsqueeze(-1)).sum(dim=1) / denom  # (B,E)

        logits = self.head(pooled)
        return logits
