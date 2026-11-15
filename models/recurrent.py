from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RecurrentCell(nn.Module):
    """
    A single recurrent cell implemented from scratch.

    We implement a GRU cell (no separate cell state).
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        # Compute r, z, n gates via two affine projections.
        self.x_proj = nn.Linear(self.input_dim, 3 * self.hidden_dim, bias=True)
        self.h_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim), h_prev: (B, hidden_dim)
        if x.dim() != 2 or h_prev.dim() != 2:
            raise ValueError("RecurrentCell expects 2D tensors for x and h_prev.")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"x last dim must be {self.input_dim}, got {x.size(-1)}")
        if h_prev.size(-1) != self.hidden_dim:
            raise ValueError(f"h_prev last dim must be {self.hidden_dim}, got {h_prev.size(-1)}")

        x_all = self.x_proj(x)
        h_all = self.h_proj(h_prev)

        x_r, x_z, x_n = x_all.chunk(3, dim=-1)
        h_r, h_z, h_n = h_all.chunk(3, dim=-1)

        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + r * h_n)

        h = (1.0 - z) * n + z * h_prev
        return h


class RecurrentEncoder(nn.Module):
    """
    Recurrent encoder with stacked GRU cells, optional bidirectionality,
    and effective-batch computation per time step (skip finished sequences).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        pad_token: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1")

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.pad_token = int(pad_token)
        self.bidirectional = bool(bidirectional)

        # IMPORTANT for palindrome: often pad_token == vocab_size
        # so we must allocate vocab_size + 1 embeddings.
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embed_dim, padding_idx=self.pad_token)

        self.drop = nn.Dropout(float(dropout))

        # Cells per layer (per direction)
        self.fw_cells = nn.ModuleList()
        self.bw_cells = nn.ModuleList() if self.bidirectional else None

        in_dim = self.embed_dim
        for _ in range(self.num_layers):
            self.fw_cells.append(RecurrentCell(in_dim, self.hidden_dim))
            if self.bidirectional:
                assert self.bw_cells is not None
                self.bw_cells.append(RecurrentCell(in_dim, self.hidden_dim))
                in_dim = 2 * self.hidden_dim
            else:
                in_dim = self.hidden_dim

    def _run_one_direction_one_layer(
        self,
        inputs: torch.Tensor,   # (B, T, D)
        lengths: torch.Tensor,  # (B,)
        cell: RecurrentCell,
        reverse: bool,
        dropout: Optional[nn.Dropout] = None,
    ) -> torch.Tensor:
        """
        Run a single GRU layer in one direction, returning all hidden states (B,T,H).
        Uses effective-batch: only update sequences with length > step.

        MPS-safe: avoids index_copy_ (not implemented on MPS).
        """
        B, T, _ = inputs.shape
        device, dtype = inputs.device, inputs.dtype

        h = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
        out = torch.zeros(B, T, self.hidden_dim, device=device, dtype=dtype)

        time_indices = range(T - 1, -1, -1) if reverse else range(T)

        for t in time_indices:
            step = (T - 1 - t) if reverse else t
            active = lengths > step
            if not torch.any(active):
                continue

            idx = torch.nonzero(active, as_tuple=False).squeeze(1)

            x_t = inputs.index_select(0, idx)[:, t, :]  # (B_active, D)
            h_prev = h.index_select(0, idx)
            h_new = cell(x_t, h_prev)

            # ✅ MPS-safe update (autograd-safe): write into a fresh clone
            h_next = h.clone()
            h_next[idx] = h_new
            h = h_next

            # store outputs
            out[idx, t, :] = h_new

        if dropout is not None:
            out = dropout(out)
        return out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) token ids
            lengths: (B,) valid lengths

        Returns:
            - if unidirectional: (B, T, H)
            - if bidirectional:  (B, T, 2H)
        """
        if x.dim() != 2:
            raise ValueError("x must be (Batch, Seq_Len)")
        if lengths.dim() != 1:
            raise ValueError("lengths must be (Batch,)")

        B, T = x.shape
        lengths = torch.clamp(lengths.to(device=x.device), min=0, max=T)

        h = self.embedding(x)  # (B,T,E)

        if not self.bidirectional:
            for layer in range(self.num_layers):
                h = self._run_one_direction_one_layer(
                    h, lengths, self.fw_cells[layer], reverse=False, dropout=None
                )
                if layer != self.num_layers - 1:
                    h = self.drop(h)
            return h  # (B,T,H)

        assert self.bw_cells is not None

        for layer in range(self.num_layers):
            fw = self._run_one_direction_one_layer(h, lengths, self.fw_cells[layer], reverse=False, dropout=None)
            bw = self._run_one_direction_one_layer(h, lengths, self.bw_cells[layer], reverse=True, dropout=None)
            h = torch.cat([fw, bw], dim=-1)  # (B,T,2H)
            if layer != self.num_layers - 1:
                h = self.drop(h)
        return h


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        pad_token: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.bidirectional = bool(bidirectional)

        self.encoder = RecurrentEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pad_token=pad_token,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        out_dim = (2 * hidden_dim) if bidirectional else hidden_dim
        self.classifier = nn.Linear(out_dim, int(num_classes))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, return_last_step_only: bool = True) -> torch.Tensor:
        enc = self.encoder(x, lengths)  # (B,T,H) or (B,T,2H)

        if not return_last_step_only:
            return self.classifier(enc)

        B, T, _ = enc.shape
        lengths = torch.clamp(lengths.to(device=enc.device), min=0, max=T)
        last_idx = torch.clamp(lengths - 1, min=0)
        bidx = torch.arange(B, device=enc.device)

        if not self.bidirectional:
            feats = enc[bidx, last_idx, :]  # (B,H)
            return self.classifier(feats)

        H = self.hidden_dim
        fw = enc[:, :, :H]
        bw = enc[:, :, H:]
        last_fw = fw[bidx, last_idx, :]
        first_bw = bw[bidx, torch.zeros_like(last_idx), :]
        feats = torch.cat([last_fw, first_bw], dim=-1)
        return self.classifier(feats)