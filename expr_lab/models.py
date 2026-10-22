"""Regressor architectures.

The point of the model zoo is the ablation: what changes when the network is
given *some* prior about periodicity?  From least to most structure:

``mlp_relu`` / ``mlp_tanh``
    plain MLP on the raw (scaled) input.
``fourier``
    fixed random Fourier features, frequencies ~ N(0, sigma^2) cycles per unit.
``fourier_learned``
    the same, but the frequencies are trainable.
``periodic_learned``
    sinusoidal features with a single learnable fundamental frequency and
    harmonics; every feature is a pure tone, so a sawtooth is representable.
``periodic_oracle``
    the same with the fundamental frequency *given* (the true period).
    This is an upper bound, not a fair competitor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_layers: int = 2, act: str = "relu", out_dim: int = 1):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(width, hidden))
            layers.append(act_cls())
            width = hidden
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FourierFeatures(nn.Module):
    """``[sin(2*pi*f.x), cos(2*pi*f.x)]`` with one frequency per coordinate."""

    def __init__(self, in_dim: int, n_features: int, sigma: float = 1.0, learnable: bool = False, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        freq = torch.randn(in_dim, n_features, generator=gen) * sigma
        if learnable:
            self.freq = nn.Parameter(freq)
        else:
            self.register_buffer("freq", freq)
        self.out_dim = 2 * n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (x @ self.freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PeriodicFeatures(nn.Module):
    """Harmonics of one fundamental frequency: sin/cos(2*pi*k*t), k=1..K."""

    def __init__(
        self,
        period: float,
        x_scale: float,
        harmonics: int = 8,
        learnable: bool = False,
        include_raw: bool = True,
    ):
        super().__init__()
        fundamental = x_scale / period  # cycles per *scaled* unit
        ks = torch.arange(1, harmonics + 1, dtype=torch.float32)
        base = ks * fundamental
        if learnable:
            self.freq = nn.Parameter(base.clone())
        else:
            self.register_buffer("freq", base)
        self.include_raw = include_raw
        self.out_dim = 2 * harmonics + (1 if include_raw else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (x * self.freq)
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.include_raw:
            feats = torch.cat([x, feats], dim=-1)
        return feats


class FeatureMLP(nn.Module):
    def __init__(self, features: nn.Module, hidden: int, n_layers: int, act: str, out_dim: int = 1):
        super().__init__()
        self.features = features
        self.net = MLP(features.out_dim, hidden, n_layers, act, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(x))


BUILTIN_MODELS = (
    "mlp_relu",
    "mlp_tanh",
    "fourier",
    "fourier_learned",
    "periodic_learned",
    "periodic_oracle",
)


def build_model(
    name: str,
    in_dim: int,
    x_scale: float,
    period: float | None = None,
    hidden: int = 256,
    layers: int = 2,
    fourier_features: int = 64,
    fourier_sigma: float = 1.0,
    harmonics: int = 8,
    seed: int = 0,
) -> nn.Module:
    if name == "mlp_relu":
        return MLP(in_dim, hidden, layers, "relu")
    if name == "mlp_tanh":
        return MLP(in_dim, hidden, layers, "tanh")
    if name == "fourier":
        feats = FourierFeatures(in_dim, fourier_features, fourier_sigma, learnable=False, seed=seed)
        return FeatureMLP(feats, hidden, layers, act="tanh")
    if name == "fourier_learned":
        feats = FourierFeatures(in_dim, fourier_features, fourier_sigma, learnable=True, seed=seed)
        return FeatureMLP(feats, hidden, layers, act="tanh")
    if name == "periodic_learned":
        base_period = period if period else 2.0
        feats = PeriodicFeatures(base_period, x_scale, harmonics, learnable=True)
        return FeatureMLP(feats, hidden, layers, act="tanh")
    if name == "periodic_oracle":
        if period is None:
            raise ValueError("periodic_oracle needs the true period")
        feats = PeriodicFeatures(period, x_scale, harmonics, learnable=False)
        return FeatureMLP(feats, hidden, layers, act="tanh")
    raise ValueError(f"unknown model {name!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Formula evaluator (E3): read an expression, evaluate it at query points
# --------------------------------------------------------------------------- #
def encode_scalar(values: torch.Tensor, n_freqs: int = 8) -> torch.Tensor:
    """NeRF-style encoding ``[v, sin(2^k pi v), cos(2^k pi v)]``."""
    freqs = (2.0 ** torch.arange(n_freqs, device=values.device, dtype=torch.float32)) * math.pi
    angles = values.unsqueeze(-1) * freqs
    return torch.cat([values.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)


class FormulaTransformer(nn.Module):
    """Sequence-to-sequence evaluator.

    Input is one sequence per formula::

        <bos> f1 f2 ... <pad> ... <sep> <q> <q> ... <eos>

    Formula tokens are ops / variables / a numeric slot ``#``; query tokens
    ``<q>`` carry the numeric coordinates.  The prediction for query ``i`` is
    read out of the hidden state at position ``i`` of the query block.
    """

    def __init__(
        self,
        vocab_size: int,
        n_dims: int = 1,
        d_model: int = 192,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        n_freqs: int = 8,
        max_len: int = 1024,
    ):
        super().__init__()
        enc_dim = 1 + 2 * n_freqs
        self.n_dims = n_dims
        self.n_freqs = n_freqs
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        # One projection per coordinate so the model can tell x from y.
        self.val = nn.ModuleList([nn.Linear(enc_dim, d_model) for _ in range(n_dims)])
        self.kind = nn.Embedding(2, d_model)  # 0 = constant, 1 = query
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(ff_mult * d_model),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        tokens: torch.Tensor,  # (B, T) long
        values: torch.Tensor,  # (B, T, n_dims) float
        kind: torch.Tensor,  # (B, T) long, 0/1
        pad_mask: torch.Tensor,  # (B, T) bool, True = ignore
        pos_ids: torch.Tensor | None = None,  # (B, T) long
        attn_mask: torch.Tensor | None = None,  # (T, T), True = blocked
        numeric: torch.Tensor | None = None,  # (B, T) float mask: token carries a value
    ) -> torch.Tensor:
        b, t = tokens.shape
        if pos_ids is None:
            pos_ids = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        pos_ids = pos_ids.clamp(max=self.pos.num_embeddings - 1)
        h = self.tok(tokens) + self.pos(pos_ids)

        # ``kind`` says *what* the number means (constant vs query point);
        # ``numeric`` says *whether* there is a number at all.  They must stay
        # separate: constants are kind=0 but still carry their value.
        carry = kind.float() if numeric is None else numeric
        enc = encode_scalar(values, self.n_freqs)  # (B, T, n_dims, enc_dim)
        coord = torch.zeros_like(h)
        for i in range(self.n_dims):
            coord = coord + self.val[i](enc[:, :, i, :])
        h = h + self.kind(kind) + carry.unsqueeze(-1) * coord

        h = self.encoder(h, mask=attn_mask, src_key_padding_mask=pad_mask)
        return self.head(self.norm(h)).squeeze(-1)


def block_query_attention(n_queries: int, max_len: int) -> torch.Tensor:
    """Keep queries independent of each other.

    Layout: ``<bos> <q>*K <sep> formula... <eos>``.  Query ``i`` may attend to
    the formula and to itself, but not to the other queries; formula tokens
    never attend to queries.  This makes the output a pure function of
    ``(formula, x_i)``, i.e. invariant to how many points are evaluated at once.
    """
    total = n_queries + max_len + 3
    idx = torch.arange(total)
    is_q = (idx >= 1) & (idx <= n_queries)
    is_f = idx >= n_queries + 1
    allow = torch.ones(total, total, dtype=torch.bool)
    q_ix = torch.nonzero(is_q).flatten()
    allow[q_ix[:, None], q_ix[None, :]] = torch.eye(len(q_ix), dtype=torch.bool)
    allow[is_f.unsqueeze(1) & is_q.unsqueeze(0)] = False
    mask = torch.full((total, total), float("-inf"))
    mask[allow] = 0.0
    return ~allow

