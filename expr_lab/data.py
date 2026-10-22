"""Dataset construction: sampling, splits, target transforms.

Contract
--------
* Inputs are scaled by a *fixed* constant (the training range), never by
  data statistics, so that the out-of-domain split really is out of domain.
* Points where the expression is undefined (NaN/inf) are kept in the arrays
  with ``valid=False``; the trainer masks them out of the loss.
* Metrics are always reported in the *original* output space, after inverting
  any target transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exprgen import Formula, Node, evaluate, vars_in


@dataclass
class TargetTransform:
    """Optional ``sign(y) * log1p(|y|)`` compression plus standardization."""

    log: bool = False
    mean: float = 0.0
    std: float = 1.0

    def compress(self, y: np.ndarray) -> np.ndarray:
        if not self.log:
            return y
        return np.sign(y) * np.log1p(np.abs(y))

    def expand(self, z: np.ndarray) -> np.ndarray:
        if not self.log:
            return z
        return np.sign(z) * np.expm1(np.abs(z))

    def fit(self, y: np.ndarray) -> "TargetTransform":
        y = y[np.isfinite(y)]
        if y.size == 0:
            self.mean, self.std = 0.0, 1.0
            return self
        z = self.compress(y)
        self.mean = float(np.mean(z))
        std = float(np.std(z))
        self.std = std if std > 1e-12 else 1.0
        return self

    def forward(self, y: np.ndarray) -> np.ndarray:
        return (self.compress(y) - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return self.expand(z * self.std + self.mean)

    def as_dict(self) -> dict:
        return {"log": self.log, "mean": self.mean, "std": self.std}


@dataclass
class Split:
    x_raw: np.ndarray  # (n, d) float64, original units
    x: np.ndarray  # (n, d) float32, scaled by ``x_scale``
    y_raw: np.ndarray  # (n,) float64, original output units
    y: np.ndarray  # (n,) float32, transformed + standardized
    valid: np.ndarray  # (n,) bool

    @property
    def valid_frac(self) -> float:
        return float(self.valid.mean())

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    def subset(self, mask: np.ndarray) -> "Split":
        return Split(
            x_raw=self.x_raw[mask],
            x=self.x[mask],
            y_raw=self.y_raw[mask],
            y=self.y[mask],
            valid=self.valid[mask],
        )


@dataclass
class DatasetConfig:
    n_train: int = 200_000
    n_val: int = 20_000
    n_in: int = 20_000
    n_out: int = 20_000
    train_hi: float = 2.0
    domain_hi: float = 3.0
    out_lo: float = 2.0
    out_hi: float = 4.0
    log_target: bool = False


@dataclass
class FormulaData:
    formula: Formula
    var_names: list[str]
    x_scale: float
    transform: TargetTransform
    train: Split
    val: Split
    test_in: Split
    test_out: Split
    cfg: DatasetConfig = field(default_factory=DatasetConfig)

    def as_dict(self) -> dict:
        return {
            "formula": self.formula.as_dict(),
            "var_names": self.var_names,
            "x_scale": self.x_scale,
            "transform": self.transform.as_dict(),
            "dataset": {
                "n_train": int(len(self.train.x)),
                "n_val": int(len(self.val.x)),
                "n_in": int(len(self.test_in.x)),
                "n_out": int(len(self.test_out.x)),
                "valid_frac_train": self.train.valid_frac,
                "valid_frac_in": self.test_in.valid_frac,
                "valid_frac_out": self.test_out.valid_frac,
            },
        }


def _sample_uniform(
    node: Node,
    n: int,
    rng: np.random.Generator,
    var_names: list[str],
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(lo, hi, size=(n, len(var_names)))
    env = {name: x[:, i] for i, name in enumerate(var_names)}
    return x, evaluate(node, env)


def _sample_two_sided(
    node: Node,
    n: int,
    rng: np.random.Generator,
    var_names: list[str],
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    mag = rng.uniform(lo, hi, size=(n, len(var_names)))
    sign = rng.choice([-1.0, 1.0], size=(n, len(var_names)))
    x = mag * sign
    env = {name: x[:, i] for i, name in enumerate(var_names)}
    return x, evaluate(node, env)


def _to_split(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    x_scale: float,
    transform: TargetTransform,
) -> Split:
    x = (x_raw / x_scale).astype(np.float32)
    valid = np.isfinite(y_raw)
    y = np.full(y_raw.shape, np.nan, dtype=np.float64)
    y[valid] = transform.forward(y_raw[valid])
    return Split(
        x_raw=x_raw,
        x=x,
        y_raw=y_raw,
        y=y.astype(np.float32),
        valid=valid,
    )


def build_dataset(formula: Formula, cfg: DatasetConfig, seed: int = 0) -> FormulaData:
    rng = np.random.default_rng(seed)
    names = sorted(vars_in(formula.node)) or ["x"]
    node = formula.node

    x_tr, y_tr = _sample_uniform(node, cfg.n_train, rng, names, -cfg.domain_hi, cfg.domain_hi)
    # Training only sees the interior of the domain; the outer shell is held out
    # so that "extrapolation" means something.
    keep = np.all(np.abs(x_tr) <= cfg.train_hi, axis=1)
    x_tr, y_tr = x_tr[keep], y_tr[keep]
    x_val, y_val = _sample_uniform(node, cfg.n_val, rng, names, -cfg.train_hi, cfg.train_hi)
    x_in, y_in = _sample_uniform(node, cfg.n_in, rng, names, -cfg.train_hi, cfg.train_hi)
    x_out, y_out = _sample_two_sided(node, cfg.n_out, rng, names, cfg.out_lo, cfg.out_hi)

    transform = TargetTransform(log=cfg.log_target).fit(y_tr)
    x_scale = cfg.train_hi
    return FormulaData(
        formula=formula,
        var_names=names,
        x_scale=x_scale,
        transform=transform,
        train=_to_split(x_tr, y_tr, x_scale, transform),
        val=_to_split(x_val, y_val, x_scale, transform),
        test_in=_to_split(x_in, y_in, x_scale, transform),
        test_out=_to_split(x_out, y_out, x_scale, transform),
        cfg=cfg,
    )


def dense_grid(node: Node, var_names: list[str], lo: float, hi: float, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic grid used for plotting slices of 1-D formulas."""
    x = np.linspace(lo, hi, n).reshape(-1, len(var_names))
    env = {name: x[:, i] for i, name in enumerate(var_names)}
    return x, evaluate(node, env)
