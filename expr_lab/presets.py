"""Experiment presets: which formulas, which architectures, how much data.

E1 ``smooth``  (+ - * / sqrt only)
E2 ``dirty``   (adds ``%``, i.e. a discontinuous sawtooth)

Each preset mixes hand-picked formulas (readable, good for figures) with
randomly sampled ones (honest, prevents cherry-picking).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data import DatasetConfig
from .exprgen import Formula, FormulaSpec, formula_from_text, sample_formula
from .train import ModelConfig

SMOOTH_FORMULAS: list[tuple[str, str, float | None]] = [
    ("poly", "x*x*x - 3*x + 1", None),
    ("ratio", "(x*x + 1) / (x*x + 0.7)", None),
    ("pole", "1 / (x*x - 1.5)", None),
    ("sqrt", "sqrt(x*x + 1)", None),
    ("sqrtshift", "sqrt(x + 3) * (x - 1)", None),
    ("mix", "(x + 2) * sqrt(x + 2.5) / (x*x + 1)", None),
]

DIRTY_FORMULAS: list[tuple[str, str, float | None]] = [
    ("mod_basic", "x % 2", 2.0),
    ("mod_shift", "(x % 3) - 1.5", 3.0),
    ("mod_scale", "2 * (x % 1.5) - 1", 1.5),
    ("mod_sqrt", "sqrt(x % 2.5)", 2.5),
    ("mod_quad", "(x % 2) * (x % 2) - 1", 2.0),
    ("mod_peak", "1 / ((x % 2) + 0.3)", 2.0),
]

E1_MODELS = ("mlp_relu", "mlp_tanh", "fourier", "fourier_learned")
E2_MODELS = ("mlp_relu", "mlp_tanh", "fourier", "fourier_learned", "periodic_learned")
E2_ORACLE_MODELS = ("periodic_oracle",)


@dataclass
class Preset:
    name: str
    tier: str
    formulas: list[Formula]
    models: list[str]
    data: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    oracle_models: list[str] = field(default_factory=list)

    def models_for(self, formula: Formula) -> list[str]:
        models = list(self.models)
        if formula.period is not None:
            models += self.oracle_models
        return models


def _random_formulas(tier: str, n: int, start_seed: int, domain: tuple[float, float]) -> list[Formula]:
    spec = FormulaSpec(tier=tier, domain=domain, require_ops=("%",) if tier == "dirty" else ())
    out: list[Formula] = []
    seed = start_seed
    while len(out) < n:
        try:
            out.append(sample_formula(spec, seed))
        except RuntimeError:
            pass
        seed += 1
    return out


def build_preset(name: str, quick: bool = False, n_random: int = 6) -> Preset:
    if name not in ("e1", "e2"):
        raise ValueError(f"unknown preset {name!r}")

    tier = "smooth" if name == "e1" else "dirty"
    table = SMOOTH_FORMULAS if tier == "smooth" else DIRTY_FORMULAS
    models = list(E1_MODELS if tier == "smooth" else E2_MODELS)
    oracle = list(E2_ORACLE_MODELS if tier == "dirty" else [])

    data = DatasetConfig(
        n_train=40_000 if quick else 200_000,
        n_val=10_000 if quick else 20_000,
        n_in=10_000 if quick else 20_000,
        n_out=10_000 if quick else 20_000,
    )
    model = ModelConfig(
        steps=1500 if quick else 4000,
        eval_every=100 if quick else 200,
        batch=512,
        hidden=128 if quick else 256,
        layers=2,
        lr=1e-3,
    )

    formulas = [formula_from_text(text, tier, tag, period) for tag, text, period in table]
    if not quick:
        formulas += _random_formulas(tier, n_random, 1000, (-3.0, 3.0))
    return Preset(
        name=name,
        tier=tier,
        formulas=formulas,
        models=models,
        data=data,
        model=model,
        oracle_models=oracle,
    )
