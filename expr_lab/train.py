"""Training loop for the single-formula regression task (E1 / E2)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F

from .data import FormulaData, Split
from .metrics import regression_metrics
from .models import build_model, count_params


@dataclass
class ModelConfig:
    name: str = "mlp_relu"
    hidden: int = 256
    layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 512
    steps: int = 4000
    eval_every: int = 200
    patience: int = 10
    fourier_features: int = 64
    fourier_sigma: float = 1.0
    harmonics: int = 8
    seed: int = 0


@dataclass
class TrainResult:
    model: torch.nn.Module
    history: list[dict] = field(default_factory=list)
    metrics: dict[str, dict] = field(default_factory=dict)
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    best_step: int = 0
    best_val_r2: float = float("nan")
    seconds: float = 0.0
    n_params: int = 0
    config: dict = field(default_factory=dict)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict(model: torch.nn.Module, data: FormulaData, split: Split, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(split.x, dtype=torch.float32, device=device)
        chunks = []
        for start in range(0, len(xb), 8192):
            z = model(xb[start : start + 8192]).cpu().numpy().astype(np.float64)
            chunks.append(data.transform.inverse(z))
    return np.concatenate(chunks) if chunks else np.zeros(0)


def _evaluate(model, data: FormulaData, device: torch.device) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ("val", "test_in", "test_out"):
        split: Split = getattr(data, name)
        pred = _predict(model, data, split, device)
        out[name] = regression_metrics(split.y_raw, pred)
        out[name]["valid_frac"] = split.valid_frac
    return out


def train_regressor(
    data: FormulaData,
    cfg: ModelConfig,
    device: torch.device | str = "auto",
    period: float | None = None,
    verbose: bool = False,
) -> TrainResult:
    device = torch.device(device) if isinstance(device, str) else device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    in_dim = data.train.x.shape[1]
    model = build_model(
        cfg.name,
        in_dim=in_dim,
        x_scale=data.x_scale,
        period=period,
        hidden=cfg.hidden,
        layers=cfg.layers,
        fourier_features=cfg.fourier_features,
        fourier_sigma=cfg.fourier_sigma,
        harmonics=cfg.harmonics,
        seed=cfg.seed,
    ).to(device)

    train = data.train
    x = torch.as_tensor(train.x[train.valid], dtype=torch.float32, device=device)
    y = torch.as_tensor(train.y[train.valid], dtype=torch.float32, device=device)
    if len(x) == 0:
        raise RuntimeError("training split has no valid points")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.steps, 1))

    history: list[dict] = []
    best = {"val_r2": -math.inf, "step": 0, "state": None}
    t0 = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        model.train()
        idx = torch.randint(0, len(x), (min(cfg.batch, len(x)),), device=device)
        pred = model(x[idx])
        loss = F.mse_loss(pred, y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = _evaluate(model, data, device)
            val_r2 = metrics["val"]["r2"]
            history.append(
                {
                    "step": step,
                    "train_mse": float(loss.item()),
                    "val_r2": val_r2,
                    "val_nrmse": metrics["val"]["nrmse"],
                    "lr": float(sched.get_last_lr()[0]),
                }
            )
            if verbose:
                print(
                    f"  step {step:>5} loss {loss.item():.5f} "
                    f"val_r2 {val_r2:.4f} in_r2 {metrics['test_in']['r2']:.4f} "
                    f"out_r2 {metrics['test_out']['r2']:.4f}"
                )
            if val_r2 > best["val_r2"]:
                best = {
                    "val_r2": val_r2,
                    "step": step,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                }
            elif step - best["step"] >= cfg.patience * cfg.eval_every:
                if verbose:
                    print(f"  early stop at step {step} (best {best['step']})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    metrics = _evaluate(model, data, device)
    predictions = {}
    for name in ("test_in", "test_out"):
        split: Split = getattr(data, name)
        predictions[name] = (split.x_raw, split.y_raw, _predict(model, data, split, device))

    return TrainResult(
        model=model,
        history=history,
        metrics=metrics,
        predictions=predictions,
        best_step=int(best["step"]),
        best_val_r2=float(best["val_r2"]),
        seconds=time.perf_counter() - t0,
        n_params=count_params(model),
        config={
            "model": asdict(cfg),
            "period": period,
            "n_params": count_params(model),
        },
    )
