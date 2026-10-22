"""Metrics that stay meaningful when the target has poles and jumps."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {"n": 0, "r2": float("nan"), "nrmse": float("nan"), "med_ape": float("nan"), "p99_abs": float("nan")}

    err = y_pred - y_true
    var = float(np.var(y_true))
    ss_res = float(np.sum(err**2))
    r2 = 1.0 - ss_res / (var * y_true.size) if var > 1e-15 else float("nan")
    std = float(np.std(y_true)) or 1.0
    scale = abs(float(np.mean(y_true))) + std
    ape = np.abs(err) / (np.abs(y_true) + 0.01 * scale)
    return {
        "n": int(y_true.size),
        "r2": float(r2),
        "nrmse": float(np.sqrt(np.mean(err**2)) / std),
        "mae": float(np.mean(np.abs(err))),
        "med_ape": float(np.median(ape)),
        "p99_ape": float(np.percentile(ape, 99)),
        "p99_abs": float(np.percentile(np.abs(err), 99)),
        "max_abs": float(np.max(np.abs(err))),
    }


def extrapolation_gap(metrics_in: dict[str, float], metrics_out: dict[str, float]) -> float:
    r2_in, r2_out = metrics_in.get("r2", float("nan")), metrics_out.get("r2", float("nan"))
    if not (np.isfinite(r2_in) and np.isfinite(r2_out)):
        return float("nan")
    return float(r2_out - r2_in)


def format_metrics(metrics: dict[str, float]) -> str:
    return f"R2={metrics['r2']:.4f} nrmse={metrics['nrmse']:.3f} medAPE={metrics['med_ape']:.4f}"
