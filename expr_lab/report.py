"""Figures.  Everything is written as PNG next to the metrics it illustrates."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import FormulaData, dense_grid
from .exprgen import Node


def _robust_ylim(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    pad = 0.15 * (hi - lo if hi > lo else 1.0)
    return (float(lo - pad), float(hi + pad))


def plot_slice(
    data: FormulaData,
    models: dict[str, "object"],
    out_png: Path,
    metrics: dict[str, dict[str, dict]] | None = None,
    n_grid: int = 3000,
) -> None:
    """True function plus one panel per trained model, extrapolation shaded."""
    node: Node = data.formula.node
    names = data.var_names
    span = data.cfg.out_hi
    x_grid, y_grid = dense_grid(node, names, -span, span, n_grid)
    x_scaled = (x_grid / data.x_scale).astype(np.float32)

    import torch

    n_models = len(models)
    ncols = 2
    nrows = int(np.ceil((n_models + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.2 * nrows), squeeze=False)
    ax_list = axes.ravel().tolist()

    ylim = _robust_ylim(y_grid)

    def decorate(ax, title: str) -> None:
        ax.axvspan(-span, -data.cfg.train_hi, color="0.9", zorder=0)
        ax.axvspan(data.cfg.train_hi, span, color="0.9", zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")

    ax0 = ax_list[0]
    sub = data.train.subset(np.isfinite(data.train.y_raw))
    if len(sub.x_raw) > 3000:
        pick = np.random.default_rng(0).choice(len(sub.x_raw), 3000, replace=False)
        sub = sub.subset(pick)
    ax0.plot(x_grid[:, 0], y_grid, color="black", linewidth=1.4, label="truth")
    ax0.scatter(sub.x_raw[:, 0], sub.y_raw, s=2, alpha=0.25, color="tab:blue", label="train")
    ax0.legend(fontsize=7, loc="best")
    decorate(ax0, "target")

    for i, (ax, (name, model)) in enumerate(zip(ax_list[1:], models.items())):
        model.eval()
        with torch.no_grad():
            z = model(torch.as_tensor(x_scaled, device=next(model.parameters()).device)).cpu().numpy()
        pred = data.transform.inverse(z.astype(np.float64))
        ax.plot(x_grid[:, 0], y_grid, color="black", linewidth=1.1, linestyle="--", alpha=0.75, label="truth")
        ax.plot(x_grid[:, 0], pred, color="tab:red", linewidth=1.5, label="pred")
        title = name
        if metrics and name in metrics:
            m = metrics[name]
            title = f"{name}\nR2 in={m['test_in']['r2']:.3f}  out={m['test_out']['r2']:.3f}"
        decorate(ax, title)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for ax in ax_list[n_models + 1 :]:
        ax.axis("off")

    fig.suptitle(f"{data.formula.tag}: {data.formula.text}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_history(results: dict[str, dict], out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for name, res in results.items():
        hist = res["history"]
        if not hist:
            continue
        steps = [h["step"] for h in hist]
        axes[0].plot(steps, [h["train_mse"] for h in hist], label=name)
        axes[1].plot(steps, [h["val_r2"] for h in hist], label=name)
    axes[0].set_title("train MSE")
    axes[1].set_title("val R2")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_preset_report(rows: list[dict], out_png: Path) -> None:
    """Median R2 inside vs outside the training range, per variant."""
    variants = sorted({r["variant"] for r in rows})
    formulas = sorted({r["formula"] for r in rows})

    def med(variant: str, split: str) -> float:
        vals = [
            float(r["r2"])
            for r in rows
            if r["variant"] == variant and r["split"] == split and np.isfinite(float(r["r2"]))
        ]
        return float(np.median(vals)) if vals else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    def bar_height(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, -1.05, 1.05))

    def label(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.1e}" if abs(value) >= 10 else f"{value:.2f}"

    width = 0.38
    xs = np.arange(len(variants))
    for offset, split, color, name in (
        (-width / 2, "test_in", "tab:blue", "in-domain"),
        (width / 2, "test_out", "tab:orange", "extrapolation"),
    ):
        values = [med(v, split) for v in variants]
        axes[0].bar(xs + offset, [bar_height(v) for v in values], width, label=name, color=color)
        for x, value in zip(xs + offset, values):
            axes[0].annotate(
                label(value),
                (x, bar_height(value)),
                ha="center",
                va="bottom" if bar_height(value) >= 0 else "top",
                fontsize=6,
                rotation=90,
            )
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(variants, rotation=20, ha="right")
    axes[0].set_ylabel("median R2 (clipped to [-1, 1])")
    axes[0].set_ylim(-1.25, 1.25)
    axes[0].set_title(f"median R2 over {len(formulas)} formulas")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, axis="y", linewidth=0.5)

    mat = np.full((len(variants), len(formulas)), np.nan)
    for i, v in enumerate(variants):
        for j, f in enumerate(formulas):
            vals = [
                float(r["r2"])
                for r in rows
                if r["variant"] == v and r["formula"] == f and r["split"] == "test_out"
            ]
            if vals:
                mat[i, j] = vals[0]
    im = axes[1].imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_yticks(range(len(variants)))
    axes[1].set_yticklabels(variants, fontsize=8)
    axes[1].set_xticks(range(len(formulas)))
    axes[1].set_xticklabels(formulas, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("extrapolation R2 per formula")
    fig.colorbar(im, ax=axes[1], fraction=0.03)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def read_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
