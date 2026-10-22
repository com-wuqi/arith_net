"""Sweep orchestration: train models, run baselines, write artifacts."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .baselines import fit_baselines
from .data import DatasetConfig, FormulaData, build_dataset
from .exprgen import Formula
from .metrics import extrapolation_gap
from .presets import Preset
from .report import plot_history, plot_preset_report, plot_slice
from .train import ModelConfig, TrainResult, resolve_device, train_regressor

METRIC_FIELDS = [
    "variant",
    "kind",
    "formula",
    "split",
    "r2",
    "nrmse",
    "mae",
    "med_ape",
    "p99_ape",
    "p99_abs",
    "max_abs",
    "valid_frac",
    "n",
]


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rows_from_payload(formula: Formula, baselines: dict, results: dict[str, TrainResult]) -> list[dict]:
    rows: list[dict] = []
    for name, splits in baselines.items():
        for split, metrics in splits.items():
            rows.append({"variant": name, "kind": "baseline", "formula": formula.tag, "split": split, **metrics})
    for name, res in results.items():
        for split, metrics in res.metrics.items():
            rows.append({"variant": name, "kind": "model", "formula": formula.tag, "split": split, **metrics})
    return rows


def run_formula(
    preset: Preset,
    formula: Formula,
    device: torch.device,
    with_baselines: bool = True,
    make_plots: bool = True,
    out_root: Path = Path("runs"),
    data_cfg: DatasetConfig | None = None,
    verbose: bool = True,
) -> tuple[list[dict], FormulaData, dict[str, TrainResult]]:
    data_cfg = data_cfg or preset.data
    run_dir = out_root / preset.name / formula.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    data = build_dataset(formula, data_cfg, seed=0)
    if verbose:
        print(
            f"[{preset.name}/{formula.tag}] {formula.text}\n"
            f"    valid: train {data.train.valid_frac:.2f} in {data.test_in.valid_frac:.2f}"
            f" out {data.test_out.valid_frac:.2f}  |  y range in "
            f"[{np.nanmin(data.test_in.y_raw):.2f}, {np.nanmax(data.test_in.y_raw):.2f}]"
        )

    baselines: dict[str, dict] = {}
    if with_baselines:
        baselines = fit_baselines(
            data.train.x_raw,
            data.train.y_raw,
            {
                "test_in": (data.test_in.x_raw, data.test_in.y_raw),
                "test_out": (data.test_out.x_raw, data.test_out.y_raw),
            },
        )

    results: dict[str, TrainResult] = {}
    for model_name in preset.models_for(formula):
        cfg: ModelConfig = replace(preset.model, name=model_name)
        try:
            res = train_regressor(data, cfg, device=device, period=formula.period, verbose=False)
        except Exception as exc:  # keep the sweep alive, record the failure
            print(f"    !! {model_name} failed: {exc}")
            continue
        results[model_name] = res
        if verbose:
            m = res.metrics
            print(
                f"    {model_name:<17} R2 in {m['test_in']['r2']:>7.4f} "
                f"out {m['test_out']['r2']:>7.4f}  gap {extrapolation_gap(m['test_in'], m['test_out']):>7.3f}"
                f"  ({res.seconds:.1f}s, {res.n_params / 1e3:.0f}k params)"
            )

    rows = _rows_from_payload(formula, baselines, results)
    write_metrics_csv(run_dir / "metrics.csv", rows)

    payload = {
        "preset": preset.name,
        "tier": preset.tier,
        "formula": formula.as_dict(),
        "data": data.as_dict(),
        "models": {name: res.config for name, res in results.items()},
        "baselines": baselines,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    torch.save({name: res.model.state_dict() for name, res in results.items()}, run_dir / "models.pt")

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "step", "train_mse", "val_r2", "val_nrmse", "lr"])
        writer.writeheader()
        for name, res in results.items():
            for entry in res.history:
                writer.writerow({"model": name, **entry})

    if make_plots and len(data.var_names) == 1 and results:
        models = {name: res.model for name, res in results.items()}
        metrics = {name: res.metrics for name, res in results.items()}
        plot_slice(data, models, run_dir / "summary.png", metrics=metrics)
        plot_history({name: {"history": res.history} for name, res in results.items()}, run_dir / "history.png")

    return rows, data, results


def sweep(
    preset: Preset,
    device: str = "auto",
    with_baselines: bool = True,
    tags: list[str] | None = None,
    make_plots: bool = True,
    out_root: Path = Path("runs"),
) -> list[dict]:
    dev = resolve_device(device)
    formulas = [f for f in preset.formulas if tags is None or f.tag in tags]
    all_rows: list[dict] = []
    t0 = time.perf_counter()
    for i, formula in enumerate(formulas, 1):
        print(f"=== [{i}/{len(formulas)}] {formula.tag} ({preset.tier}) on {dev}")
        rows, _, _ = run_formula(
            preset,
            formula,
            dev,
            with_baselines=with_baselines,
            make_plots=make_plots,
            out_root=out_root,
        )
        all_rows.extend(rows)

    out_dir = out_root / preset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(out_dir / "summary.csv", all_rows)
    if make_plots and all_rows:
        plot_preset_report(all_rows, out_dir / "report.png")
    print(f"\ntotal {time.perf_counter() - t0:.1f}s, {len(all_rows)} metric rows -> {out_dir}")
    _print_table(all_rows)
    return all_rows


def _print_table(rows: list[dict]) -> None:
    variants = sorted({r["variant"] for r in rows})
    print(f"\n{'variant':<18}{'R2 in (med)':>13}{'R2 out (med)':>15}{'medAPE out (med)':>18}")
    for v in variants:
        def med(split: str, key: str) -> float:
            vals = [
                float(r[key])
                for r in rows
                if r["variant"] == v and r["split"] == split and np.isfinite(float(r[key]))
            ]
            return float(np.median(vals)) if vals else float("nan")

        print(f"{v:<18}{med('test_in', 'r2'):>13.4f}{med('test_out', 'r2'):>15.4f}{med('test_out', 'med_ape'):>18.4f}")
