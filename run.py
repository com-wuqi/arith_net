"""Command line entry point.

Examples
--------
    uv run python run.py list
    uv run python run.py gen --seed 42 --tier dirty
    uv run python run.py train --preset e1 --quick
    uv run python run.py train --preset e2 --tags mod_basic,mod_peak
    uv run python run.py report --preset e1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from expr_lab.data import dense_grid
from expr_lab.experiments import sweep
from expr_lab.exprgen import FormulaSpec, count_ops, depth, evaluate, sample_formula
from expr_lab.presets import build_preset
from expr_lab.report import plot_preset_report, read_summary


def _cmd_gen(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = FormulaSpec(tier=args.tier, max_depth=args.depth, require_ops=("%",) if args.tier == "dirty" else ())
    formula = sample_formula(spec, args.seed)
    s = formula.stats
    print(f"formula ({formula.tier}, seed {args.seed}): {formula.text}")
    print(f"  ops={count_ops(formula.node)} depth={depth(formula.node)}")
    if s is not None:
        print(
            f"  valid={s.valid_frac:.2f} std={s.std:.3f} p99|y|={s.p99_abs:.3f} "
            f"y in [{s.min_valid:.2f}, {s.max_valid:.2f}]"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x, y = dense_grid(formula.node, ["x"], -4.0, 4.0, 4000)
    rng = np.random.default_rng(args.seed)
    xs = rng.uniform(-2, 2, 600)
    ys = evaluate(formula.node, {"x": xs})

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axvspan(-4, -2, color="0.9", zorder=0)
    ax.axvspan(2, 4, color="0.9", zorder=0)
    ax.plot(x[:, 0], y, color="black", linewidth=1.3, label="f(x)")
    ax.scatter(xs, ys, s=3, alpha=0.3, color="tab:blue", label="sample points")
    ax.set_title(f"{formula.tier} / seed {args.seed}: {formula.text}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    dest = out / f"gen-{formula.tier}-{args.seed}.png"
    fig.savefig(dest, dpi=130)
    print(f"  plot -> {dest}")


def _cmd_list(args: argparse.Namespace) -> None:
    for name in ("e1", "e2"):
        preset = build_preset(name)
        print(f"\npreset {name}  (tier={preset.tier}, models={','.join(preset.models)})")
        print(f"  optional: {','.join(preset.oracle_models) or '-'}")
        for f in preset.formulas:
            print(f"  {f.tag:<12} period={str(f.period):<6} {f.text}")


def _cmd_train(args: argparse.Namespace) -> None:
    preset = build_preset(args.preset, quick=args.quick)
    tags = args.tags.split(",") if args.tags else None
    sweep(
        preset,
        device=args.device,
        with_baselines=not args.no_baselines,
        tags=tags,
        out_root=Path(args.out),
    )


def _cmd_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out) / args.preset
    rows = read_summary(out_dir / "summary.csv")
    if not rows:
        raise SystemExit(f"no summary at {out_dir / 'summary.csv'}; run train first")
    plot_preset_report(rows, out_dir / "report.png")
    print(f"report -> {out_dir / 'report.png'}")


def _cmd_e3(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from expr_lab.e3 import E3Config, train_e3

    cfg = E3Config(
        n_vars=args.vars,
        steps=args.steps,
        batch_formulas=args.batch,
        d_model=args.d_model,
        n_layers=args.layers,
        pool_size=args.pool_size,
        eval_size=args.eval_size,
        k_queries=args.k,
        buckets=tuple(int(b) for b in args.buckets.split(",")),
        loss_weighting="none" if args.no_reweight else "per_formula",
        seed=args.seed,
    )
    if args.quick:
        cfg = replace(
            cfg,
            pool_size=min(cfg.pool_size, 800),
            eval_size=min(cfg.eval_size, 30),
            d_model=128,
            n_layers=4,
            batch_formulas=min(cfg.batch_formulas, 32),
        )
    train_e3(
        cfg,
        out_dir=Path(args.out) / (args.name or "e3"),
        device=args.device,
        force_pool=args.rebuild_pool,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen", help="sample and plot one random formula")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--tier", default="dirty", choices=["smooth", "dirty"])
    p_gen.add_argument("--depth", type=int, default=4)
    p_gen.add_argument("--out", default="runs/gen")
    p_gen.set_defaults(func=_cmd_gen)

    p_list = sub.add_parser("list", help="show the formula sets of each preset")
    p_list.set_defaults(func=_cmd_list)

    p_train = sub.add_parser("train", help="run a sweep")
    p_train.add_argument("--preset", default="e1", choices=["e1", "e2"])
    p_train.add_argument("--quick", action="store_true", help="small data / few steps")
    p_train.add_argument("--tags", default=None, help="comma separated formula tags")
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--no-baselines", action="store_true")
    p_train.add_argument("--out", default="runs")
    p_train.set_defaults(func=_cmd_train)

    p_report = sub.add_parser("report", help="re-plot an existing sweep")
    p_report.add_argument("--preset", default="e1", choices=["e1", "e2"])
    p_report.add_argument("--out", default="runs")
    p_report.set_defaults(func=_cmd_report)

    p_e3 = sub.add_parser("e3", help="train the general formula evaluator transformer")
    p_e3.add_argument("--quick", action="store_true")
    p_e3.add_argument("--steps", type=int, default=20000)
    p_e3.add_argument("--batch", type=int, default=48)
    p_e3.add_argument("--d-model", type=int, default=192)
    p_e3.add_argument("--layers", type=int, default=6)
    p_e3.add_argument("--vars", type=int, default=1, choices=[1, 2, 3])
    p_e3.add_argument("--k", type=int, default=64, help="query points per formula")
    p_e3.add_argument("--buckets", default="1,2,3", help="comma separated depth buckets")
    p_e3.add_argument("--no-reweight", action="store_true", help="disable per-formula loss weighting")
    p_e3.add_argument("--pool-size", type=int, default=6000)
    p_e3.add_argument("--eval-size", type=int, default=120)
    p_e3.add_argument("--seed", type=int, default=0)
    p_e3.add_argument("--device", default="auto")
    p_e3.add_argument("--rebuild-pool", action="store_true")
    p_e3.add_argument("--name", default="e3", help="subdirectory under --out")
    p_e3.add_argument("--out", default="runs")
    p_e3.set_defaults(func=_cmd_e3)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
