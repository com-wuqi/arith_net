"""Print the result tables used in README.md as markdown.

    uv run python scripts/summarize.py e1
    uv run python scripts/summarize.py e2
    uv run python scripts/summarize.py e3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import sys
from pathlib import Path

import numpy as np

RUNS = Path("runs")


def _load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def preset(name: str) -> None:
    rows = _load(RUNS / name / "summary.csv")
    variants = sorted({r["variant"] for r in rows})
    print("| model | R2 in | R2 out | medAPE out | formulas with R2_out > 0.5 |")
    print("|---|---|---|---|---|")

    def med(variant: str, split: str, key: str) -> float:
        vals = [_f(r[key]) for r in rows if r["variant"] == variant and r["split"] == split]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    def n_good(variant: str) -> int:
        vals = [_f(r["r2"]) for r in rows if r["variant"] == variant and r["split"] == "test_out"]
        return int(sum(1 for v in vals if np.isfinite(v) and v > 0.5))

    for v in variants:
        print(
            f"| `{v}` | {med(v, 'test_in', 'r2'):.3f} | {med(v, 'test_out', 'r2'):+.3f} "
            f"| {med(v, 'test_out', 'med_ape'):.3f} | {n_good(v)} |"
        )


def e3(name: str = "e3") -> None:
    run = RUNS / name
    metrics = _load(run / "metrics.csv")
    groups: dict[str, list[dict]] = {}
    for r in metrics:
        for key in (f"b{r['bucket']}-{r['domain']}", f"{'mod' if r['has_mod'] == 'True' else 'nomod'}-{r['domain']}"):
            groups.setdefault(key, []).append(r)

    print("| group | n | median R2 | median NRMSE | frac R2 > 0.9 |")
    print("|---|---|---|---|---|")
    for key in sorted(groups):
        rows = groups[key]
        r2 = [v for v in (_f(r["r2"]) for r in rows) if np.isfinite(v)]
        nrmse = [v for v in (_f(r["nrmse"]) for r in rows) if np.isfinite(v)]
        print(
            f"| {key} | {len(rows)} | {np.median(r2) if r2 else float('nan'):+.3f} "
            f"| {np.median(nrmse) if nrmse else float('nan'):.3f} "
            f"| {np.mean([v > 0.9 for v in r2]) if r2 else float('nan'):.2f} |"
        )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "e1"
    if which in ("e1", "e2"):
        preset(which)
    else:
        e3(which)

