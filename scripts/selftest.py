"""Invariants that the experiment relies on. Run: uv run python scripts/selftest.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from expr_lab.data import DatasetConfig, TargetTransform, build_dataset
from expr_lab.e3 import EOS, HASH, Q, SEP, E3Config, PoolItem, Vocab, block_query_attention, make_batch
from expr_lab.exprgen import (
    FormulaSpec,
    count_ops,
    evaluate,
    ops_in,
    parse,
    sample_formula,
    to_string,
)
from expr_lab.presets import build_preset


def check(name: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    print("exprgen")
    x = np.linspace(-3, 3, 1001)

    # floor-mod, not fmod
    node = parse("x % 2")
    y = evaluate(node, {"x": x})
    check("mod is floor semantics (x=-0.5 -> 1.5)", abs(evaluate(node, {"x": np.array([-0.5])})[0] - 1.5) < 1e-12)
    check("mod is periodic with period 2", np.nanmax(np.abs(y - evaluate(node, {"x": x + 2}))) < 1e-9)

    # invalid points become NaN instead of 0 / inf
    check("sqrt of negative is NaN", np.isnan(evaluate(parse("sqrt(x)"), {"x": np.array([-1.0])})[0]))
    check("division by zero is NaN", np.isnan(evaluate(parse("1 / x"), {"x": np.array([0.0])})[0]))

    # text <-> AST round trip
    spec = FormulaSpec(tier="dirty", max_depth=4, require_ops=("%",))
    rt_bad = 0
    mod_bad = 0
    for seed in range(150):
        f = sample_formula(spec, seed)
        if parse(to_string(f.node)) != f.node:
            rt_bad += 1
        if "%" not in ops_in(f.node):
            mod_bad += 1
    check("150 dirty formulas: text round-trips exactly", rt_bad == 0)
    check("150 dirty formulas: all contain %", mod_bad == 0)

    # generator filters
    spec = FormulaSpec(tier="smooth", max_depth=4, allow_affine=False)
    tiny = 0
    for seed in range(100):
        f = sample_formula(spec, seed)
        if count_ops(f.node) < spec.min_ops or f.stats is None:
            tiny += 1
        if f.stats is not None and f.stats.std < spec.min_std:
            tiny += 1
    check("smooth formulas are non-degenerate", tiny == 0)

    print("data")
    preset = build_preset("e2", quick=True)
    data = build_dataset(preset.formulas[0], DatasetConfig(n_train=20_000, n_val=2_000, n_in=2_000, n_out=2_000))
    check("train is inside |x|<=2", np.abs(data.train.x_raw).max() <= 2.0 + 1e-9)
    check("test_in is inside |x|<=2", np.abs(data.test_in.x_raw).max() <= 2.0 + 1e-9)
    check("test_out is outside |x|>=2", np.abs(data.test_out.x_raw).min() >= 2.0 - 1e-9)
    check("inputs are scaled by the fixed train bound", np.abs(data.train.x).max() <= 1.0 + 1e-6)
    check("invalid points are masked, not zero-filled", bool(data.train.valid.any()) and np.isnan(data.train.y[~data.train.valid]).all())

    transform = data.transform
    probe = data.train.y_raw[data.train.valid][:100]
    check("target transform is invertible", np.allclose(transform.inverse(transform.forward(probe)), probe, atol=1e-9))

    print("e3")
    cfg = E3Config(pool_size=8, eval_size=2, k_queries=16, n_vars=1)
    vocab = Vocab(cfg.var_names)

    def make_item(text: str) -> PoolItem:
        node = parse(text)
        ids, consts = vocab.encode(node)
        return PoolItem(
            text=text,
            node=node,
            ids=ids,
            consts=consts,
            bucket=2,
            n_ops=count_ops(node),
            has_mod="%" in ops_in(node),
            has_sqrt="sqrt" in ops_in(node),
        )

    item = make_item("x % 2 + sqrt(x)")
    other = make_item("(x + 1) * (x % 3) - sqrt(x + 2) / 2")
    rng = np.random.default_rng(0)
    transform = TargetTransform(log=True).fit(np.linspace(-1, 1, 100))
    b16 = make_batch([item], rng, cfg, transform)
    b32 = make_batch([item], rng, E3Config(k_queries=32, n_vars=1), transform)
    check("query tokens sit right after <bos>", bool((b16.tokens[0, 1 : 1 + 16] == Q).all()))
    check(
        "query positions are a constant (independent of K)",
        bool((b16.pos[0, 1:17] == 1).all() and (b32.pos[0, 1:33] == 1).all()),
    )
    check("sep token follows the query block", int(b16.tokens[0, 1 + 16]) == SEP)
    check("eos is the last token", int(b16.tokens[0, -1]) == EOS)

    pair = make_batch([item, other], rng, cfg, transform)
    max_len = max(len(item.ids), len(other.ids))
    short = 0 if len(item.ids) < max_len else 1
    start = 2 + cfg.k_queries
    n_short = len([item, other][short].ids)
    check("unused formula slots are masked", bool(pair.pad[short, start + n_short : start + max_len].all()))

    # constants must reach the model as *numbers*, not as nameless tokens
    hash_positions = [j for j, tok in enumerate(item.ids) if tok == HASH]
    check("every constant slot carries a numeric value", bool(hash_positions) and bool(pair.num[0, start + hash_positions[0]] == 1.0))
    check(
        "constant numeric value equals the literal in the formula",
        bool(np.isclose(float(pair.values[0, start + hash_positions[0], 0]), item.consts[0])),
    )
    check("structural tokens carry no numeric value", float(pair.num[0, start]) == 0.0)

    mask = block_query_attention(16, len(item.ids))
    q = torch.arange(1, 17)
    eye = torch.eye(16, dtype=torch.bool)
    check("queries cannot attend to other queries", bool((mask[q[:, None], q[None, :]] == ~eye).all()))
    f_ix = torch.arange(17, mask.shape[0])
    check("formula tokens cannot attend to queries", bool(mask[f_ix[:, None], q[None, :]].all()))
    check("queries can attend to the formula", not bool(mask[1, f_ix].all()))
    check("every row has something to attend to", bool((~mask).any(dim=1).all()))

    print("\nall invariants hold")


if __name__ == "__main__":
    main()
