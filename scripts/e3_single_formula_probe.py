"""Diagnostic 2: one formula, fresh queries every step (memorisation is impossible)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from expr_lab.data import TargetTransform
from expr_lab.e3 import E3Config, PoolItem, Vocab, make_batch, predict, to_device
from expr_lab.exprgen import count_ops, ops_in, parse, to_prefix_tokens, constants_in_order
from expr_lab.metrics import regression_metrics
from expr_lab.models import FormulaTransformer, count_params
from expr_lab.train import resolve_device


def item_of(text: str, vocab: Vocab) -> PoolItem:
    node = parse(text)
    ids, consts = vocab.encode(node)
    return PoolItem(
        text=text,
        node=node,
        ids=ids,
        consts=consts,
        bucket=1,
        n_ops=count_ops(node),
        has_mod="%" in ops_in(node),
        has_sqrt="sqrt" in ops_in(node),
    )


def run(text: str, steps: int = 800, k: int = 32, batch: int = 8, d_model: int = 128, layers: int = 4) -> None:
    dev = resolve_device("auto")
    cfg = E3Config(k_queries=k, batch_formulas=batch, n_vars=1)
    vocab = Vocab(cfg.var_names)
    item = item_of(text, vocab)
    rng = np.random.default_rng(0)
    probe = make_batch([item] * 8, rng, cfg, TargetTransform(log=True))
    ys = probe.y_raw[np.isfinite(probe.y_raw)]
    transform = TargetTransform(log=True).fit(ys)

    model = FormulaTransformer(len(vocab), d_model=d_model, n_layers=layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    print(f"\n=== {text} | params {count_params(model)/1e3:.0f}k")
    for step in range(1, steps + 1):
        b = to_device(make_batch([item] * batch, rng, cfg, transform), dev)
        model.train()
        z = model(b.tokens, b.values, b.kind, b.pad, b.pos, b.mask, b.num)[:, 1 : 1 + k]
        loss = ((z - b.y) ** 2)[b.valid].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            grid = np.linspace(-2, 2, 64).reshape(1, -1, 1)
            gb = make_batch([item], rng, cfg, transform, xs=grid)
            pred = transform.inverse(predict(model, gb, dev).astype(np.float64))[0]
            m = regression_metrics(gb.y_raw[0], pred)
            print(f"  step {step:>4} train {loss.item():.4f} | grid r2 {m['r2']:+.3f} nrmse {m['nrmse']:.3f}")


if __name__ == "__main__":
    for text in ["x + 2", "2 * x", "x * x", "x % 2"]:
        run(text)

