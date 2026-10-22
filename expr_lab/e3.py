"""E3: one transformer that learns to *evaluate* arbitrary formulas.

Training and test formulas are disjoint.  The model sees the formula as a
prefix token sequence plus K query points, and must output f(x) for every
query.  Data is generated on the fly from a cached pool of formulas.

Layout of one sequence (queries first, so query positions never move)::

    <bos> <q>*K <sep> formula tokens (padded) <eos>

Everything is bidirectional: a query token attends to the whole formula.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import TargetTransform
from .exprgen import (
    Formula,
    FormulaSpec,
    Node,
    count_ops,
    evaluate,
    ops_in,
    parse,
    sample_formula,
    to_prefix_tokens,
    constants_in_order,
    vars_in,
)
from .metrics import regression_metrics
from .models import FormulaTransformer, block_query_attention, count_params
from .train import resolve_device

VOCAB_BASE = ["<pad>", "<bos>", "<eos>", "<sep>", "<q>", "#", "+", "-", "*", "/", "%", "sqrt"]
PAD, BOS, EOS, SEP, Q, HASH = 0, 1, 2, 3, 4, 5

BUCKET_MIN_OPS = {1: 1, 2: 2, 3: 3}
BUCKET_MAX_OPS = {1: 2, 2: 4, 3: 7}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class E3Config:
    n_vars: int = 1
    buckets: tuple[int, ...] = (1, 2, 3)
    pool_size: int = 6000  # per bucket
    eval_size: int = 120  # per bucket
    k_queries: int = 64
    d_model: int = 192
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: float = 4.0
    n_freqs: int = 8
    lr: float = 6e-4
    weight_decay: float = 0.01
    warmup: int = 200
    batch_formulas: int = 48
    steps: int = 6000
    eval_every: int = 400
    patience: int = 8
    grad_clip: float = 1.0
    in_hi: float = 2.0
    out_lo: float = 2.0
    out_hi: float = 4.0
    loss_weighting: str = "per_formula"  # "per_formula" | "none"
    seed: int = 0
    cache_dir: str = "runs/e3/cache"

    @property
    def var_names(self) -> tuple[str, ...]:
        return tuple("xyz"[: self.n_vars])


# --------------------------------------------------------------------------- #
# Vocabulary and pools
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, var_names: tuple[str, ...]):
        self.itos = VOCAB_BASE + [v for v in var_names if v not in VOCAB_BASE]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, node: Node) -> tuple[list[int], list[float]]:
        ids = [self.stoi[t] for t in to_prefix_tokens(node)]
        consts = constants_in_order(node)
        return ids, consts


@dataclass
class PoolItem:
    text: str
    node: Node
    ids: list[int]
    consts: list[float]
    bucket: int
    n_ops: int
    has_mod: bool
    has_sqrt: bool


def _spec_for_bucket(cfg: E3Config, bucket: int) -> FormulaSpec:
    return FormulaSpec(
        tier="dirty",
        var_names=cfg.var_names,
        domain=(-3.0, 3.0),
        max_depth=bucket,
        min_ops=BUCKET_MIN_OPS[bucket],
        max_ops=BUCKET_MAX_OPS[bucket],
        allow_affine=True,
        min_valid_frac=0.6,
        min_rel_std=0.02,
    )


def _make_pool(cfg: E3Config, bucket: int, size: int, seed0: int, vocab: Vocab) -> list[PoolItem]:
    spec = _spec_for_bucket(cfg, bucket)
    items: list[PoolItem] = []
    seed = seed0
    attempts = 0
    while len(items) < size and attempts < size * 12:
        attempts += 1
        try:
            f: Formula = sample_formula(spec, seed)
        except RuntimeError:
            seed += 1
            continue
        seed += 1
        ids, consts = vocab.encode(f.node)
        if len(ids) > 64:
            continue
        items.append(
            PoolItem(
                text=f.text,
                node=f.node,
                ids=ids,
                consts=consts,
                bucket=bucket,
                n_ops=count_ops(f.node),
                has_mod="%" in ops_in(f.node),
                has_sqrt="sqrt" in ops_in(f.node),
            )
        )
    return items


def build_pools(cfg: E3Config, vocab: Vocab, force: bool = False) -> tuple[dict[int, list[PoolItem]], dict[int, list[PoolItem]]]:
    """Deterministic, disk-cached formula pools (train / eval are disjoint)."""
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = f"pool-v{cfg.n_vars}-d{'-'.join(map(str, cfg.buckets))}-{cfg.pool_size}x{cfg.eval_size}"
    path = cache / f"{key}.json"

    payload: dict | None = None
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {"train": {}, "eval": {}}
        for bucket in cfg.buckets:
            train = _make_pool(cfg, bucket, cfg.pool_size, 0, vocab)
            held = _make_pool(cfg, bucket, cfg.eval_size, 500_000, vocab)
            payload["train"][str(bucket)] = [i.text for i in train]
            payload["eval"][str(bucket)] = [i.text for i in held]
            print(f"  bucket {bucket}: {len(train)} train / {len(held)} eval formulas")
        path.write_text(json.dumps(payload), encoding="utf-8")

    def hydrate(texts: list[str], bucket: int) -> list[PoolItem]:
        out = []
        for text in texts:
            node = parse(text)
            ids, consts = vocab.encode(node)
            out.append(
                PoolItem(
                    text=text,
                    node=node,
                    ids=ids,
                    consts=consts,
                    bucket=bucket,
                    n_ops=count_ops(node),
                    has_mod="%" in ops_in(node),
                    has_sqrt="sqrt" in ops_in(node),
                )
            )
        return out

    train = {b: hydrate(payload["train"][str(b)], b) for b in cfg.buckets}
    held = {b: hydrate(payload["eval"][str(b)], b) for b in cfg.buckets}
    return train, held


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
@dataclass
class Batch:
    tokens: torch.Tensor
    values: torch.Tensor
    kind: torch.Tensor
    pad: torch.Tensor
    pos: torch.Tensor
    y: torch.Tensor  # (B, K) transformed target
    valid: torch.Tensor  # (B, K) bool
    y_raw: np.ndarray
    x: np.ndarray  # (B, K, n_dims)
    mask: torch.Tensor
    num: torch.Tensor


def _sample_queries(rng: np.random.Generator, n: int, k: int, n_dims: int, cfg: E3Config, out_of_domain: bool) -> np.ndarray:
    if not out_of_domain:
        return rng.uniform(-cfg.in_hi, cfg.in_hi, size=(n, k, n_dims))
    mag = rng.uniform(cfg.out_lo, cfg.out_hi, size=(n, k, n_dims))
    sign = rng.choice([-1.0, 1.0], size=(n, k, n_dims))
    return mag * sign


def make_batch(
    items: list[PoolItem],
    rng: np.random.Generator,
    cfg: E3Config,
    transform: TargetTransform,
    out_of_domain: bool = False,
    xs: np.ndarray | None = None,
) -> Batch:
    n = len(items)
    k = cfg.k_queries
    d = cfg.n_vars
    if xs is not None:
        xs = np.asarray(xs, dtype=np.float64)
        n, k, d = xs.shape
        assert n == len(items)
    else:
        xs = _sample_queries(rng, n, k, d, cfg, out_of_domain)

    max_len = max(len(it.ids) for it in items)
    total = k + max_len + 3
    tokens = np.full((n, total), PAD, dtype=np.int64)
    values = np.zeros((n, total, d), dtype=np.float32)
    kind = np.zeros((n, total), dtype=np.int64)
    num = np.zeros((n, total), dtype=np.float32)
    pad = np.ones((n, total), dtype=bool)
    pos = np.zeros((n, total), dtype=np.int64)
    y_raw = np.full((n, k), np.nan, dtype=np.float64)

    for i, it in enumerate(items):
        tokens[i, 0] = BOS
        pad[i, 0] = False
        pos[i, 0] = 0

        # The query block is a *set*: every query shares one position id so the
        # model cannot depend on K or on the query order.
        tokens[i, 1 : 1 + k] = Q
        pad[i, 1 : 1 + k] = False
        kind[i, 1 : 1 + k] = 1
        values[i, 1 : 1 + k, :] = xs[i].astype(np.float32)
        num[i, 1 : 1 + k] = 1.0
        pos[i, 1 : 1 + k] = 1

        tokens[i, 1 + k] = SEP
        pad[i, 1 + k] = False
        pos[i, 1 + k] = 2

        start = 2 + k
        tokens[i, start : start + len(it.ids)] = it.ids
        pad[i, start : start + len(it.ids)] = False
        pos[i, start : start + len(it.ids)] = 3 + np.arange(len(it.ids))
        cidx = 0
        for j, tok in enumerate(it.ids):
            if tok == HASH:
                values[i, start + j, 0] = float(it.consts[cidx])
                kind[i, start + j] = 0
                num[i, start + j] = 1.0
                cidx += 1

        tokens[i, start + max_len] = EOS
        pad[i, start + max_len] = False
        pos[i, start + max_len] = 3 + max_len

        env = {name: xs[i][:, idx] for idx, name in enumerate(cfg.var_names)}
        y_raw[i] = evaluate(it.node, env)

    valid = np.isfinite(y_raw)
    y = np.zeros((n, k), dtype=np.float32)
    y[valid] = transform.forward(y_raw[valid]).astype(np.float32)

    pos = torch.as_tensor(pos)
    return Batch(
        tokens=torch.as_tensor(tokens),
        values=torch.as_tensor(values),
        kind=torch.as_tensor(kind),
        pad=torch.as_tensor(pad),
        pos=pos,
        mask=block_query_attention(k, max_len),
        num=torch.as_tensor(num),
        y=torch.as_tensor(y),
        valid=torch.as_tensor(valid),
        y_raw=y_raw,
        x=xs,
    )


def to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        tokens=batch.tokens.to(device),
        values=batch.values.to(device),
        kind=batch.kind.to(device),
        pad=batch.pad.to(device),
        pos=batch.pos.to(device),
        mask=batch.mask.to(device),
        num=batch.num.to(device),
        y=batch.y.to(device),
        valid=batch.valid.to(device),
        y_raw=batch.y_raw,
        x=batch.x,
    )


@torch.no_grad()
def predict(model: FormulaTransformer, batch: Batch, device: torch.device) -> np.ndarray:
    model.eval()
    b = to_device(batch, device)
    z = model(b.tokens, b.values, b.kind, b.pad, b.pos, b.mask, b.num)
    k = b.x.shape[1]
    return z[:, 1 : 1 + k].cpu().numpy()


# --------------------------------------------------------------------------- #
# Target transform fitted on the training pool
# --------------------------------------------------------------------------- #
def fit_transform(train_pool: dict[int, list[PoolItem]], cfg: E3Config, seed: int = 0) -> TargetTransform:
    rng = np.random.default_rng(seed)
    chunks = []
    for bucket_items in train_pool.values():
        pick = rng.choice(len(bucket_items), size=min(120, len(bucket_items)), replace=False)
        for idx in pick:
            item = bucket_items[idx]
            xs = rng.uniform(-cfg.in_hi, cfg.in_hi, (256, cfg.n_vars))
            env = {name: xs[:, i] for i, name in enumerate(cfg.var_names)}
            y = evaluate(item.node, env)
            chunks.append(y[np.isfinite(y)])
    return TargetTransform(log=True).fit(np.concatenate(chunks))


# --------------------------------------------------------------------------- #
# Curriculum
# --------------------------------------------------------------------------- #
def bucket_weights(buckets: tuple[int, ...], progress: float) -> np.ndarray:
    if len(buckets) == 1:
        return np.ones(1)
    anchors = np.array([[0.70, 0.30, 0.00], [0.30, 0.50, 0.20], [0.15, 0.35, 0.50]])
    if len(buckets) == 2:
        anchors = anchors[:, :2]
    points = np.linspace(0.0, 1.0, len(anchors))
    weights = np.array([np.interp(progress, points, anchors[:, j]) for j in range(len(buckets))])
    weights = np.clip(weights, 0.0, None)
    return weights / weights.sum()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_pool(
    model: FormulaTransformer,
    pool: dict[int, list[PoolItem]],
    transform: TargetTransform,
    cfg: E3Config,
    device: torch.device,
    max_per_bucket: int | None = None,
    out_of_domain: bool = False,
    batch_size: int = 32,
) -> list[dict]:
    rng = np.random.default_rng(1234)
    rows: list[dict] = []
    for bucket, items in pool.items():
        subset = items if max_per_bucket is None else items[:max_per_bucket]
        for start in range(0, len(subset), batch_size):
            chunk = subset[start : start + batch_size]
            batch = make_batch(chunk, rng, cfg, transform, out_of_domain=out_of_domain)
            z = predict(model, batch, device)
            pred = transform.inverse(z.astype(np.float64))
            for i, item in enumerate(chunk):
                m = regression_metrics(batch.y_raw[i], pred[i])
                rows.append(
                    {
                        "bucket": bucket,
                        "formula": item.text,
                        "n_ops": item.n_ops,
                        "has_mod": item.has_mod,
                        "has_sqrt": item.has_sqrt,
                        "domain": "out" if out_of_domain else "in",
                        **m,
                    }
                )
    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    nrmse = [r["nrmse"] for r in rows if np.isfinite(r["nrmse"])]
    r2 = [r["r2"] for r in rows if np.isfinite(r["r2"])]
    return {
        "n": len(rows),
        "median_nrmse": float(np.median(nrmse)) if nrmse else float("nan"),
        "median_r2": float(np.median(r2)) if r2 else float("nan"),
        "frac_r2_gt_0.9": float(np.mean([v > 0.9 for v in r2])) if r2 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class E3Result:
    model: FormulaTransformer
    history: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    summary: dict[str, dict] = field(default_factory=dict)
    seconds: float = 0.0
    best: dict = field(default_factory=dict)
    n_params: int = 0


def train_e3(
    cfg: E3Config,
    out_dir: Path = Path("runs/e3"),
    device: str = "auto",
    force_pool: bool = False,
    verbose: bool = True,
) -> E3Result:
    dev = resolve_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    vocab = Vocab(cfg.var_names)
    if verbose:
        print(f"[e3] device={dev} vocab={len(vocab)} buckets={cfg.buckets}")
    t_pool = time.perf_counter()
    train_pool, eval_pool = build_pools(cfg, vocab, force=force_pool)
    transform = fit_transform(train_pool, cfg, seed=cfg.seed)
    if verbose:
        print(f"[e3] pools ready in {time.perf_counter() - t_pool:.1f}s, target stats {transform.as_dict()}")

    model = FormulaTransformer(
        vocab_size=len(vocab),
        n_dims=cfg.n_vars,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ff_mult=cfg.ff_mult,
        n_freqs=cfg.n_freqs,
    ).to(dev)
    n_params = count_params(model)
    if verbose:
        print(f"[e3] model: {n_params / 1e6:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))

    def lr_at(step: int) -> float:
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    rng = np.random.default_rng(cfg.seed + 1)
    history: list[dict] = []
    best = {"nrmse": math.inf, "step": 0, "state": None}
    t0 = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step - 1)

        progress = (step - 1) / max(1, cfg.steps - 1)
        weights = bucket_weights(cfg.buckets, progress)
        bucket = int(rng.choice(list(cfg.buckets), p=weights))
        pool = train_pool[bucket]
        pick = rng.choice(len(pool), size=cfg.batch_formulas, replace=True)
        batch = to_device(make_batch([pool[i] for i in pick], rng, cfg, transform), dev)

        model.train()
        z = model(batch.tokens, batch.values, batch.kind, batch.pad, batch.pos, batch.mask, batch.num)[:, 1 : 1 + cfg.k_queries]
        diff = (z - batch.y) ** 2
        mask = batch.valid
        if not mask.any():
            continue
        if cfg.loss_weighting == "per_formula":
            # A formula like ``-12.9 + x`` has a large offset but tiny variance;
            # under a globally standardized target its loss is dominated by a
            # handful of large-magnitude formulas and the model just regresses
            # to the mean.  Weight each formula by 1/var so every formula gets
            # the same gradient budget.  Only used for training.
            cnt = mask.sum(1).clamp(min=1)
            filled = torch.where(mask, batch.y, torch.zeros_like(batch.y))
            mean = filled.sum(1) / cnt
            var = torch.where(mask, (batch.y - mean.unsqueeze(1)) ** 2, torch.zeros_like(batch.y)).sum(1) / cnt
            weight = 1.0 / (var + 0.1)
            per_formula = torch.where(mask, diff, torch.zeros_like(diff)).sum(1) / cnt
            loss = (weight * per_formula).sum() / weight.sum()
        else:
            loss = diff[mask].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            rows = evaluate_pool(model, eval_pool, transform, cfg, dev, max_per_bucket=40)
            per_bucket = {
                b: summarize([r for r in rows if r["bucket"] == b]) for b in cfg.buckets
            }
            overall = summarize(rows)
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "lr": lr_at(step - 1),
                    **{f"nrmse_b{b}": per_bucket[b]["median_nrmse"] for b in cfg.buckets},
                    **{f"r2_b{b}": per_bucket[b]["median_r2"] for b in cfg.buckets},
                    "median_nrmse": overall["median_nrmse"],
                    "median_r2": overall["median_r2"],
                }
            )
            if verbose:
                bstr = " ".join(
                    f"b{b}:{per_bucket[b]['median_nrmse']:.3f}" for b in cfg.buckets
                )
                print(
                    f"  step {step:>5} loss {loss.item():.4f} "
                    f"nrmse {bstr} | medR2 {overall['median_r2']:+.3f}"
                )
            if overall["median_nrmse"] < best["nrmse"]:
                best = {
                    "nrmse": overall["median_nrmse"],
                    "step": step,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                }
            elif step - best["step"] >= cfg.patience * cfg.eval_every:
                if verbose:
                    print(f"  early stop at step {step} (best {best['step']})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    metrics = evaluate_pool(model, eval_pool, transform, cfg, dev, out_of_domain=False)
    metrics += evaluate_pool(model, eval_pool, transform, cfg, dev, out_of_domain=True)
    summary = {
        f"{domain}_b{b}": summarize(
            [r for r in metrics if r["bucket"] == b and r["domain"] == domain]
        )
        for domain in ("in", "out")
        for b in cfg.buckets
    }
    summary["in_all"] = summarize([r for r in metrics if r["domain"] == "in"])
    summary["out_all"] = summarize([r for r in metrics if r["domain"] == "out"])
    summary["in_mod"] = summarize([r for r in metrics if r["domain"] == "in" and r["has_mod"]])
    summary["in_nomod"] = summarize([r for r in metrics if r["domain"] == "in" and not r["has_mod"]])
    summary["in_sqrt"] = summarize([r for r in metrics if r["domain"] == "in" and r["has_sqrt"]])
    summary["in_nosqrt"] = summarize([r for r in metrics if r["domain"] == "in" and not r["has_sqrt"]])

    result = E3Result(
        model=model,
        history=history,
        metrics=metrics,
        summary=summary,
        seconds=time.perf_counter() - t0,
        best={"step": best["step"], "nrmse": best["nrmse"]},
        n_params=n_params,
    )
    _save(result, cfg, transform, out_dir, vocab)
    try:
        plot_e3(model, eval_pool, transform, cfg, dev, history, out_dir / "summary.png")
    except Exception as exc:  # plotting must never kill a finished run
        print(f"  plot failed: {exc}")
    if verbose:
        _print_summary(summary)
        print(f"[e3] done in {result.seconds:.1f}s -> {out_dir}")
    return result


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _save(result: E3Result, cfg: E3Config, transform: TargetTransform, out_dir: Path, vocab: Vocab) -> None:
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "vocab": vocab.itos,
                "transform": transform.as_dict(),
                "n_params": result.n_params,
                "best": result.best,
                "summary": result.summary,
                "elapsed_seconds": result.seconds,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    torch.save(result.model.state_dict(), out_dir / "model.pt")

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result.history[0].keys()) if result.history else ["step"])
        writer.writeheader()
        writer.writerows(result.history)

    fields = ["bucket", "domain", "formula", "n_ops", "has_mod", "has_sqrt", "n", "r2", "nrmse", "mae", "med_ape", "p99_ape"]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.metrics)


def _print_summary(summary: dict[str, dict]) -> None:
    print(f"\n{'group':<12}{'n':>6}{'medR2':>9}{'medNRMSE':>11}{'R2>0.9':>9}")
    for key, s in summary.items():
        print(f"{key:<12}{s['n']:>6}{s['median_r2']:>9.3f}{s['median_nrmse']:>11.3f}{s['frac_r2_gt_0.9']:>9.2f}")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_e3(
    model: FormulaTransformer,
    eval_pool: dict[int, list[PoolItem]],
    transform: TargetTransform,
    cfg: E3Config,
    device: torch.device,
    history: list[dict],
    out_png: Path,
    n_per_bucket: int = 2,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])

    ax_hist = fig.add_subplot(gs[0, :])
    steps = [h["step"] for h in history]
    for b in cfg.buckets:
        ax_hist.plot(steps, [h.get(f"nrmse_b{b}", np.nan) for h in history], label=f"nrmse bucket {b}")
    ax_hist.plot(steps, [h.get("median_r2", np.nan) for h in history], color="black", linestyle="--", label="median R2")
    ax_hist.set_xlabel("step")
    ax_hist.set_title("held-out formulas: error over training")
    ax_hist.grid(alpha=0.25, linewidth=0.5)
    ax_hist.legend(fontsize=7, ncol=4)

    if cfg.n_vars == 1:
        grid = np.linspace(-cfg.out_hi, cfg.out_hi, 320).reshape(1, -1, 1)
        picks: list[PoolItem] = []
        for b in cfg.buckets:
            picks.extend(eval_pool[b][:n_per_bucket])
        picks = picks[:3]
        rng = np.random.default_rng(7)
        for col, item in enumerate(picks):
            ax = fig.add_subplot(gs[1, col])
            batch = make_batch([item], rng, cfg, transform, xs=grid)
            z = predict(model, batch, device)
            pred = transform.inverse(z.astype(np.float64))[0]
            ax.axvspan(-cfg.out_hi, -cfg.in_hi, color="0.9", zorder=0)
            ax.axvspan(cfg.in_hi, cfg.out_hi, color="0.9", zorder=0)
            ax.plot(grid[0, :, 0], batch.y_raw[0], "k--", linewidth=1.1, label="truth")
            ax.plot(grid[0, :, 0], pred, "tab:red", linewidth=1.4, label="pred")
            ax.set_title(f"d={item.bucket}: {item.text[:46]}", fontsize=8)
            ax.grid(alpha=0.25, linewidth=0.5)
            if col == 0:
                ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
