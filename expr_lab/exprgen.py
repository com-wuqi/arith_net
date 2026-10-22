"""Random arithmetic expressions: AST, parser, generator, numeric evaluator.

Design notes
------------
* Modulo uses **floor semantics** (``numpy.mod``): for a positive modulus ``m``
  the result lies in ``[0, m)`` and the function is a sawtooth with jumps at
  ``x = k*m``.  This is deliberately *not* ``fmod`` (which follows the sign of
  the dividend); the choice is part of the task definition.
* Evaluation is done in float64 and any non-finite result (division by zero,
  sqrt of a negative number, overflow) becomes ``NaN``.  Callers should treat
  NaN as "point outside the domain" and mask it out of the loss, never fill 0.
* Expressions are only accepted if they are *interesting*: defined on a decent
  fraction of the domain, with a bounded output range and non-trivial variance.
  See :class:`FormulaSpec` for the thresholds.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

BINARY_OPS = ("+", "-", "*", "/", "%")
UNARY_FUNCS = ("sqrt",)
PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2}
ATOM_PREC = 3


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #
class Node:
    """Base class for expression nodes."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Num(Node):
    value: float


@dataclass(frozen=True, slots=True)
class Var(Node):
    name: str


@dataclass(frozen=True, slots=True)
class BinOp(Node):
    op: str
    left: Node
    right: Node


@dataclass(frozen=True, slots=True)
class UnOp(Node):
    op: str
    child: Node


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_num(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value - round(value)) < 1e-12 and abs(value) < 1e9:
        return str(int(round(value)))
    short = f"{value:g}"
    if float(short) == value:
        return short
    return repr(float(value))


def to_string(node: Node) -> str:
    """Render an AST as infix text with only the parentheses that are needed."""
    text = _render(node)
    return text


def _render(node: Node, parent_prec: int = 0, right_side: bool = False) -> str:
    if isinstance(node, Num):
        body = _fmt_num(node.value)
        if node.value < 0:
            body = f"({body})"
        return body
    if isinstance(node, Var):
        return node.name
    if isinstance(node, UnOp):
        return f"{node.op}({_render(node.child)})"
    if isinstance(node, BinOp):
        prec = PREC[node.op]
        left = _render(node.left, prec, False)
        right = _render(node.right, prec, True)
        body = f"{left} {node.op} {right}"
        # Left child needs parens only when it binds looser; right child also
        # when it binds equally tightly (subtraction / division are not
        # associative).
        need = parent_prec > prec or (right_side and parent_prec == prec)
        return f"({body})" if need else body
    raise TypeError(f"unknown node {node!r}")


def to_prefix_tokens(node: Node) -> list[str]:
    """Prefix (S-expression) token list, used later by the transformer model."""
    if isinstance(node, Num):
        return ["#"]  # numeric slot, value carried separately
    if isinstance(node, Var):
        return [node.name]
    if isinstance(node, UnOp):
        return [node.op, *to_prefix_tokens(node.child)]
    if isinstance(node, BinOp):
        return [node.op, *to_prefix_tokens(node.left), *to_prefix_tokens(node.right)]
    raise TypeError(node)


def constants_in_order(node: Node) -> list[float]:
    """Constants in prefix order (matches :func:`to_prefix_tokens`)."""
    if isinstance(node, Num):
        return [node.value]
    if isinstance(node, Var):
        return []
    if isinstance(node, UnOp):
        return constants_in_order(node.child)
    if isinstance(node, BinOp):
        return constants_in_order(node.left) + constants_in_order(node.right)
    raise TypeError(node)


# --------------------------------------------------------------------------- #
# Structure helpers
# --------------------------------------------------------------------------- #
def depth(node: Node) -> int:
    if isinstance(node, (Num, Var)):
        return 0
    if isinstance(node, UnOp):
        return 1 + depth(node.child)
    return 1 + max(depth(node.left), depth(node.right))


def count_ops(node: Node) -> int:
    if isinstance(node, (Num, Var)):
        return 0
    if isinstance(node, UnOp):
        return 1 + count_ops(node.child)
    return 1 + count_ops(node.left) + count_ops(node.right)


def vars_in(node: Node) -> set[str]:
    if isinstance(node, Var):
        return {node.name}
    if isinstance(node, Num):
        return set()
    if isinstance(node, UnOp):
        return vars_in(node.child)
    return vars_in(node.left) | vars_in(node.right)


def ops_in(node: Node) -> set[str]:
    if isinstance(node, (Num, Var)):
        return set()
    if isinstance(node, UnOp):
        return {node.op} | ops_in(node.child)
    return {node.op} | ops_in(node.left) | ops_in(node.right)


def is_affine(node: Node) -> bool:
    """True if the expression is a polynomial of degree <= 1 in the variables."""
    if isinstance(node, (Num, Var)):
        return True
    if isinstance(node, UnOp):
        return False
    if node.op in ("+", "-"):
        return is_affine(node.left) and is_affine(node.right)
    if node.op == "*":
        return (isinstance(node.left, Num) or isinstance(node.right, Num)) and (
            is_affine(node.left) and is_affine(node.right)
        )
    if node.op == "/":
        return isinstance(node.right, Num) and is_affine(node.left)
    return False


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(node: Node, env: Mapping[str, np.ndarray]) -> np.ndarray:
    """Evaluate with float64; non-finite results are returned as NaN."""
    with np.errstate(all="ignore"):
        out = _eval(node, env)
    out = np.asarray(out, dtype=np.float64)
    if out.ndim == 0:
        ones = next(iter(env.values()), np.zeros(1))
        out = np.full(ones.shape, float(out))
    return np.where(np.isfinite(out), out, np.nan)


def _eval(node: Node, env: Mapping[str, np.ndarray]):
    if isinstance(node, Num):
        return np.float64(node.value)
    if isinstance(node, Var):
        return env[node.name]
    if isinstance(node, UnOp):
        inner = _eval(node.child, env)
        if node.op == "sqrt":
            return np.sqrt(inner)
        raise ValueError(f"unknown function {node.op!r}")
    if isinstance(node, BinOp):
        left = _eval(node.left, env)
        right = _eval(node.right, env)
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right
        if node.op == "%":
            return np.mod(left, right)
        raise ValueError(f"unknown operator {node.op!r}")
    raise TypeError(f"unknown node {node!r}")


# --------------------------------------------------------------------------- #
# Simplification
# --------------------------------------------------------------------------- #
def _clean(value: float) -> float:
    """Round folded constants so rendered formulas stay readable."""
    if not math.isfinite(value):
        return value
    nearest = round(value)
    if abs(value - nearest) < 1e-9:
        return float(nearest)
    return round(value, 6)


def constant_fold(node: Node) -> Node:
    if isinstance(node, (Num, Var)):
        return node
    if isinstance(node, UnOp):
        child = constant_fold(node.child)
        if isinstance(child, Num):
            with np.errstate(all="ignore"):
                value = float(np.sqrt(child.value))
            return Num(_clean(value))
        return UnOp(node.op, child)
    left = constant_fold(node.left)
    right = constant_fold(node.right)
    if isinstance(left, Num) and isinstance(right, Num):
        with np.errstate(all="ignore"):
            value = float(_eval(BinOp(node.op, left, right), {"x": np.zeros(1)}))
        return Num(_clean(value))
    return BinOp(node.op, left, right)


def _collect_mul(node: Node) -> tuple[float, list[Node]]:
    """Flatten a product into one constant factor plus the variable factors."""
    if isinstance(node, BinOp) and node.op == "*":
        c1, a = _collect_mul(node.left)
        c2, b = _collect_mul(node.right)
        return c1 * c2, a + b
    if isinstance(node, Num):
        return node.value, []
    return 1.0, [node]


def _rebuild_mul(factors: Sequence[Node]) -> Node:
    acc = factors[0]
    for factor in factors[1:]:
        acc = BinOp("*", acc, factor)
    return acc


def _is_num(node: Node, value: float | None = None) -> bool:
    if not isinstance(node, Num):
        return False
    return value is None or abs(node.value - value) < 1e-12


def simplify(node: Node) -> Node:
    """Fold constants and drop identities such as ``x * 1`` or ``x + 0``."""
    node = constant_fold(node)
    if isinstance(node, (Num, Var)):
        return node
    if isinstance(node, UnOp):
        # Re-fold: simplifying the child can turn it into a constant
        # (e.g. ``x - x`` -> 0), which may make the whole node constant.
        return constant_fold(UnOp(node.op, simplify(node.child)))

    left = simplify(node.left)
    right = simplify(node.right)
    op = node.op

    if op == "+":
        if _is_num(left, 0.0):
            return right
        if _is_num(right, 0.0):
            return left
    elif op == "-":
        if _is_num(right, 0.0):
            return left
        if left == right:
            return Num(0.0)
    elif op == "*":
        if _is_num(left, 0.0) or _is_num(right, 0.0):
            return Num(0.0)
        if _is_num(left, 1.0):
            return right
        if _is_num(right, 1.0):
            return left
        if _is_num(left, -1.0):
            return _negate(right)
        if _is_num(right, -1.0):
            return _negate(left)
        const, factors = _collect_mul(BinOp(op, left, right))
        const = _clean(const)
        if not factors:
            return Num(const)
        if const == 0.0:
            return Num(0.0)
        if const != 1.0:
            factors = [Num(const), *factors]
        return _rebuild_mul(factors)
    elif op == "/":
        if _is_num(right, 1.0):
            return left
        if _is_num(left, 0.0) and not _may_be_zero(right):
            return Num(0.0)
        if isinstance(right, Num) and right.value != 0.0 and isinstance(left, BinOp) and left.op == "*":
            const, factors = _collect_mul(left)
            if factors and const != 1.0:
                scaled = _clean(const / right.value)
                if scaled != 1.0:
                    factors = [Num(scaled), *factors]
                return _rebuild_mul(factors)
    return constant_fold(BinOp(op, left, right))


def _negate(node: Node) -> Node:
    if isinstance(node, Num):
        return Num(-node.value)
    return BinOp("-", Num(0.0), node)


def _may_be_zero(node: Node) -> bool:
    if isinstance(node, Num):
        return abs(node.value) < 1e-12
    return True


# --------------------------------------------------------------------------- #
# Parser (for hand-written formulas and for round-trip tests)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+)"
    r"|(?P<name>[A-Za-z_]\w*)"
    r"|(?P<op>[-+*/%()]))"
)


def tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise ValueError(f"cannot tokenize {text[pos:]!r} in {text!r}")
        pos = m.end()
        if m.group("num") is not None:
            tokens.append(("num", m.group("num")))
        elif m.group("name") is not None:
            tokens.append(("name", m.group("name")))
        else:
            tokens.append(("op", m.group("op")))
    tokens.append(("eof", ""))
    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[tuple[str, str]], text: str) -> None:
        self.tokens = tokens
        self.i = 0
        self.text = text

    def peek(self) -> tuple[str, str]:
        return self.tokens[self.i]

    def take(self) -> tuple[str, str]:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, value: str) -> None:
        kind, val = self.take()
        if val != value:
            raise ValueError(f"expected {value!r}, got {val!r} in {self.text!r}")

    def parse(self) -> Node:
        node = self.expr()
        kind, val = self.peek()
        if kind != "eof":
            raise ValueError(f"unexpected trailing {val!r} in {self.text!r}")
        return node

    def expr(self) -> Node:
        node = self.term()
        while True:
            kind, val = self.peek()
            if kind == "op" and val in ("+", "-"):
                self.take()
                node = BinOp(val, node, self.term())
            else:
                return node

    def term(self) -> Node:
        node = self.factor()
        while True:
            kind, val = self.peek()
            if kind == "op" and val in ("*", "/", "%"):
                self.take()
                node = BinOp(val, node, self.factor())
            else:
                return node

    def factor(self) -> Node:
        kind, val = self.peek()
        if kind == "op" and val == "-":
            self.take()
            return _negate(self.factor())
        if kind == "op" and val == "+":
            self.take()
            return self.factor()
        return self.primary()

    def primary(self) -> Node:
        kind, val = self.take()
        if kind == "num":
            return Num(float(val))
        if kind == "name":
            if val in UNARY_FUNCS:
                self.expect("(")
                child = self.expr()
                self.expect(")")
                return UnOp(val, child)
            if val == "nan":
                return Num(float("nan"))
            return Var(val)
        if kind == "op" and val == "(":
            node = self.expr()
            self.expect(")")
            return node
        raise ValueError(f"unexpected {val!r} in {self.text!r}")


def parse(text: str) -> Node:
    """Parse infix text such as ``"(x*x + 1) / (x + 3)"``."""
    return _Parser(tokenize(text), text).parse()


# --------------------------------------------------------------------------- #
# Domain probing / validation
# --------------------------------------------------------------------------- #
@dataclass
class ProbeStats:
    valid_frac: float
    p99_abs: float
    std: float
    median_abs: float
    min_valid: float
    max_valid: float
    lin_r2: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "valid_frac": self.valid_frac,
            "p99_abs": self.p99_abs,
            "std": self.std,
            "median_abs": self.median_abs,
            "min_valid": self.min_valid,
            "max_valid": self.max_valid,
            "lin_r2": self.lin_r2,
        }


def probe(
    node: Node,
    domain: tuple[float, float] = (-3.0, 3.0),
    n: int = 4000,
    rng: np.random.Generator | None = None,
) -> ProbeStats:
    rng = rng or np.random.default_rng(0)
    names = sorted(vars_in(node)) or ["x"]
    xs = rng.uniform(domain[0], domain[1], size=(n, len(names)))
    env = {name: xs[:, i] for i, name in enumerate(names)}
    y = evaluate(node, env)
    valid = np.isfinite(y)
    vf = float(valid.mean())
    if vf == 0.0:
        return ProbeStats(0.0, math.inf, 0.0, math.inf, math.nan, math.nan)
    vals = y[valid]
    return ProbeStats(
        valid_frac=vf,
        p99_abs=float(np.percentile(np.abs(vals), 99)),
        std=float(np.std(vals)),
        median_abs=float(np.median(np.abs(vals))),
        min_valid=float(vals.min()),
        max_valid=float(vals.max()),
        lin_r2=_linear_fit_r2(xs[valid], vals),
    )


def _linear_fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R2 of a plain least-squares linear fit; ~1 means the formula is secretly
    affine (e.g. ``x*x*(5/x)``) and therefore a boring target."""
    if len(y) < 3:
        return float("nan")
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-15:
        return 1.0
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# Random generation
# --------------------------------------------------------------------------- #
TIER_OPS: dict[str, tuple[str, ...]] = {
    "smooth": ("+", "-", "*", "/"),
    "dirty": ("+", "-", "*", "/", "%"),
}

_OP_WEIGHTS = {"+": 1.0, "-": 1.0, "*": 1.0, "/": 0.75, "%": 0.55}
_NICE_CONSTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


@dataclass
class FormulaSpec:
    tier: str = "smooth"
    var_names: tuple[str, ...] = ("x",)
    domain: tuple[float, float] = (-3.0, 3.0)
    max_depth: int = 4
    min_ops: int = 2
    max_ops: int = 9
    require_ops: tuple[str, ...] = ()
    min_valid_frac: float = 0.5
    max_p99_abs: float = 1e3
    min_std: float = 1e-2
    min_rel_std: float = 0.05
    max_linear_r2: float = 0.9999
    allow_affine: bool = False
    p_unary: float = 0.22
    p_stop: float = 0.25
    const_range: tuple[float, float] = (-5.0, 5.0)

    @property
    def ops(self) -> tuple[str, ...]:
        return TIER_OPS[self.tier]


def _rand_const(rng: np.random.Generator, spec: FormulaSpec, positive: bool = False) -> float:
    if rng.random() < 0.7:
        value = float(rng.choice(_NICE_CONSTS))
    else:
        value = float(rng.uniform(*spec.const_range))
        value = round(value, 2)
    if positive:
        return abs(value) if abs(value) > 0.25 else 1.0
    if rng.random() < 0.45:
        value = -value
    return value


def _gen(rng: np.random.Generator, spec: FormulaSpec, remaining: int) -> Node:
    if remaining <= 0:
        return _leaf(rng, spec)

    roll = rng.random()
    if roll < spec.p_stop:
        return _leaf(rng, spec)

    ops = [op for op in spec.ops]
    weights = np.array([_OP_WEIGHTS[op] for op in ops], dtype=np.float64)
    op = str(rng.choice(ops, p=weights / weights.sum()))

    if op == "%":
        # The modulus is always a positive constant: ``x % m`` is then a clean
        # periodic sawtooth with period ``m``, which is the thing we want to
        # study.  A formula-valued modulus would make the "period" meaningless.
        return BinOp("%", _gen(rng, spec, remaining - 1), Num(_rand_const(rng, spec, positive=True)))

    if rng.random() < spec.p_unary:
        func = str(rng.choice(UNARY_FUNCS))
        return UnOp(func, _gen(rng, spec, remaining - 1))

    left = _gen(rng, spec, remaining - 1)
    right = _gen(rng, spec, remaining - 1)
    if op == "/":
        # Keep the denominator away from an exact constant zero.
        if isinstance(right, Num) and abs(right.value) < 0.3:
            right = Num(_rand_const(rng, spec, positive=True) + 0.4)
        if isinstance(left, Num) and abs(left.value) < 1e-9:
            left = _leaf(rng, spec)
    if op in "+-/" and left == right:
        # ``x + x`` / ``x - x`` / ``x / x`` are noise, not structure.
        right = _leaf(rng, spec) if rng.random() < 0.6 else Num(_rand_const(rng, spec))
    return BinOp(op, left, right)


def _leaf(rng: np.random.Generator, spec: FormulaSpec) -> Node:
    if rng.random() < 0.62:
        return Var(str(rng.choice(spec.var_names)))
    return Num(_rand_const(rng, spec))


def _has_trivial_self_op(node: Node) -> bool:
    """Reject ``a + a`` / ``a - a`` / ``a / a`` anywhere in the tree."""
    if isinstance(node, (Num, Var)):
        return False
    if isinstance(node, UnOp):
        return _has_trivial_self_op(node.child)
    if node.op in ("+", "-", "/", "%") and node.left == node.right:
        return True
    return _has_trivial_self_op(node.left) or _has_trivial_self_op(node.right)


def _has_structural_nonlinearity(node: Node) -> bool:
    """True if the expression is more than a polynomial (sqrt / mod / variable
    denominator / product of two non-constant factors)."""
    if isinstance(node, (Num, Var)):
        return False
    if isinstance(node, UnOp):
        return True
    if node.op in ("+", "-"):
        return _has_structural_nonlinearity(node.left) or _has_structural_nonlinearity(node.right)
    if node.op == "/":
        return not isinstance(node.right, Num) or _has_structural_nonlinearity(node.left)
    if node.op == "*":
        both = not isinstance(node.left, Num) and not isinstance(node.right, Num)
        return both or _has_structural_nonlinearity(node.left) or _has_structural_nonlinearity(node.right)
    # "%" is always structural
    return True


def _looks_interesting(node: Node, spec: FormulaSpec) -> bool:
    n_ops = count_ops(node)
    if n_ops < spec.min_ops or n_ops > spec.max_ops:
        return False
    if not vars_in(node):
        return False
    if not spec.allow_affine:
        if is_affine(node):
            return False
        if not _has_structural_nonlinearity(node):
            return False
    if _has_trivial_self_op(node):
        return False
    if any(op not in ops_in(node) for op in spec.require_ops):
        return False
    return True


@dataclass
class Formula:
    node: Node
    text: str
    tier: str
    seed: int
    tag: str
    period: float | None = None
    stats: ProbeStats | None = None

    def as_dict(self) -> dict:
        out = {
            "text": self.text,
            "tier": self.tier,
            "seed": self.seed,
            "tag": self.tag,
            "n_ops": count_ops(self.node),
            "depth": depth(self.node),
            "ops": "".join(sorted(ops_in(self.node))),
            "period": self.period,
        }
        if self.stats is not None:
            out.update({f"probe_{k}": v for k, v in self.stats.as_dict().items()})
        return out


def sample_formula(spec: FormulaSpec, seed: int, max_attempts: int = 400) -> Formula:
    """Sample a random expression that survives the validity filters."""
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        node = simplify(_gen(rng, spec, spec.max_depth))
        if not _looks_interesting(node, spec):
            continue
        stats = probe(node, spec.domain, rng=rng)
        if stats.valid_frac < spec.min_valid_frac:
            continue
        if stats.p99_abs > spec.max_p99_abs:
            continue
        if stats.std < spec.min_std:
            continue
        if stats.p99_abs > 0 and stats.std / stats.p99_abs < spec.min_rel_std:
            continue
        if np.isfinite(stats.lin_r2) and stats.lin_r2 > spec.max_linear_r2 and not spec.allow_affine:
            continue
        return Formula(
            node=node,
            text=to_string(node),
            tier=spec.tier,
            seed=seed,
            tag=f"{spec.tier}-{seed}",
            stats=stats,
        )
    raise RuntimeError(f"failed to sample a valid {spec.tier} formula for seed {seed}")


def formula_from_text(
    text: str,
    tier: str,
    tag: str,
    period: float | None = None,
    domain: tuple[float, float] = (-3.0, 3.0),
    seed: int = -1,
) -> Formula:
    node = simplify(parse(text))
    stats = probe(node, domain)
    return Formula(
        node=node,
        text=to_string(node),
        tier=tier,
        seed=seed,
        tag=tag,
        period=period,
        stats=stats,
    )


# --------------------------------------------------------------------------- #
# CLI helper
# --------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="print random formulas")
    parser.add_argument("--tier", default="dirty", choices=sorted(TIER_OPS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    spec = FormulaSpec(tier=args.tier, max_depth=args.depth)
    for seed in range(args.start, args.start + args.seeds):
        try:
            f = sample_formula(spec, seed)
        except RuntimeError as exc:
            print(f"seed {seed:>3}: <{exc}>")
            continue
        s = f.stats
        assert s is not None
        print(
            f"seed {seed:>3} | ops={count_ops(f.node)} d={depth(f.node)}"
            f" | valid={s.valid_frac:.2f} std={s.std:.2f} p99={s.p99_abs:.2f}"
            f" | {f.text}"
        )


if __name__ == "__main__":
    _main()
