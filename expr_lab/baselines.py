"""Classical baselines.  A neural net that cannot beat these is not learning
anything interesting about the formula.
"""

from __future__ import annotations

import numpy as np

from .metrics import regression_metrics


def _clean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    return x[mask], y[mask]


def _subsample(x: np.ndarray, y: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= n:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx], y[idx]


def fit_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int = 0,
    with_forest: bool = True,
) -> dict[str, dict[str, dict[str, float]]]:
    """Fit every baseline on the training split and evaluate on each split.

    ``splits`` maps a split name to ``(x, y_true)``.  Returns
    ``{baseline: {split: metrics}}``.
    """
    x_tr, y_tr = _clean(np.asarray(x_train, np.float64), np.asarray(y_train, np.float64))
    out: dict[str, dict[str, dict[str, float]]] = {}

    def record(name: str, predict) -> None:
        out[name] = {}
        for split_name, (x_s, y_s) in splits.items():
            x_s = np.asarray(x_s, np.float64)
            y_s = np.asarray(y_s, np.float64)
            keep = np.isfinite(y_s) & np.all(np.isfinite(x_s), axis=1)
            pred = predict(x_s[keep])
            out[name][split_name] = regression_metrics(y_s[keep], pred)

    mean_value = float(np.mean(y_tr))
    record("mean", lambda x: np.full(len(x), mean_value))

    if len(x_tr) == 0:
        return out

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    xk, yk = _subsample(x_tr, y_tr, 40_000, seed)
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1).fit(xk, yk)
    record("knn", lambda x: knn.predict(x))

    xp, yp = _subsample(x_tr, y_tr, 40_000, seed + 1)
    poly = make_pipeline(PolynomialFeatures(degree=9, include_bias=True), StandardScaler(), Ridge(alpha=1e-4))
    poly.fit(xp, yp)
    record("poly9", lambda x: poly.predict(x))

    if with_forest:
        xf, yf = _subsample(x_tr, y_tr, 60_000, seed + 2)
        forest = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=seed,
        ).fit(xf, yf)
        record("forest", lambda x: forest.predict(x))

    return out
