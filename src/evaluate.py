"""
evaluate.py — metrics + comparison helpers, implemented in NumPy.

scikit-learn is used only in the tests, to sanity-check these numbers.
"""

from __future__ import annotations

import time

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))


def regression_report(y_log_true, y_log_pred) -> dict:
    """Metrics on log(price) and, after exp(), on price in AZN."""
    p_true, p_pred = np.exp(y_log_true), np.exp(y_log_pred)
    return {
        "RMSE_log": rmse(y_log_true, y_log_pred),
        "MAE_log": mae(y_log_true, y_log_pred),
        "R2_log": r2(y_log_true, y_log_pred),
        "RMSE_AZN": rmse(p_true, p_pred),
        "MAE_AZN": mae(p_true, p_pred),
        "R2_AZN": r2(p_true, p_pred),
        "MAPE": mape(p_true, p_pred),
    }


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]] (positive class = 1)."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _ranks(x):
    """Average ranks (1-based), ties get the mean of their ranks."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    # boundaries of tie groups
    starts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + e + 1) / 2.0
    return ranks


def roc_auc(y_true, scores) -> float:
    """
    AUC = P(score of a random positive > score of a random negative), via the
    Mann-Whitney U statistic on ranks (ties count 1/2).
    """
    y_true = np.asarray(y_true).astype(int)
    r = _ranks(scores)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    u = r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(y_true, scores):
    """(fpr, tpr) arrays, thresholds = every distinct score (descending)."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    s, yt = np.asarray(scores, float)[order], y_true[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tps = np.cumsum(yt)[last]
    fps = (last + 1) - tps
    return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]]


def average_precision(y_true, scores) -> float:
    """PR-AUC as average precision: sum_k (R_k - R_{k-1}) P_k."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    s, yt = np.asarray(scores, float)[order], y_true[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tps = np.cumsum(yt)[last]
    precision = tps / (last + 1)
    recall = tps / yt.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def classification_report(y_true, y_pred, scores) -> dict:
    p, r, f = precision_recall_f1(y_true, y_pred)
    return {"Accuracy": accuracy(y_true, y_pred), "Precision": p, "Recall": r, "F1": f,
            "ROC_AUC": roc_auc(y_true, scores), "PR_AUC": average_precision(y_true, scores)}


# --- Resampling ---------------------------------------------------------------
def kfold_indices(n, k=5, seed=42, strata=None):
    """List of (train_idx, test_idx) for k folds; stratified if `strata` given."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(n, dtype=int)
    groups = [np.arange(n)] if strata is None else [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for g in groups:
        g = rng.permutation(g)
        fold_of[g] = np.arange(len(g)) % k
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def cross_validate(fit_predict, X, y, k=5, seed=42, strata=None):
    """
    fit_predict(X_tr, y_tr, X_te, y_te) -> dict of metrics for that fold.
    Any preprocessing (e.g. scaling) must happen INSIDE fit_predict using
    the fold's training rows only. Returns {metric: (mean, std)} plus
    per-fold wall-clock time.
    """
    rows = []
    for tr, te in kfold_indices(len(y), k, seed, strata):
        t0 = time.perf_counter()
        m = fit_predict(X[tr], y[tr], X[te], y[te])
        m["time_s"] = time.perf_counter() - t0
        rows.append(m)
    return {key: (float(np.mean([r[key] for r in rows])), float(np.std([r[key] for r in rows])))
            for key in rows[0]}


def bootstrap_std(metric_fn, y_true, y_pred, n_boot=200, seed=42) -> float:
    """Std of a test-set metric over bootstrap resamples of the test rows."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(metric_fn(np.asarray(y_true)[i], np.asarray(y_pred)[i]))
    return float(np.std(vals))


def timed(fn, *args, **kwargs):
    """Run fn and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict, keys=None, title: str = "") -> str:
    """
    Pretty-print a table: rows = models, columns = metrics. Values may be
    floats or (mean, std) tuples. Returns the table as a string.
    """
    keys = keys or list(next(iter(name_to_metrics.values())).keys())
    w0 = max(len(n) for n in name_to_metrics) + 2

    def fmt(v):
        if isinstance(v, tuple):
            return f"{v[0]:.4f}±{v[1]:.4f}"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1e4 else f"{v:,.0f}"
        return "n/a" if v is None else str(v)

    cells = {n: [fmt(m.get(k)) for k in keys] for n, m in name_to_metrics.items()}
    widths = [max(len(k), *(len(c[i]) for c in cells.values())) + 2 for i, k in enumerate(keys)]
    lines = []
    if title:
        lines.append(title)
    lines.append("Model".ljust(w0) + "".join(k.rjust(w) for k, w in zip(keys, widths)))
    lines.append("-" * len(lines[-1]))
    for n, c in cells.items():
        lines.append(n.ljust(w0) + "".join(v.rjust(w) for v, w in zip(c, widths)))
    out = "\n".join(lines)
    print(out)
    return out
