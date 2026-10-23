"""Unvalidated capacity-allocation surrogate; no neural-network training.

Fit directional changes in class-pair confusion log ratios, then compute macro-F1.
Frozen-mask diagnostics are covariates, never assumed to be widening effects.
NumPy only. All calibration and validation here are development analyses.
"""
from dataclasses import dataclass
import math

import numpy as np


def checked_cm(value):
    cm = np.asarray(value, dtype=float)
    if (cm.ndim != 2 or cm.shape[0] != cm.shape[1] or len(cm) < 2
            or not np.isfinite(cm).all() or (cm < 0).any()
            or (cm.sum(1) <= 0).any()):
        raise ValueError("Every semantic class requires finite, positive support")
    return cm


def macro_f1(cm):
    cm = checked_cm(cm)
    return float(np.mean(2 * np.diag(cm) / (cm.sum(0) + cm.sum(1))))


def block_widths(channels):
    if len(channels) != 4 or any(type(x) is not int or x <= 0 for x in channels):
        raise ValueError("Expected four positive integer channel widths")
    a, b, c, d = channels
    return np.array([a, b, b, c, d], dtype=float)


def feasible(cost, reference_cost, tolerance=.01):
    return all(abs(cost[k] / reference_cost[k] - 1) <= tolerance + 1e-12
               for k in ("params", "macs"))


@dataclass
class Reference:
    channels: list
    cm: np.ndarray
    loss: np.ndarray
    sensitivity: np.ndarray  # five stages x K classes; mean mask loss delta
    separation: np.ndarray  # five stages x K x K; symmetric feature separation

    def __post_init__(self):
        self.cm = checked_cm(self.cm)
        k = len(self.cm)
        block_widths(self.channels)
        for key, shape in (("loss", (k,)), ("sensitivity", (5, k)),
                           ("separation", (5, k, k))):
            arr = np.asarray(getattr(self, key), dtype=float)
            if arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError(f"Invalid {key}")
            setattr(self, key, arr)
        if (self.loss < 0).any() or (self.separation < 0).any():
            raise ValueError("Loss/separation must be nonnegative")


def pair_features(ref, channels, mode="class_pair"):
    """No candidate outcomes are accepted by this prediction interface.

    Controls retain directional widths; class_agnostic pools diagnostics over
    classes equally, frequency_only uses supports alone. Positive and negative
    directions have separate coefficients. The reference maps to exact zero.
    """
    if mode not in ("architecture_only", "frequency_only", "class_agnostic", "class_pair"):
        raise ValueError(mode)
    delta = np.log(block_widths(channels) / block_widths(ref.channels))
    k = len(ref.cm)
    support = ref.cm.sum(1)
    freq = support / support.sum()
    probs = ref.cm / support[:, None]
    recall = np.diag(probs)
    rows = []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            features = []
            for s in range(5):
                if mode == "architecture_only":
                    context = [1.]
                elif mode == "frequency_only":
                    context = [1., freq[i], freq[j]]
                elif mode == "class_agnostic":
                    offdiag = ref.separation[s][~np.eye(k, dtype=bool)]
                    context = [1., float(ref.loss.mean()),
                               float(ref.sensitivity[s].mean()),
                               float(np.log1p(offdiag).mean())]
                else:
                    context = [1., probs[i, j], recall[i], recall[j],
                               freq[i], freq[j], ref.loss[i], ref.loss[j],
                               ref.sensitivity[s, i], ref.sensitivity[s, j],
                               math.log1p(ref.separation[s, i, j])]
                # Distinct shrink/grow responses; no assumed symmetry.
                for direction in (max(delta[s], 0), max(-delta[s], 0)):
                    features.extend(direction * np.asarray(context))
            rows.append(features)
    return np.asarray(rows, dtype=float)


def log_ratios(cm, alpha=.5):
    cm = checked_cm(cm)
    return np.array([math.log((cm[i, j] + alpha) / (cm[i, i] + alpha))
                     for i in range(len(cm)) for j in range(len(cm)) if i != j])


class ConfusionResponse:
    def __init__(self, mode="class_pair", ridge=1., alpha=.5):
        if ridge <= 0 or alpha <= 0:
            raise ValueError("Positive regularization/smoothing required")
        self.mode, self.ridge, self.alpha = mode, ridge, alpha

    def fit(self, records):
        if not records:
            raise ValueError("No calibration records")
        xs, ys, weights = [], [], []
        for item in records:
            ref = item["reference"]
            target = checked_cm(item["target_cm"])
            if target.shape != ref.cm.shape or not np.array_equal(target.sum(1), ref.cm.sum(1)):
                raise ValueError("Paired predictions must use the same class supports")
            if np.array_equal(block_widths(item["channels"]), block_widths(ref.channels)):
                raise ValueError("Calibration requires an actual capacity transfer")
            x = pair_features(ref, item["channels"], self.mode)
            xs.append(x)
            ys.append(log_ratios(target, self.alpha) - log_ratios(ref.cm, self.alpha))
            # Each retrained run has equal total weight, including different K.
            weights.extend([1 / (len(records) * len(x))] * len(x))
        x, y, w = np.vstack(xs), np.concatenate(ys), np.asarray(weights)
        self.scale = np.sqrt(np.sum(w[:, None] * x*x, axis=0))
        self.scale[self.scale < 1e-12] = 1.
        z = x / self.scale
        self.coef = np.linalg.solve(z.T @ (w[:, None] * z)
                                    + self.ridge * np.eye(z.shape[1]), z.T @ (w*y))
        self.training_ids = [r["id"] for r in records]
        self.fitted = True
        return self

    def predict_cm(self, ref, channels):
        if not getattr(self, "fitted", False):
            raise ValueError("Calibrate on development retraining records first")
        if channels == ref.channels:
            return ref.cm.copy()
        response = (pair_features(ref, channels, self.mode) / self.scale) @ self.coef
        ratios = log_ratios(ref.cm, self.alpha) + response
        k = len(ref.cm)
        logits = np.zeros((k, k))
        t = 0
        for i in range(k):
            for j in range(k):
                if i != j:
                    logits[i, j] = ratios[t]
                    t += 1
        logits -= logits.max(1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(1, keepdims=True)
        return ref.cm.sum(1)[:, None] * probabilities

    def rank(self, ref, candidates, reference_cost, tolerance=.01):
        # Eligibility checks actual P/M, not channel count or proxy estimates.
        eligible = [c for c in candidates if feasible(c["cost"], reference_cost, tolerance)]
        scores = [{"name": c["name"], "channels": c["channels"], "cost": c["cost"],
                   "predicted_macro_f1_pct": 100 * macro_f1(self.predict_cm(ref, c["channels"]))}
                  for c in eligible]
        return sorted(scores, key=lambda x: (-x["predicted_macro_f1_pct"], x["name"]))


def heldout_indices(records, axis):
    """Hold out a whole allocation across tasks/seeds, or an entire dataset.

    No random pair-row/sample split: those would leak architecture outcomes.
    """
    if axis not in ("allocation", "dataset"):
        raise ValueError(axis)
    keys = [tuple(r["channels"]) if axis == "allocation" else r["dataset"] for r in records]
    for key in sorted(set(keys)):
        train = [i for i, x in enumerate(keys) if x != key]
        test = [i for i, x in enumerate(keys) if x == key]
        if not train or not test:
            raise ValueError("Need multiple independent allocation/task groups")
        yield str(key), train, test
