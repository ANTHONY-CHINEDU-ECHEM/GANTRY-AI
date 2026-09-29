"""Predictive risk models.

Three calibrated logistic models and one softmax model, trained in pure numpy
with Adam in a couple of seconds:

* probability that final cost exceeds baseline by more than 20 percent
* probability that duration exceeds plan by more than 25 percent
* probability that the project is cancelled or descoped
* likely critical issues for the profile (softmax over the 15 issue classes)

Every prediction comes with its main drivers, derived from each feature's
contribution to the linear score, so a project manager can see why the model
thinks a project is exposed rather than receiving an unexplained number.
"""

import json

import numpy as np

from .utils import safe_float, sigmoid, softmax_rows, sub

CATEGORICAL_FEATURES = ["industry", "delivery_methodology", "contract_type", "organisation_size",
                        "requirements_volatility", "regulatory_exposure", "region"]
NUMERIC_FEATURES = ["technical_complexity", "sponsor_engagement", "pm_experience_years", "team_size",
                    "vendor_count", "dependency_count", "stakeholder_count", "team_turnover_pct",
                    "baseline_budget_gbp", "planned_duration_days"]
LOG_FEATURES = {"pm_experience_years", "team_size", "baseline_budget_gbp", "planned_duration_days",
                "vendor_count", "dependency_count", "stakeholder_count"}
TARGETS = {
    "cost_overrun_over_20pct": "Cost overrun above 20 percent",
    "schedule_overrun_over_25pct": "Schedule overrun above 25 percent",
    "cancelled_or_descoped": "Cancellation or descoping",
}

FRIENDLY = {
    "technical_complexity": "technical complexity", "sponsor_engagement": "sponsor engagement",
    "pm_experience_years": "project manager experience", "team_size": "team size",
    "vendor_count": "number of vendors", "dependency_count": "number of dependencies",
    "stakeholder_count": "number of stakeholders", "team_turnover_pct": "team turnover",
    "baseline_budget_gbp": "budget size", "planned_duration_days": "planned duration",
}


def _targets(frame):
    return {
        "cost_overrun_over_20pct": (frame["cost_ratio"] > 1.2).to_numpy(float),
        "schedule_overrun_over_25pct": (frame["schedule_ratio"] > 1.25).to_numpy(float),
        "cancelled_or_descoped": frame["outcome_status"].isin(["Cancelled", "Descoped"]).to_numpy(float),
    }


class FeatureEncoder:
    def __init__(self, categories=None, means=None, stds=None):
        self.categories = categories or {}
        self.means = means or {}
        self.stds = stds or {}

    def fit(self, frame):
        self.categories = {c: sorted(frame[c].astype(str).unique()) for c in CATEGORICAL_FEATURES}
        for c in NUMERIC_FEATURES:
            v = self._raw(frame[c].to_numpy(float), c)
            self.means[c] = float(v.mean())
            self.stds[c] = float(v.std()) or 1.0
        return self

    @staticmethod
    def _raw(values, column):
        return np.log1p(np.maximum(values, 0.0)) if column in LOG_FEATURES else values

    @property
    def names(self):
        out = []
        for c in CATEGORICAL_FEATURES:
            out.extend("{}={}".format(c, v) for v in self.categories[c])
        out.extend(NUMERIC_FEATURES)
        return out

    def transform_records(self, records):
        rows = []
        for r in records:
            vec = []
            for c in CATEGORICAL_FEATURES:
                value = str(r.get(c, ""))
                vec.extend(1.0 if value == v else 0.0 for v in self.categories[c])
            for c in NUMERIC_FEATURES:
                raw = float(r.get(c, self._inverse_mean(c)))
                val = self._raw(np.array([raw]), c)[0]
                vec.append(sub(val, self.means[c]) / self.stds[c])
            rows.append(vec)
        return np.asarray(rows, dtype=np.float64)

    def _inverse_mean(self, c):
        m = self.means[c]
        return float(np.expm1(m)) if c in LOG_FEATURES else m

    def to_json(self):
        return {"categories": self.categories, "means": self.means, "stds": self.stds}


def _adam(grad_fn, w, steps=400, lr=0.05):
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2 = 0.9, 0.999
    for t in range(1, steps + 1):
        g = grad_fn(w)
        m = b1 * m + sub(1.0, b1) * g
        v = b2 * v + sub(1.0, b2) * g * g
        mh = m / sub(1.0, b1 ** t)
        vh = v / sub(1.0, b2 ** t)
        w = np.subtract(w, lr * mh / (np.sqrt(vh) + 1.0 / 1e8))
    return w


def _fit_logistic(x, y, l2=0.02):
    xb = np.hstack([x, np.ones((x.shape[0], 1))])

    def grad(w):
        p = sigmoid(xb @ w)
        g = xb.T @ np.subtract(p, y) / len(y)
        reg = l2 * w
        reg[x.shape[1]] = 0.0
        return g + reg

    return _adam(grad, np.zeros(xb.shape[1]))


def _fit_softmax(x, y, classes, l2=0.02):
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    onehot = np.eye(classes)[y]

    def grad(w):
        p = softmax_rows(xb @ w)
        g = xb.T @ np.subtract(p, onehot) / len(y)
        reg = l2 * w
        reg[x.shape[1]] = 0.0
        return g + reg

    return _adam(grad, np.zeros((xb.shape[1], classes)), steps=500)


def auc(y, score):
    """Area under the ROC curve via the rank sum formulation."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float(sub(ranks[pos].sum(), n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class RiskModel:
    def __init__(self, encoder=None, weights=None, issue_weights=None, issue_labels=None, metrics=None):
        self.encoder = encoder
        self.weights = weights or {}
        self.issue_weights = issue_weights
        self.issue_labels = issue_labels or []
        self.metrics = metrics or {}

    @classmethod
    def train(cls, frame, seed=11, log=print):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(frame))
        cut = int(0.8 * len(frame))
        train_idx, test_idx = idx[:cut], idx[cut:]
        enc = FeatureEncoder().fit(frame.iloc[train_idx])
        x = enc.transform_records(frame.to_dict("records"))
        targets = _targets(frame)
        weights, metrics = {}, {}
        for name, y in targets.items():
            w = _fit_logistic(x[train_idx], y[train_idx])
            weights[name] = w
            p = sigmoid(np.hstack([x[test_idx], np.ones((len(test_idx), 1))]) @ w)
            metrics[name] = {"test_auc": safe_float(auc(y[test_idx], p), 3),
                             "base_rate": safe_float(y.mean(), 3)}
            log("  {} test AUC {:.3f}".format(name, metrics[name]["test_auc"]))
        labels = sorted(frame["critical_issue"].unique())
        yi = frame["critical_issue"].map({v: i for i, v in enumerate(labels)}).to_numpy()
        wi = _fit_softmax(x[train_idx], yi[train_idx], len(labels))
        probs = softmax_rows(np.hstack([x[test_idx], np.ones((len(test_idx), 1))]) @ wi)
        top3 = np.argsort(np.negative(probs), axis=1)[:, :3]
        hit = np.any(top3 == yi[test_idx][:, None], axis=1).mean()
        acc = (np.argmax(probs, axis=1) == yi[test_idx]).mean()
        chance = 1.0 / len(labels)
        metrics["issue_classifier"] = {"top1_accuracy": safe_float(acc, 3), "top3_accuracy": safe_float(hit, 3),
                                       "chance_top1": safe_float(chance, 3)}
        log("  issue classifier top1 {:.3f} top3 {:.3f} (chance {:.3f})".format(acc, hit, chance))
        return cls(enc, weights, wi, labels, metrics)

    def save(self, directory):
        np.savez(directory / "risk_model.npz", issue=self.issue_weights,
                 **{"t_" + k: v for k, v in self.weights.items()})
        with open(directory / "risk_model.json", "w", encoding="utf8") as fh:
            json.dump({"encoder": self.encoder.to_json(), "issue_labels": self.issue_labels,
                       "metrics": self.metrics}, fh, indent=2)

    @classmethod
    def load(cls, directory):
        with open(directory / "risk_model.json", encoding="utf8") as fh:
            meta = json.load(fh)
        arrays = np.load(directory / "risk_model.npz")
        enc = meta["encoder"]
        encoder = FeatureEncoder(enc["categories"], enc["means"], enc["stds"])
        weights = {k[2:]: arrays[k] for k in arrays.files if k.startswith("t_")}
        return cls(encoder, weights, arrays["issue"], meta["issue_labels"], meta["metrics"])

    def _drivers(self, xrow, w, top=4):
        contrib = xrow * w[:len(xrow)]
        names = self.encoder.names
        order = np.argsort(np.negative(np.abs(contrib)))
        out = []
        for i in order:
            if abs(contrib[i]) < 0.05:
                break
            name = names[i]
            if "=" in name:
                col, val = name.split("=", 1)
                if xrow[i] == 0:
                    continue
                label = "{} is {}".format(col.replace("_", " "), val)
            else:
                direction = "high" if xrow[i] > 0 else "low"
                label = "{} {}".format(direction, FRIENDLY.get(name, name))
            out.append({"factor": label, "effect": "raises risk" if contrib[i] > 0 else "lowers risk",
                        "weight": safe_float(abs(contrib[i]), 3)})
            if len(out) >= top:
                break
        return out

    def predict(self, profile):
        x = self.encoder.transform_records([profile])
        xb = np.hstack([x, np.ones((1, 1))])
        risks = {}
        for name, w in self.weights.items():
            p = float(sigmoid(xb @ w)[0])
            risks[name] = {"label": TARGETS[name], "probability": safe_float(p, 3),
                           "base_rate": self.metrics.get(name, {}).get("base_rate"),
                           "drivers": self._drivers(x[0], w)}
        probs = softmax_rows(xb @ self.issue_weights)[0]
        order = np.argsort(np.negative(probs))[:5]
        issues = [{"issue": self.issue_labels[i], "probability": safe_float(probs[i], 3)} for i in order]
        return {"risks": risks, "likely_issues": issues}
