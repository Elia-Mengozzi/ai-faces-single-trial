"""Fixed linear estimators. Every fitted transformation sees training rows only."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def nuisance_residuals(train, test, design_train, design_test, ridge=1e-8):
    """Fit the nuisance regression on training data and apply it to both sets."""
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    train_shape, test_shape = train.shape, test.shape
    train = train.reshape(len(train), -1)
    test = test.reshape(len(test), -1)
    mean = design_train.mean(axis=0)
    scale = design_train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    design_train = np.column_stack([np.ones(len(train)), (design_train - mean) / scale])
    design_test = np.column_stack([np.ones(len(test)), (design_test - mean) / scale])
    penalty = np.eye(design_train.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design_train.T @ design_train + penalty, design_train.T @ train)
    return ((train - design_train @ coefficients).reshape(train_shape),
            (test - design_test @ coefficients).reshape(test_shape))


def fit_diagonal_lda(x, y, shrinkage=0.1):
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1} or min(np.bincount(y.astype(int))) < 2:
        raise ValueError("Both training classes require at least two observations")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite training features")
    class0, class1 = x[y == 0], x[y == 1]
    mean0, mean1 = class0.mean(axis=0), class1.mean(axis=0)
    sum_squares = ((class0 - mean0) ** 2).sum(axis=0) + ((class1 - mean1) ** 2).sum(axis=0)
    variance = sum_squares / (len(x) - 2)
    valid = variance[(variance > 1e-12) & np.isfinite(variance)]
    target = np.median(valid) if len(valid) else 1.0
    variance = np.maximum((1 - shrinkage) * variance + shrinkage * target, target * 1e-6)
    weight = (mean1 - mean0) / variance
    intercept = -0.5 * np.dot(mean1 + mean0, weight)
    score_scale = np.sqrt(max(np.sum(weight**2 * variance), 1e-12))
    return weight / score_scale, intercept / score_scale


def linear_scores(train, y, test, learner, config):
    train = np.asarray(train, dtype=np.float64).reshape(len(train), -1)
    test = np.asarray(test, dtype=np.float64).reshape(len(test), -1)
    if learner == "dlda":
        weight, intercept = fit_diagonal_lda(train, y, config["variance_shrinkage"])
        return test @ weight + intercept
    if learner == "logistic":
        scaler = StandardScaler().fit(train)
        model = LogisticRegression(C=config["logistic_C"], class_weight="balanced",
                                   solver="lbfgs", max_iter=2000, tol=1e-6,
                                   random_state=config["seed"])
        model.fit(scaler.transform(train), y)
        if model.n_iter_.max() >= model.max_iter:
            raise RuntimeError("Logistic regression did not converge")
        return model.decision_function(scaler.transform(test))
    raise ValueError(learner)
