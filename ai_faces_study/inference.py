"""Shared-position permutations and crossed bootstrap intervals."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .analysis import MODEL_SPECS, load_data, make_folds, prepare_fold
from .io import save_json
from .models import fit_diagonal_lda
from .raw import FACE_CODES, GENERATED_CODES, MALE_FACE_CODES


def shared_item_labels(meta, canonical, n_permutations, seed, scheme="shuffle"):
    rng = np.random.default_rng(seed)
    items, labels, strata = [], [], []
    for sequence in sorted(canonical, key=int):
        for position, code in enumerate(canonical[sequence]):
            if code not in FACE_CODES:
                continue
            items.append((int(sequence), position))
            labels.append(int(code in GENERATED_CODES))
            strata.append((int(sequence), int(code in MALE_FACE_CODES)))
    lookup = {item: i for i, item in enumerate(items)}
    observed = np.asarray(labels, dtype=np.uint8)
    permuted = np.repeat(observed[:, None], n_permutations + 1, axis=1)
    for stratum in sorted(set(strata)):
        indices = np.array([i for i, group in enumerate(strata) if group == stratum])
        stratum_labels = observed[indices]
        for column in range(1, n_permutations + 1):
            if scheme == "shuffle":
                permuted[indices, column] = rng.permutation(stratum_labels)
            elif scheme == "rotation":
                permuted[indices, column] = np.roll(stratum_labels, rng.integers(len(indices)))
            else:
                raise ValueError(scheme)
    trial_rows = [lookup[(int(sequence), int(position))]
                  for sequence, position in zip(meta.sequence, meta.position)]
    result = permuted[np.asarray(trial_rows, dtype=np.intp)]
    if not np.array_equal(result[:, 0], meta.y.to_numpy()):
        raise ValueError("Canonical labels disagree with trial labels")
    return result


def batch_lda_scores(train, test, labels, shrinkage=.1):
    """Vectorized exact refit of class means and pooled diagonal variances."""
    x = np.asarray(train, dtype=np.float64).reshape(len(train), -1)
    z = np.asarray(test, dtype=np.float64).reshape(len(test), -1)
    labels = np.asarray(labels, dtype=np.float64)
    n1 = labels.sum(axis=0)
    n0 = len(x) - n1
    if np.any(n1 < 2) or np.any(n0 < 2):
        raise ValueError("Degenerate permuted training classes")
    sum1 = x.T @ labels
    mu1 = sum1 / n1
    mu0 = (x.sum(axis=0)[:, None] - sum1) / n0
    variance = ((x*x).sum(axis=0)[:, None] - n1*mu1*mu1 - n0*mu0*mu0) / (len(x) - 2)
    positive_variances = (column[column > 1e-12] for column in variance.T)
    target = np.array([np.median(values) if values.size else 1.0 for values in positive_variances])
    variance = np.maximum((1 - shrinkage)*variance + shrinkage*target, target*1e-6)
    weights = (mu1 - mu0) / variance
    intercepts = -.5 * np.sum((mu1 + mu0) * weights, axis=0)
    return z @ weights + intercepts


def auc_columns(labels, scores):
    labels = np.asarray(labels)
    n1 = labels.sum(axis=0)
    n0 = len(labels) - n1
    if np.any(n1 == 0) or np.any(n0 == 0):
        raise ValueError("A permuted test cell has one class; null scheme needs review")
    ranks = rankdata(scores, axis=0, method="average")
    return ((ranks * labels).sum(axis=0) - n1*(n1+1)/2) / (n1*n0)


def permutation_distribution(data, meta, names, config, canonical, models, scheme):
    seed = config["seed"] + (scheme == "rotation")
    labels = shared_item_labels(meta, canonical, config["permutations"], seed, scheme)
    total = {model: np.zeros(labels.shape[1]) for model in models}
    folds = make_folds(meta, config)
    shrinkage = config["variance_shrinkage"]
    n_cells = 0
    for index, fold in enumerate(folds, 1):
        print(f"Permutations ({scheme}): fold {index}/{len(folds)}", flush=True)
        train, test = np.asarray(fold.train), np.asarray(fold.test)
        features = prepare_fold(data, train, test, names, config)
        train_labels, test_labels = labels[train], labels[test]
        test_subjects = meta.subject.iloc[test].to_numpy()
        cells = [np.flatnonzero(test_subjects == subject) for subject in fold.held_subjects]
        for model in models:
            key, learner = MODEL_SPECS[model]
            if learner != "dlda":
                raise ValueError("Vectorized inference supports the fixed DLDA only")
            train_features, test_features = features[key]
            scores = batch_lda_scores(train_features, test_features, train_labels, shrinkage)
            # Check the observed AUC against a separate fit on the original labels.
            weight, intercept = fit_diagonal_lda(train_features, train_labels[:, 0], shrinkage)
            scalar = test_features.reshape(len(test_features), -1) @ weight + intercept
            for rows in cells:
                cell_labels = test_labels[rows]
                auc = auc_columns(cell_labels, scores[rows])
                scalar_auc = auc_columns(cell_labels[:, :1], scalar[rows, None])[0]
                if abs(auc[0] - scalar_auc) > 1e-10:
                    raise AssertionError("Scalar and batched observed AUC differ")
                total[model] += auc
        n_cells += len(cells)
    return {model: values / n_cells for model, values in total.items()}


def conditional_bootstrap(matrix, n_resamples=10000, seed=1):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Complete finite participant-by-sequence matrix required")
    rng = np.random.default_rng(seed)
    rows = rng.integers(matrix.shape[0], size=(n_resamples, matrix.shape[0]))
    columns = rng.integers(matrix.shape[1], size=(n_resamples, matrix.shape[1]))
    subject = matrix.mean(axis=1)[rows].mean(axis=1)
    crossed = matrix[rows[:, :, None], columns[:, None, :]].mean(axis=(1,2))
    return {"estimate": float(matrix.mean()),
            "subject_conditional_ci95": np.quantile(subject, [.025,.975]).tolist(),
            "crossed_fixed_prediction_ci95": np.quantile(crossed, [.025,.975]).tolist()}


def holm(p_values):
    keys = sorted(p_values, key=p_values.get)
    adjusted = {}
    running_max = 0.0
    for rank, key in enumerate(keys):
        running_max = min(1.0, max(running_max, (len(keys) - rank) * p_values[key]))
        adjusted[key] = running_max
    return adjusted


def infer(output, config):
    data, meta, names = load_data(output)
    canonical = json.loads((output / "canonical_sequences.json").read_text())
    cells = pd.read_csv(output / "cell_metrics.csv")
    primary = config["primary_model"]
    permutations = config["permutations"]
    resamples, seed = config["bootstrap_resamples"], config["seed"]
    distributions = permutation_distribution(data, meta, names, config, canonical, config["inference_family"], "shuffle")
    tests = {}
    for model, values in distributions.items():
        observed = float(cells.loc[cells.model == model, "auc"].mean())
        if abs(values[0] - observed) > 1e-10:
            raise AssertionError("Observed permutation statistic differs from saved AUC")
        exceedances = int(np.sum(values[1:] >= observed - 1e-12))
        tests[model] = {"observed_auc": observed, "exceedances": exceedances,
                        "permutations": permutations,
                        "p_one_sided": (exceedances + 1) / (permutations + 1),
                        "null_quantiles": np.quantile(values[1:], [.025,.5,.975]).tolist()}
    adjusted = holm({k: v["p_one_sided"] for k, v in tests.items()})
    for model in tests:
        tests[model]["holm_p_four_tests"] = adjusted[model]
    rotations = permutation_distribution(data, meta, names, config, canonical, [primary], "rotation")
    rot = rotations[primary]
    rotation_p = float((1 + np.sum(rot[1:] >= rot[0] - 1e-12)) / len(rot))
    np.savez_compressed(output / "permutation_distributions.npz", **distributions, primary_rotation=rot)
    matrices = {model: frame.pivot(index="subject", columns="sequence", values="auc").to_numpy()
                for model, frame in cells.groupby("model")}
    uncertainty = {model: conditional_bootstrap(matrix, resamples, seed) for model, matrix in matrices.items()}
    contrasts = {}
    for other in ("eeg16_logistic_adjusted", "eeg32_dlda_adjusted", "eog_dlda", "prestim_dlda"):
        contrasts[f"{primary}_minus_{other}"] = conditional_bootstrap(matrices[primary] - matrices[other], resamples, seed)
    save_json(output / "inference.json", {
        "tests": tests, "rotation_sensitivity": {"p_one_sided": rotation_p, "permutations": permutations},
        "auc_uncertainty": uncertainty, "paired_auc_contrasts_descriptive": contrasts,
        "scope": [
            "Origin labels are exchangeable within sequence and face-sex strata under the shuffle null.",
            "Offsets for the rotation sensitivity are drawn within ordered strata.",
            "Bootstrap intervals condition on the fitted out-of-fold predictions.",
        ],
    })
