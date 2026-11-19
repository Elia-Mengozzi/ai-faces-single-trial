"""Fit and evaluate the study linear decoders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .io import save_json
from .models import linear_scores, nuisance_residuals
from .raw import GENERATED_CODES
from .splits import crossed_holdout

MODEL_SPECS = {
    "eeg16_dlda_adjusted": ("eeg16", "dlda"),
    "eeg16_logistic_adjusted": ("eeg16", "logistic"),
    "eeg32_dlda_adjusted": ("eeg32", "dlda"),
    "eeg32_logistic_adjusted": ("eeg32", "logistic"),
    "posterior10_dlda_adjusted": ("posterior10", "dlda"),
    "eeg16_dlda_unadjusted": ("unadjusted", "dlda"),
    "prestim_dlda": ("prestim", "dlda"),
    "eog_dlda": ("eog", "dlda"),
    "metadata_dlda": ("presentation_metadata", "dlda"),
}


def load_data(output: Path):
    paths = sorted((output / "features").glob("S*.npz"))
    if not paths:
        raise ValueError("No prepared subject feature caches found")
    feature_chunks = {name: [] for name in ("eeg", "eog", "prestim", "metadata")}
    trial_frames = []
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            keep = cache["keep"].astype(bool)
            for name, chunks in feature_chunks.items():
                chunks.append(cache[name][keep])
            frame = pd.DataFrame({
                "subject": np.repeat(path.stem, int(keep.sum())),
                **{key: cache[key][keep] for key in ("sequence", "position", "trial", "event_code")},
            })
            frame["y"] = frame.event_code.isin(GENERATED_CODES).astype(int)
            trial_frames.append(frame)
            names = cache["channel_names"].astype(str).tolist()
    data = {name: np.concatenate(chunks).astype(np.float64) for name, chunks in feature_chunks.items()}
    return data, pd.concat(trial_frames, ignore_index=True), names


def make_folds(meta, config):
    return crossed_holdout(meta.subject.tolist(), meta.sequence.tolist(),
                           config["participant_groups"], config["sequences"])


def prepare_fold(data, train, test, names, config):
    metadata_train, metadata_test = data["metadata"][train], data["metadata"][test]
    eeg_train, eeg_test = data["eeg"][train], data["eeg"][test]
    eog_train, eog_test = data["eog"][train], data["eog"][test]
    joint_train = np.column_stack([metadata_train, eog_train.reshape(len(train), -1)])
    joint_test = np.column_stack([metadata_test, eog_test.reshape(len(test), -1)])
    adjusted_train, adjusted_test = nuisance_residuals(eeg_train, eeg_test, joint_train, joint_test)
    primary = [names.index(channel) for channel in config["primary_montage"]]
    comparison = [names.index(channel) for channel in config["comparison_montage"]]
    features = {name: (adjusted_train[:, channels], adjusted_test[:, channels])
                for name, channels in (("eeg16", primary), ("eeg32", comparison), ("posterior10", primary[:10]))}
    features["unadjusted"] = (eeg_train[:, primary], eeg_test[:, primary])
    features["prestim"] = nuisance_residuals(
        data["prestim"][train][:, primary], data["prestim"][test][:, primary], metadata_train, metadata_test,
    )
    features["eog"] = nuisance_residuals(eog_train, eog_test, metadata_train, metadata_test)
    features["presentation_metadata"] = (metadata_train, metadata_test)
    return features


def auc_rank(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("AUC requires equal-length vectors and finite scores")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("AUC requires both classes")
    ranks = rankdata(scores, method="average")
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cell_metrics(predictions, model_names):
    rows = []
    for (subject, sequence), frame in predictions.groupby(["subject", "sequence"], sort=True):
        if frame.y.nunique() != 2 or frame.y.value_counts().min() < 2:
            raise ValueError(f"Too few retained trials in {subject}, sequence {sequence}")
        for model in model_names:
            scores = frame[model].to_numpy()
            auc = roc_auc_score(frame.y, scores)
            if abs(auc - auc_rank(frame.y, scores)) > 1e-12:
                raise AssertionError("Independent AUC implementations disagree")
            rows.append({"subject": subject, "sequence": int(sequence), "model": model,
                         "n": len(frame), "n_ai": int(frame.y.sum()), "auc": float(auc),
                         "balanced_accuracy": float(balanced_accuracy_score(frame.y, scores >= 0))})
    return pd.DataFrame(rows)


def analyze(output, config):
    data, meta, names = load_data(output)
    folds = make_folds(meta, config)
    predictions = meta.copy()
    for model in MODEL_SPECS:
        predictions[model] = np.nan
    predictions["fold"] = ""
    diagnostics = []
    y = meta.y.to_numpy()
    for number, fold in enumerate(folds, 1):
        print(f"Models: outer fold {number}/{len(folds)} {fold.name}", flush=True)
        train, test = np.asarray(fold.train), np.asarray(fold.test)
        features = prepare_fold(data, train, test, names, config)
        for name, (feature, learner) in MODEL_SPECS.items():
            xt, xv = features[feature]
            predictions.loc[test, name] = linear_scores(xt, y[train], xv, learner, config)
        predictions.loc[test, "fold"] = fold.name
        diagnostics.append({"fold": fold.name, "n_train": len(train), "n_test": len(test),
                            "held_subjects": list(fold.held_subjects), "held_sequence": fold.held_sequence,
                            "subject_overlap": len(set(meta.subject.iloc[train]) & set(meta.subject.iloc[test])),
                            "sequence_overlap": len(set(meta.sequence.iloc[train]) & set(meta.sequence.iloc[test]))})
    if predictions[list(MODEL_SPECS)].isna().any().any():
        raise ValueError("Missing predictions")
    predictions.to_csv(output / "oof_predictions.csv.gz", index=False)
    cells = cell_metrics(predictions, MODEL_SPECS)
    cells.to_csv(output / "cell_metrics.csv", index=False)
    save_json(output / "fold_audit.json", diagnostics)
    summary = {model: {"auc": float(frame.auc.mean()), "balanced_accuracy": float(frame.balanced_accuracy.mean())}
               for model, frame in cells.groupby("model")}
    save_json(output / "descriptive_results.json", {"n_subjects": int(meta.subject.nunique()), "n_trials": len(meta),
              "n_cells": len(cells) // len(MODEL_SPECS), "models": summary,
              "status": "point estimates; see inference.json for conditional tests and uncertainty"})
