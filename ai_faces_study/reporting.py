"""Generate the aggregate tables used by the manuscript."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import MODEL_SPECS
from .inference import conditional_bootstrap
from .io import save_json
from .raw import FACE_CODES

MODEL_LABELS = {
    "eeg16_dlda_adjusted": "EEG, 16 electrodes, adjusted, dLDA (primary)",
    "eeg16_logistic_adjusted": "EEG, 16 electrodes, adjusted, LR-L2",
    "eeg32_dlda_adjusted": "EEG, 32 electrodes, adjusted, dLDA",
    "eeg32_logistic_adjusted": "EEG, 32 electrodes, adjusted, LR-L2",
    "posterior10_dlda_adjusted": "EEG, posterior 10 electrodes, adjusted, dLDA",
    "eeg16_dlda_unadjusted": "EEG, 16 electrodes, unadjusted, dLDA",
    "prestim_dlda": "Prestimulus EEG control",
    "eog_dlda": "Poststimulus EOG control",
    "metadata_dlda": "Presentation-metadata control",
}


def category_retention(output: Path) -> pd.DataFrame:
    available = dict.fromkeys(FACE_CODES, 0)
    retained = dict.fromkeys(FACE_CODES, 0)
    for path in sorted((output / "features").glob("S*.npz")):
        with np.load(path, allow_pickle=False) as cache:
            codes = cache["event_code"]
            keep = cache["keep"]
            for code in FACE_CODES:
                in_category = codes == code
                available[code] += int(in_category.sum())
                retained[code] += int((in_category & keep).sum())
    return pd.DataFrame(
        {
            "event_code": code,
            "before_rejection": available[code],
            "retained": retained[code],
            "retention_rate": retained[code] / available[code],
        }
        for code in FACE_CODES
    )


def report(output: Path, config: dict) -> None:
    inference = json.loads((output / "inference.json").read_text())
    estimates = json.loads((output / "descriptive_results.json").read_text())["models"]
    qc = json.loads((output / "qc_summary.json").read_text())
    frozen = json.loads((output / "frozen_config.json").read_text())
    folder = output / "review"
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODEL_SPECS:
        intervals = inference["auc_uncertainty"][model]
        subject_ci = intervals["subject_conditional_ci95"]
        crossed_ci = intervals["crossed_fixed_prediction_ci95"]
        test = inference["tests"].get(model, {})
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                **estimates[model],
                "subject_ci_low": subject_ci[0],
                "subject_ci_high": subject_ci[1],
                "crossed_ci_low": crossed_ci[0],
                "crossed_ci_high": crossed_ci[1],
                "conditional_permutation_p": test.get("p_one_sided"),
                "holm_p_four_tests": test.get("holm_p_four_tests"),
            }
        )
    pd.DataFrame(rows).to_csv(folder / "model_summary.csv", index=False)
    category_retention(output).to_csv(folder / "attrition_summary.csv", index=False)

    cells = pd.read_csv(output / "cell_metrics.csv")
    primary = config["primary_model"]
    eeg = cells[cells.model == primary].pivot(index="subject", columns="sequence", values="auc")
    metadata = cells[cells.model == "metadata_dlda"].pivot(index="subject", columns="sequence", values="auc")
    contrast = conditional_bootstrap(
        (eeg - metadata).to_numpy(), config["bootstrap_resamples"], config["seed"],
    )
    save_json(folder / "metadata_contrast.json", contrast)

    summary = {
        "title": config["title"],
        "config_sha256": frozen["config_sha256"],
        "n_subjects": qc["n_included"],
        "n_faces_before_rejection": qc["n_faces"],
        "n_trials": qc["n_retained"],
        "primary_model": primary,
        "models": {model: estimates[model] for model in MODEL_SPECS},
        "inference": inference,
        "gentask_audit": qc["gentask"],
        "native_sample_rates": sorted({row["native_hz"] for row in qc["subjects"]}),
    }
    save_json(folder / "summary.json", summary)
