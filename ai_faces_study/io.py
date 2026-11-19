"""Configuration and file I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def guard_output(output: Path, sources: list[Path]) -> None:
    out = output.resolve()
    for source in sources:
        src = source.resolve()
        if out == src or src in out.parents or out in src.parents:
            raise ValueError("Output and read-only source roots must not overlap")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Validate the acquisition and analysis settings before processing."""
    sample_rate = config["sampling"]["native_hz"]
    if sample_rate != 500 or config["eog_channels"] != ["VEOG", "HEOG"]:
        raise ValueError("The acquisition requires 500 Hz and VEOG/HEOG channels")
    if config["sequences"] != list(range(1, 9)):
        raise ValueError("The analysis requires canonical sequences 1 to 8")
    for montage, count in (("primary_montage", 16), ("comparison_montage", 32)):
        channels = config[montage]
        if len(channels) != count or len(set(channels)) != count:
            raise ValueError(f"{montage} requires {count} unique channels")
    groups = config["participant_groups"]
    subjects = [s for group in groups for s in group]
    if sorted(map(len, groups)) != [5, 5, 5, 5, 6] or len(set(subjects)) != 26:
        raise ValueError("The participant groups must partition 26 subjects as 5,5,5,5,6")
    family = config["inference_family"]
    if len(family) != 4 or len(set(family)) != 4 or config["primary_model"] not in family:
        raise ValueError("The four permutation tests must include the primary model")
    if not 0 <= config["variance_shrinkage"] <= 1 or config["logistic_C"] <= 0:
        raise ValueError("Invalid classifier regularisation")
    if config["permutations"] < 1 or config["bootstrap_resamples"] < 1:
        raise ValueError("Resampling counts must be positive")
    cutoff = config["filter"]
    if not 0 < cutoff["highpass_hz"] < cutoff["lowpass_hz"] < sample_rate / 2:
        raise ValueError("Filter cutoffs must be ordered and below the Nyquist frequency")
    if min(config["artifacts"]["eeg_peak_to_peak_uv"], config["artifacts"]["eog_peak_to_peak_uv"]) <= 0:
        raise ValueError("Artifact thresholds must be positive")
