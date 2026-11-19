"""Read Curry recordings and extract native-rate EEG features."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from .io import config_digest, digest_file, save_json

GENERATED_CODES = (50, 60)
MALE_FACE_CODES = (50, 70)
FACE_CODES = (*GENERATED_CODES, 70, 80)
STIM_CODES = (11, *FACE_CODES)
STIM_CODE_TEXT = tuple(map(str, STIM_CODES))
DAT_EVENT = re.compile(r"^\s*\d+\s+\d+\s+(" + "|".join(STIM_CODE_TEXT) + r")\s+")


def dat_codes(path: Path) -> list[int]:
    return [int(match.group(1)) for line in path.read_text(errors="replace").splitlines()
            if (match := DAT_EVENT.match(line))]


def canonical_sequences(root: Path) -> tuple[dict[int, list[int]], dict]:
    by_subject = {}
    for path in sorted(root.glob("S*/**/*.dat")):
        if "training" in path.name.lower():
            continue
        match = re.search(r"seq-?(\d+)", path.name, re.I) or re.search(r"-(\d+)\.dat$", path.name, re.I)
        if not match or not 1 <= int(match.group(1)) <= 8:
            continue
        subject = path.relative_to(root).parts[0]
        sequence = int(match.group(1))
        codes = dat_codes(path)
        subject_logs = by_subject.setdefault(subject, {})
        if sequence in subject_logs and subject_logs[sequence] != codes:
            raise ValueError(f"Conflicting Gentask logs: {subject}, sequence {sequence}")
        subject_logs[sequence] = codes
    complete = [s for s, values in by_subject.items() if set(values) == set(range(1, 9))]
    if not complete:
        raise ValueError("No complete eight-sequence Gentask log set")
    canonical = by_subject[min(complete)]
    mismatches = []
    for subject, sequences in by_subject.items():
        for q, codes in sequences.items():
            if codes != canonical[q][:len(codes)]:
                mismatches.append([subject, q])
    if mismatches:
        raise ValueError(f"Gentask sequence disagreement: {mismatches}")
    if len({tuple(x) for x in canonical.values()}) != 8:
        raise ValueError("Canonical event patterns are not unique")
    return canonical, {"subjects_with_logs": len(by_subject),
                       "complete_log_sets": len(complete),
                       "code_patterns_agree": True,
                       "face_counts": {str(q): sum(c in FACE_CODES for c in v)
                                       for q, v in canonical.items()},
                       "image_identity_verified": False}


def match_sequence(observed, canonical):
    observed = list(observed)
    matches = [q for q, expected in canonical.items() if observed == expected[:len(observed)]]
    if len(matches) != 1:
        raise ValueError(f"Non-unique event sequence match: {matches}")
    return matches[0]


def source_path(root: Path, subject: str) -> Path:
    paths = sorted((root / subject / "EEG").glob("*.cdt"))
    if subject == "S01":
        preferred = [p for p in paths if " Data Data" not in p.name]
        if preferred:
            paths = preferred
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one selected Curry recording for {subject}")
    return paths[0]


def signed_baseline(epoch: np.ndarray, times: np.ndarray, start=-0.1):
    select = (times >= start - 1e-9) & (times < -1e-9)
    if not np.any(select):
        raise ValueError("Empty baseline")
    return epoch - epoch[..., select].mean(axis=-1, keepdims=True)


def bins(epoch, times, start, stop, width):
    values = []
    for left in np.arange(start, stop - 1e-9, width):
        select = (times >= left - 1e-9) & (times < left + width - 1e-9)
        if not np.any(select):
            raise ValueError("Empty feature bin")
        values.append(epoch[..., select].astype(np.float64).mean(axis=-1))
    return np.stack(values, axis=-1).astype(np.float32)


def filter_chunk(values, hp, lp, state_hp, state_lp):
    x = values.astype(np.float64, copy=True) * 1e6
    x[:128] -= x[:128].mean(axis=0, keepdims=True)
    x, state_hp = sosfilt(hp, x, axis=1, zi=state_hp)
    x, state_lp = sosfilt(lp, x, axis=1, zi=state_lp)
    return x.astype(np.float32), state_hp, state_lp


def prepare_subject(root, subject, canonical, config, output):
    import mne
    source = source_path(root, subject)
    source_files = sorted(source.parent.glob(source.name + "*"))
    provenance = [{"kind": p.name[len(source.name):] or ".cdt", "bytes": p.stat().st_size,
                   "sha256": digest_file(p)} for p in source_files]
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in source_files]
    cache_path = output / "features" / f"{subject}.npz"
    qc_path = output / "qc" / f"{subject}.json"
    if cache_path.exists() and qc_path.exists():
        qc = json.loads(qc_path.read_text())
        if qc["config_sha256"] != config_digest(config) or qc["sources"] != provenance:
            raise ValueError("Existing cache has different configuration or sources; choose a fresh output")
        return qc
    raw = mne.io.read_raw_curry(source, preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    if fs != config["sampling"]["native_hz"]:
        raise ValueError(f"Unexpected native sampling rate for {subject}: {fs}")
    names = list(raw.ch_names[:128]) + config["eog_channels"]
    required = list(dict.fromkeys(config["primary_montage"] + config["comparison_montage"]))
    if len(raw.ch_names) != 131 or any(c not in names[:128] for c in required):
        raise ValueError(f"Unexpected montage for {subject}")
    selected = [names.index(c) for c in required]
    picks = [raw.ch_names.index(c) for c in names]
    f = config["filter"]
    hp = butter(f["highpass_order"], f["highpass_hz"], btype="highpass", fs=fs, output="sos")
    lp = butter(f["lowpass_order"], f["lowpass_hz"], btype="lowpass", fs=fs, output="sos")
    state_hp = np.zeros((len(hp), 130, 2))
    state_lp = np.zeros((len(lp), 130, 2))
    continuous_path = output / "runtime" / f"{subject}_continuous.dat"
    continuous = np.memmap(continuous_path, mode="w+", dtype="float32", shape=(130, raw.n_times))
    for start in range(0, raw.n_times, 25000):
        end = min(start + 25000, raw.n_times)
        block, state_hp, state_lp = filter_chunk(raw.get_data(picks=picks, start=start, stop=end), hp, lp, state_hp, state_lp)
        continuous[:, start:end] = block

    annotations = sorted(zip(raw.annotations.onset, raw.annotations.description), key=lambda x: x[0])
    events = [(float(onset), int(code)) for onset, code in annotations if str(code) in STIM_CODE_TEXT]
    onsets = np.array([e[0] for e in events])
    codes = np.array([e[1] for e in events])
    samples = raw.time_as_index(onsets, use_rounding=True)
    blocks = np.cumsum(np.r_[0, np.diff(onsets) > 5]).astype(int)
    sequence = np.zeros(len(events), dtype=int)
    position = np.zeros(len(events), dtype=int)
    unique_blocks = np.unique(blocks)
    for block in unique_blocks:
        idx = np.flatnonzero(blocks == block)
        sequence[idx] = match_sequence(codes[idx], canonical)
        position[idx] = np.arange(len(idx))
    if set(sequence) != set(range(1, 9)) or len(unique_blocks) != 8:
        raise ValueError(f"Incomplete sequence coverage: {subject}")
    current_offsets = np.arange(round(-.1 * fs), round(.8 * fs))
    times = current_offsets / fs
    limits = config["artifacts"]
    fields = {k: [] for k in ("eeg", "prestim", "eog", "metadata", "event_code", "sequence", "position", "trial", "keep")}
    for trial, (sample, code) in enumerate(zip(samples, codes)):
        if sample + current_offsets[0] < 0 or sample + current_offsets[-1] >= raw.n_times:
            raise ValueError(f"Epoch crosses recording boundary: {subject}, trial {trial}")
        if code not in FACE_CODES:
            continue
        epoch = np.asarray(continuous[:, sample + current_offsets])
        eeg, eog = epoch[selected], epoch[128:]
        eeg_ptp = float(np.ptp(eeg, axis=1).max())
        eog_ptp = float(np.ptp(eog, axis=1).max())
        keep = eeg_ptp <= limits["eeg_peak_to_peak_uv"] and eog_ptp <= limits["eog_peak_to_peak_uv"]
        values = {"eeg": bins(signed_baseline(eeg, times), times, 0, .6, .04),
                  "prestim": bins(eeg, times, -.1, 0, .02),
                  "eog": bins(eog, times, 0, .6, .04),
                  "metadata": [position[trial], blocks[trial], trial],
                  "event_code": code, "sequence": sequence[trial], "position": position[trial], "trial": trial,
                  "keep": keep}
        for key, value in values.items():
            fields[key].append(value)
    arrays = {k: np.asarray(v) for k, v in fields.items()}
    arrays["channel_names"] = np.asarray(required)
    np.savez_compressed(cache_path, **arrays)
    after = [(p.stat().st_size, p.stat().st_mtime_ns) for p in source_files]
    if before != after:
        raise RuntimeError("Source file changed during processing")
    qc = {"subject": subject,
          "config_sha256": config_digest(config), "sources": provenance,
          "native_hz": fs, "n_raw_channels": len(raw.ch_names), "n_events": len(events),
          "n_faces": len(fields["keep"]), "n_retained": int(np.sum(fields["keep"])),
          "retained_by_code": {str(c): int(np.sum((arrays["event_code"] == c) & arrays["keep"])) for c in FACE_CODES},
          "source_stats_unchanged": True}
    save_json(qc_path, qc)
    raw.close()
    del continuous
    continuous_path.unlink()
    return qc


def prepare(root: Path, output: Path, config: dict):
    for name in ("features", "qc", "runtime"):
        (output / name).mkdir(parents=True, exist_ok=True)
    canonical, log_audit = canonical_sequences(root)
    save_json(output / "canonical_sequences.json", canonical)
    expected = sorted(s for g in config["participant_groups"] for s in g)
    available = sorted(p.name for p in root.glob("S*") if p.is_dir())
    if expected != available:
        raise ValueError(f"Available subjects differ from the configured analysis population: {available}")
    qc = []
    for index, subject in enumerate(expected, 1):
        print(f"Preparing {subject} ({index}/{len(expected)})", flush=True)
        qc.append(prepare_subject(root, subject, canonical, config, output))
    save_json(output / "qc_summary.json", {"subjects": qc, "gentask": log_audit,
              "n_included": len(qc), "n_faces": sum(x["n_faces"] for x in qc),
              "n_retained": sum(x["n_retained"] for x in qc)})
