#!/usr/bin/env python3
"""Run the single-trial EEG analysis."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

from ai_faces_study.io import config_digest, digest_file, guard_output, load_config, save_json

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "ai_faces_study"
STAGES = ("prepare", "analyze", "infer", "report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "study.json")
    args = parser.parse_args()
    stages = STAGES if args.stage == "all" else (args.stage,)
    if "prepare" in stages and args.raw_root is None:
        parser.error("prepare requires --raw-root")
    config = load_config(args.config)
    config_hash = config_digest(config)
    output = args.output
    entrypoint = Path(__file__).resolve()
    protected_paths = [PACKAGE, ROOT / ".git", ROOT / "study.json", entrypoint, args.config]
    if args.raw_root is not None:
        protected_paths.append(args.raw_root)
    guard_output(output, protected_paths)
    runtime = output.resolve() / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
    os.environ["_MNE_FAKE_HOME_DIR"] = str(runtime)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    implementation = {str(p.relative_to(ROOT)): digest_file(p)
                      for p in sorted(PACKAGE.glob("*.py"))}
    implementation_path = output / "frozen_implementation.json"
    if implementation_path.exists():
        if json.loads(implementation_path.read_text()) != implementation:
            raise ValueError("Analysis implementation changed; use a fresh output directory")
    else:
        save_json(implementation_path, implementation)

    code_hashes = {**implementation, entrypoint.name: digest_file(entrypoint)}

    def record_stage(stage):
        packages = ("numpy", "scipy", "mne", "curryreader", "scikit-learn", "pandas", "matplotlib")
        save_json(output / f"provenance_{stage}.json", {
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {name: version(name) for name in packages},
            "config_sha256": config_hash,
            "code_sha256": code_hashes,
        })

    previous_preparation = output / "provenance_prepare.json"
    if previous_preparation.exists():
        recorded = json.loads(previous_preparation.read_text())["code_sha256"]
        raw_key = "ai_faces_study/raw.py"
        if recorded.get(raw_key) != code_hashes[raw_key]:
            raise ValueError("Feature-extraction code changed; use a fresh output directory")
    feature_manifest = output / "feature_manifest.json"
    if feature_manifest.exists():
        for relative, expected_hash in json.loads(feature_manifest.read_text()).items():
            if digest_file(output / relative) != expected_hash:
                raise ValueError("Prepared feature cache changed; do not reuse this run")
    frozen = output / "frozen_config.json"
    if frozen.exists():
        if json.loads(frozen.read_text())["config_sha256"] != config_hash:
            raise ValueError("Frozen configuration differs; use a different output directory")
    else:
        save_json(frozen, {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                          "config_sha256": config_hash, "config": config,
                          "status": "retrospective"})
    for stage in stages:
        if stage == "prepare":
            from ai_faces_study.raw import prepare
            prepare(args.raw_root, output, config)
            caches = sorted((output / "features").glob("S*.npz"))
            save_json(feature_manifest, {str(path.relative_to(output)): digest_file(path) for path in caches})
        elif stage == "analyze":
            from ai_faces_study.analysis import analyze
            analyze(output, config)
        elif stage == "infer":
            from ai_faces_study.inference import infer
            infer(output, config)
        elif stage == "report":
            from ai_faces_study.reporting import report
            report(output, config)
        record_stage(stage)
    if args.stage == "all":
        record_stage("all")


if __name__ == "__main__":
    main()
