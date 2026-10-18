"""Batch stage-0 runner over a corpus of recordings (PVQD first).

Outputs, all under ``--out``:

``reports/<stem>.json``   the 21-field Schema v0.2.1 report per recording
``diagnostics/<stem>.json`` gate list, bypass evidence, periodicity stats
``segments/<stem>.wav``   segment A at 16 kHz mono (``--write-audio``)
``summary.json``          aggregate counts the milestone report cites
``env.json``              interpreter, library versions, constant table, hashes

The runner never touches the locked test speakers unless ``--include-locked``
is passed explicitly; the default reads the lock list from ``--lock-test`` when
one is supplied.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from . import constants as C
    from .audio import load_wav
    from .pipeline import run_stage0
    from .report import SCHEMA_FIELDS, field_completeness
    from .vad import SileroVad
except ImportError:
    import constants as C
    from audio import load_wav
    from pipeline import run_stage0
    from report import SCHEMA_FIELDS, field_completeness
    from vad import SileroVad


def _read_lock_list(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def speaker_from_stem(stem: str) -> str:
    """PVQD speaker id is the stem up to the first separator."""
    for sep in (" ", "_"):
        if sep in stem:
            return stem.split(sep, 1)[0]
    return stem


def environment() -> dict:
    import onnxruntime
    import scipy
    import soundfile

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "soundfile": soundfile.__version__,
        "onnxruntime": onnxruntime.__version__,
        "constant_table": C.constant_table(),
        "vad_model_sha256": C.VAD_MODEL_SHA256,
        "seg_ver": C.SEG_VER,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--lock-test", type=Path, default=None,
                        help="speaker-per-line lock list to exclude")
    parser.add_argument("--include-locked", action="store_true")
    parser.add_argument("--write-audio", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    files = sorted(args.audio_dir.glob("*.wav"))
    if args.limit:
        files = files[:args.limit]
    locked = _read_lock_list(args.lock_test)

    args.out.mkdir(parents=True, exist_ok=True)
    for sub in ("reports", "diagnostics", "segments"):
        (args.out / sub).mkdir(exist_ok=True)
    (args.out / "env.json").write_text(
        json.dumps(environment(), ensure_ascii=False, indent=2))

    vad_kwargs = {"model_path": args.model} if args.model else {}
    detector = SileroVad(**vad_kwargs)

    rows = []
    started = time.time()
    skipped_locked = 0
    for index, path in enumerate(files, 1):
        stem = path.stem
        speaker = speaker_from_stem(stem)
        if locked and speaker in locked and not args.include_locked:
            skipped_locked += 1
            continue
        audio = load_wav(path)
        result = run_stage0(audio, vad=detector)
        (args.out / "reports" / f"{stem}.json").write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2, allow_nan=False))
        (args.out / "diagnostics" / f"{stem}.json").write_text(
            json.dumps({**result.diagnostics, "source_sha256": audio.source_sha256},
                       ensure_ascii=False, indent=2, allow_nan=False, default=str))
        if args.write_audio:
            import soundfile as sf

            sf.write(str(args.out / "segments" / f"{stem}.wav"),
                     result.segment_a.astype(np.float32), C.TARGET_SR)
        rows.append({
            "stem": stem,
            "speaker_id": speaker,
            "report": result.report,
            "diagnostics": result.diagnostics,
        })
        print(f"[{index}/{len(files)}] {stem}  T_voiced={result.report['T_voiced']:.2f}s "
              f"reject={result.report['reject_code']}")

    summary = summarize(rows, skipped_locked=skipped_locked,
                        elapsed_s=time.time() - started)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("per_file",)}, ensure_ascii=False, indent=2))
    return 0


def summarize(rows: list[dict], *, skipped_locked: int, elapsed_s: float) -> dict:
    """Aggregate the milestone-report numbers from per-file rows."""
    n = len(rows)
    codes = Counter()
    for row in rows:
        if row["report"]["reject"]:
            codes[row["report"]["reject_code"]] += 1
        else:
            codes["pass"] += 1
    triggered = Counter()
    for row in rows:
        for gate in row["diagnostics"]["triggered_gates"]:
            triggered[gate] += 1

    completeness = Counter()
    for row in rows:
        for name, info in field_completeness(row["report"]).items():
            if info["complete"]:
                completeness[name] += 1

    def values(key):
        return np.asarray([row["report"][key] for row in rows], dtype=np.float64)

    return {
        "files": n,
        "skipped_locked": skipped_locked,
        "elapsed_s": elapsed_s,
        "bypass_severe": int(sum(row["report"]["bypass_severe"] for row in rows)),
        "reject_distribution": dict(sorted(codes.items())),
        "gate_trigger_counts": dict(sorted(triggered.items())),
        "field_complete_fraction": {name: completeness[name] / n if n else 0.0
                                    for name in SCHEMA_FIELDS},
        "T_raw": _describe(values("T_raw")),
        "T_speech": _describe(values("T_speech")),
        "T_voiced": _describe(values("T_voiced")),
        "voiced_ratio": _describe(values("voiced_ratio")),
        "vowel_coverage": _describe(values("vowel_coverage")),
        "snr_db": _describe(values("snr_db")),
        "d": _describe(values("d")),
        "miss_rate": _describe(values("miss_rate")),
        "per_file": rows,
    }


def _describe(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": int(values.size), "finite": 0}
    return {
        "n": int(values.size),
        "finite": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
