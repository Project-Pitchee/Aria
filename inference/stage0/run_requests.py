"""Run stage 0 over the G6 feature-request list (one row per endpoint).

Reads ``requests.jsonl`` (see docs/m2/README.md), resolves each ``audio_path``
against ``--dataset``, verifies the SHA-256 before touching the audio, and
writes under ``--out``:

``reports.jsonl``      one line per endpoint: request keys + 21-field report
``diagnostics.jsonl``  gate list, bypass evidence, periodicity stats
``segments/<path>``    segment A as 16 kHz float32 WAV (``--write-audio``)
``env.json``           interpreter, libraries, constant table, input hashes

Rows keep request order, so reruns diff line by line. The output directory
must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage0 import constants as C  # noqa: E402
from stage0.audio import load_wav  # noqa: E402
from stage0.pipeline import run_stage0  # noqa: E402
from stage0.run_corpus import environment  # noqa: E402
from stage0.vad import DEFAULT_MODEL_PATH, SileroVad  # noqa: E402

_VAD: SileroVad | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _init(model_path: str) -> None:
    global _VAD
    _VAD = SileroVad(model_path=model_path)


def _finite_or_null(value):
    """JSON has no NaN/Infinity; the G6 contract wants explicit null instead."""
    if isinstance(value, dict):
        return {k: _finite_or_null(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_null(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _distance_bin(request: dict) -> str | None:
    """Degraded endpoints carry a simulated 15/30/50 cm bin; clean is 6 cm source."""
    value = f"{int(request['distance_cm'])}cm"
    return value if value in ("15cm", "30cm", "50cm") else None


def _process(job: tuple[int, dict, str, str | None]) -> tuple[int, dict, dict, np.ndarray | None]:
    index, request, dataset, segment_dir = job
    path = Path(dataset) / request["audio_path"]
    if _sha256(path) != request["audio_sha256"]:
        raise ValueError(f"sha256 mismatch for {request['audio_path']}")
    audio = load_wav(path)
    result = run_stage0(audio, vad=_VAD, distance_bin=_distance_bin(request))
    diagnostics = _finite_or_null(result.diagnostics)
    segment = result.segment_a.astype(np.float32) if segment_dir else None
    return index, _finite_or_null(result.report), diagnostics, segment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--write-audio", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output {args.out}")
    requests = [json.loads(line) for line in args.requests.read_text().splitlines()
                if line.strip()]
    if args.limit:
        requests = requests[:args.limit]
    args.out.mkdir(parents=True)
    segment_dir = args.out / "segments" if args.write_audio else None

    env = environment()
    env.update({
        "requests_path": str(args.requests),
        "requests_sha256": _sha256(args.requests),
        "dataset": str(args.dataset),
        "vad_model_path": str(args.model),
        "vad_model_sha256_actual": _sha256(args.model),
    })
    if env["vad_model_sha256_actual"] != C.VAD_MODEL_SHA256:
        raise SystemExit("VAD model hash differs from the pinned constant table")
    (args.out / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2))

    jobs = [(i, r, str(args.dataset), str(segment_dir) if segment_dir else None)
            for i, r in enumerate(requests)]
    results: list[tuple[dict, dict] | None] = [None] * len(jobs)
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                      initargs=(str(args.model),)) as pool:
        for done, (index, report, diagnostics, segment) in enumerate(
                pool.imap_unordered(_process, jobs, chunksize=4), 1):
            results[index] = (report, diagnostics)
            if segment is not None:
                import soundfile as sf

                target = segment_dir / requests[index]["audio_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(target), segment, C.TARGET_SR, subtype="FLOAT")
            if done % 50 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {time.time() - started:.0f}s", flush=True)

    with (args.out / "reports.jsonl").open("w") as rep, \
            (args.out / "diagnostics.jsonl").open("w") as diag:
        for request, (report, diagnostics) in zip(requests, results):
            rep.write(json.dumps({**request, "quality_report": report},
                                 ensure_ascii=False, allow_nan=False) + "\n")
            diag.write(json.dumps({"audio_path": request["audio_path"], **diagnostics},
                                  ensure_ascii=False, allow_nan=False, default=str) + "\n")
    print(f"done {len(jobs)} endpoints in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
