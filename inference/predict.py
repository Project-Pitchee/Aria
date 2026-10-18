#!/usr/bin/env python3
"""Aria v0.2.3.3 cold-start inference entry point.

Two input contracts, both supported here:

    raw recording       -> stage 0 -> segment A -> 9 native features -> ABI' -> S
    segment A (16 kHz)  ->            9 native features -> ABI' -> S

Chain (SSOT §3.3, see ``../model_card.md``):

    stage 0 (seg_ver stage0-v0.2.1.2)  ->  segment A, 16 kHz mono float
    native 9-feature engine            ->  x_1..x_9
    L3 coefficients (weights/model.json)
        ABI'_raw = b0 + sum_j b_j x_j
        ABI'     = clip(ABI'_raw, 0, 10)          G6 is inactive (identity)
    dimension chain g (weights/g_table.json)
        S = g(ABI')  in [0, 100]
    y_hat_mm = 10 * ABI'   (PVQD CAPE-V overall severity, mm)

Only the 57 scorable recordings of the §2.5 locked benchmark were produced
through the raw-recording path; anything that is not a PVQD protocol compound
recording is outside the validated scope of this package.

Dependencies: numpy, scipy (g tails), soundfile, onnxruntime (stage 0 VAD only).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import norm

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from inference.native_features import abi_native as N  # noqa: E402
from inference.stage0 import constants as C  # noqa: E402
from inference.stage0.audio import load_wav  # noqa: E402
from inference.stage0.pipeline import run_stage0  # noqa: E402
from inference.stage0.vad import SileroVad  # noqa: E402

ABI_DECIMALS = 10
GATE_ORDER = ("R1", "R2", "R3", "R4", "R5", "M_T5", "W_窗数")
# The package's own caliber, i.e. the one whose numbers the §2.5 locked
# benchmark reports: R1 and W_窗数 waived, R3 waived when voiced_ratio >= 0.30.
# R2 / R4 / R5 / M_T5 and the severe bypass reject as usual.
RELAXABLE = {"R1", "W_窗数"}
R3_WAIVE_VOICED_RATIO = 0.30


# --------------------------------------------------------------------------- #
# package loading
# --------------------------------------------------------------------------- #
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_vad_model(vad_model=None) -> Path:
    """Silero v5 ONNX path: explicit > $ARIA_VAD_MODEL > package-bundled copy."""
    import os

    candidates = [vad_model, os.environ.get("ARIA_VAD_MODEL"),
                  Path(__file__).resolve().parent / "models" / "SileroVAD.onnx"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            path = Path(candidate)
            actual = sha256_file(path)
            if actual != C.VAD_MODEL_SHA256:
                raise SystemExit(
                    f"VAD model SHA-256 mismatch: {path} is {actual}, the pinned "
                    f"stage-0 constant table requires {C.VAD_MODEL_SHA256}")
            return path
    raise SystemExit(
        "no Silero VAD model found. Pass --vad-model PATH, set $ARIA_VAD_MODEL, or "
        f"place the pinned SileroVAD.onnx (SHA-256 {C.VAD_MODEL_SHA256}) at "
        f"{Path(__file__).resolve().parent / 'models' / 'SileroVAD.onnx'}")


def load_package(root=None) -> dict:
    """Load weights/model.json + weights/g_table.json and verify they bind."""
    root = Path(root) if root else _PACKAGE_ROOT
    model_path, g_path = root / "weights" / "model.json", root / "weights" / "g_table.json"
    model = json.loads(model_path.read_text())
    g_table = json.loads(g_path.read_text())
    want = model["g_table"]["sha256"]
    actual = sha256_file(g_path)
    if want != actual:
        raise SystemExit(f"g_table.json SHA-256 mismatch: model.json says {want}, file is {actual}")
    return {
        "root": root,
        "model_path": model_path,
        "g_table_path": g_path,
        "model_sha256": sha256_file(model_path),
        "g_table_sha256": actual,
        "model": model,
        "g_table": g_table,
        "feature_names": list(model["feature_names"]),
        "b0": float(model["abi_prime_raw"]["coefficients"]["b0"]),
        "b": np.asarray(model["abi_prime_raw"]["coefficients"]["b"], dtype=float),
        "clip": tuple(float(v) for v in model["abi_prime_raw"]["clip"]),
    }


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
def abi_prime(values: dict, package: dict) -> float:
    """L3 linear head on the nine native features, clipped to [0, 10]."""
    x = np.array([values[k] for k in package["feature_names"]], dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("ABI' needs all nine features finite")
    raw = package["b0"] + float(x @ package["b"])
    lo, hi = package["clip"]
    return float(np.clip(raw, lo, hi))


def severity_from_abi(abi: float, package: dict) -> float:
    """Dimension chain g: ABI' in [0, 10] -> S in [0, 100]."""
    table = package["g_table"]
    r = np.asarray(table["knots"], dtype=float)
    p = np.asarray(table["p"], dtype=float)
    sigma = float(table["sigma"])
    if abi < r[0]:
        return float(100.0 * norm.cdf(norm.ppf(p[0]) - (r[0] - abi) / sigma))
    if abi > r[-1]:
        return float(100.0 * norm.cdf(norm.ppf(p[-1]) + (abi - r[-1]) / sigma))
    return float(100.0 * np.interp(abi, r, p))


def nine_features(segment_a: np.ndarray, sample_rate: int = C.TARGET_SR) -> dict:
    """Nine native ABI variables; NaN where Praat would be undefined."""
    segment_a = np.asarray(segment_a, dtype=np.float32).astype(np.float64)
    return N.segment_features(segment_a, sample_rate)


def score_segment(segment_a: np.ndarray, package: dict,
                  sample_rate: int = C.TARGET_SR) -> dict:
    """segment A -> nine features -> ABI' -> S."""
    values = nine_features(segment_a, sample_rate)
    finite = {k: (float(v) if v is not None and np.isfinite(v) else None)
              for k, v in values.items()}
    complete = all(finite[k] is not None for k in package["feature_names"])
    out = {"features": finite, "abi_prime": None, "y_hat_mm": None, "s": None}
    if complete:
        abi = abi_prime(finite, package)
        out.update(abi_prime=abi, y_hat_mm=10.0 * abi,
                   s=severity_from_abi(abi, package))
    return out


def segment_from_recording(audio_path, vad_model=None, vad: SileroVad | None = None) -> dict:
    """Raw recording -> stage 0 -> segment A + 21-field report + diagnostics."""
    audio_path = Path(audio_path)
    if vad is None:
        vad = SileroVad(model_path=resolve_vad_model(vad_model))
    result = run_stage0(load_wav(audio_path), vad=vad)
    return {
        "sha256": sha256_file(audio_path),
        "report": result.report,
        "diagnostics": _jsonable(result.diagnostics),
        "segment_a": result.segment_a.astype(np.float32),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# --------------------------------------------------------------------------- #
# gate calibers
# --------------------------------------------------------------------------- #
def statuses(report: dict, diagnostics: dict, values) -> dict:
    """The two published calibers for one recording.

    ``training`` is the caliber of this package's own numbers (the §2.5 locked
    benchmark and the dev metrics): R1 and W_窗数 waived, R3 waived when
    ``voiced_ratio >= 0.30``. ``product`` fires all seven gates.
    """
    gates = tuple(g for g in GATE_ORDER if g in set(diagnostics["triggered_gates"]))

    def reject_or(reason):
        return f"rejected:{reason}"

    if report["bypass_severe"]:
        product = training = "bypass_severe"
        return {"product": product, "training": training, "triggered_gates": list(gates),
                "reject_code": None}

    product = (reject_or(report["reject_code"]) if report["reject"] else "scored")

    remaining = set(gates) - RELAXABLE
    if "R3" in remaining and report["voiced_ratio"] >= R3_WAIVE_VOICED_RATIO:
        remaining.discard("R3")
    if remaining:
        training = reject_or("|".join(g for g in GATE_ORDER if g in remaining))
    elif values is None or any(values.get(k) is None for k in N.FEATURE_NAMES):
        training = reject_or("missing_features")
    else:
        training = "scored"
    return {"product": product, "training": training,
            "triggered_gates": list(gates), "reject_code": report["reject_code"]}


def _distance_bin(distance_cm) -> str | None:
    value = f"{int(distance_cm)}cm"
    return value if value in ("15cm", "30cm", "50cm") else None


# --------------------------------------------------------------------------- #
# one recording, end to end
# --------------------------------------------------------------------------- #
def predict_recording(audio_path, package: dict, vad_model=None, vad=None,
                      distance_cm: int = 6, return_segment: bool = False) -> dict:
    # the pipeline's segment A is a 16 kHz float32 mono waveform; the cast keeps
    # an in-memory run bit-identical to one that round-trips the written WAV.
    seg = segment_from_recording(audio_path, vad_model=vad_model, vad=vad)
    report, diagnostics = seg["report"], seg["diagnostics"]
    scored = score_segment(seg["segment_a"], package)
    st = statuses(report, diagnostics, scored["features"])
    out = {
        "audio_path": str(audio_path),
        "audio_sha256": seg["sha256"],
        "stage0": {"seg_ver": report["seg_ver"], "report": report, "diagnostics": diagnostics},
        "segment_a": {"seconds": float(len(seg["segment_a"]) / C.TARGET_SR),
                      "sample_rate": C.TARGET_SR},
        **scored,
        "status": st["training"],
        "status_product_caliber": st["product"],
        "triggered_gates": st["triggered_gates"],
    }
    if return_segment:
        out["_segment_a"] = seg["segment_a"]
    return out


# --------------------------------------------------------------------------- #
# batch mode (one row per request, corpus order preserved)
# --------------------------------------------------------------------------- #
_VAD = None
_VAD_MODEL = None


def _init_worker(vad_model):
    global _VAD, _VAD_MODEL
    _VAD_MODEL = vad_model
    _VAD = SileroVad(model_path=resolve_vad_model(vad_model))


def _batch_job(job):
    index, request, dataset, outdir, write_audio = job
    try:
        path = Path(dataset) / request["audio_path"]
        if request.get("audio_sha256") and sha256_file(path) != request["audio_sha256"]:
            raise ValueError(f"sha256 mismatch for {request['audio_path']}")
        row = predict_recording(path, load_package(), distance_cm=request.get("distance_cm", 6),
                                vad=_VAD, return_segment=write_audio)
        if write_audio:
            import soundfile as sf

            target = Path(outdir) / "segments" / request["audio_path"]
            if target.suffix.lower() != ".wav":
                target = target.with_suffix(target.suffix + ".wav")
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(target), row.pop("_segment_a"), C.TARGET_SR, subtype="FLOAT")
        row["speaker_id"] = request.get("speaker_id")
        return index, row, None
    except Exception as exc:  # noqa: BLE001 - one bad file must not lose the run
        return index, None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"


def predict_corpus(requests, dataset, outdir, package: dict, vad_model=None,
                   workers: int = 8, write_audio: bool = False) -> dict:
    outdir = Path(outdir)
    if outdir.exists():
        raise SystemExit(f"refusing to overwrite existing output {outdir}")
    outdir.mkdir(parents=True)
    jobs = [(i, r, str(dataset), str(outdir), write_audio)
            for i, r in enumerate(requests)]
    results: list = [None] * len(jobs)
    errors: dict = {}
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                         initargs=(vad_model,)) as pool:
            for index, row, error in pool.imap_unordered(_batch_job, jobs, chunksize=2):
                results[index] = row
                if error:
                    errors[index] = error
    else:
        _init_worker(vad_model)
        for job in jobs:
            index, row, error = _batch_job(job)
            results[index] = row
            if error:
                errors[index] = error

    primary, product = [], []
    prim_dist, prod_dist = Counter(), Counter()
    for request, row, index in zip(requests, results, range(len(results))):
        sid = request.get("speaker_id")
        if row is None:
            reason = errors.get(index, "unknown failure")
            primary.append([sid, "", "", f"error:{reason}"])
            product.append([sid, f"error:{reason}", "", ""])
            prim_dist[f"error:{reason}"] += 1
            prod_dist[f"error:{reason}"] += 1
            continue
        prim_dist[row["status"]] += 1
        prod_dist[row["status_product_caliber"]] += 1
        # The published interface emits a prediction only for scored rows; a
        # rejected row carries no ŷ, even when the nine features were computable.
        if row["status"] != "scored" or row["abi_prime"] is None:
            primary.append([sid, "", "", row["status"]])
        else:
            primary.append([sid, f"{row['abi_prime']:.{ABI_DECIMALS}f}",
                            f"{row['y_hat_mm']:.{ABI_DECIMALS}f}", row["status"]])
        product.append([sid, row["status_product_caliber"],
                        row["status_product_caliber"].split(":")[-1]
                        if ":" in row["status_product_caliber"] else "",
                        "|".join(row["triggered_gates"])])

    _write_csv(outdir / "predictions.csv",
               ["speaker_id", "abi_prime", "y_hat_mm", "status"], primary)
    _write_csv(outdir / "predictions_product_caliber.csv",
               ["speaker_id", "status", "reject_code", "triggered_gates"], product)

    summary = {
        "model_ver": package["model"]["model_ver"],
        "seg_ver": C.SEG_VER,
        "feature_engine": "native",
        "status_caliber": ("training / package caliber: R1 & W_窗数 waived, R3 waived when "
                           "voiced_ratio>=0.30; R2/R4/R5/M_T5 & bypass reject"),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__},
        "package": {"model_json_sha256": package["model_sha256"],
                    "g_table_json_sha256": package["g_table_sha256"]},
        "n_requests": len(requests),
        "status_distribution": dict(sorted(prim_dist.items())),
        "product_status_distribution": dict(sorted(prod_dist.items())),
        "errors": {str(k): v for k, v in sorted(errors.items())},
        "outputs_sha256": {
            "predictions.csv": sha256_file(outdir / "predictions.csv"),
            "predictions_product_caliber.csv":
                sha256_file(outdir / "predictions_product_caliber.csv"),
        },
    }
    (outdir / "run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, default=_PACKAGE_ROOT,
                       help="package root holding weights/ (default: this package)")
    parser.add_argument("--vad-model", type=Path,
                       help="SileroVad.onnx; default $ARIA_VAD_MODEL or inference/models/")
    parser.add_argument("--audio", type=Path, help="one raw recording")
    parser.add_argument("--segment-a", type=Path, help="one segment A WAV (16 kHz mono)")
    parser.add_argument("--requests", type=Path, help="requests.jsonl: audio_path, "
                       "audio_sha256, distance_cm, speaker_id")
    parser.add_argument("--dataset", type=Path, help="root the request audio_paths resolve in")
    parser.add_argument("--out", type=Path, help="output JSON for --audio / --segment-a")
    parser.add_argument("--outdir", type=Path, help="output directory for --requests")
    parser.add_argument("--write-audio", action="store_true",
                       help="also write each segment A under --outdir/segments/")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    package = load_package(args.package)
    if args.requests:
        if not args.dataset or not args.outdir:
            parser.error("--requests needs --dataset and --outdir")
        requests = [json.loads(line) for line in
                    args.requests.read_text().splitlines() if line.strip()]
        summary = predict_corpus(requests, args.dataset, args.outdir, package,
                                 vad_model=args.vad_model, workers=args.workers,
                                 write_audio=args.write_audio)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.segment_a:
        import soundfile as sf

        pcm, sr = sf.read(str(args.segment_a))
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        scored = score_segment(pcm, package, sr)
        out = {"source": str(args.segment_a), "sha256": sha256_file(args.segment_a),
               "segment_a": {"seconds": float(len(pcm) / sr), "sample_rate": sr}, **scored}
    elif args.audio:
        out = predict_recording(args.audio, package, vad_model=args.vad_model)
    else:
        parser.error("give one of --audio, --segment-a, --requests")

    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
