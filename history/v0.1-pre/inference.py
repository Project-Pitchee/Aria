"""Minimal Aria v0.1-pre inference (NumPy only).

Input rows are raw canonical features in the order
[14 acoustic | 192 embedding mean | 192 embedding std].  The model uses
F2=[14 acoustic | 192 embedding mean] plus 14 availability indicators.
No labels are loaded or required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_LAYOUT = "acoustic14_embedding_mean192_embedding_std192"
N_ACOUSTIC = 14
N_EMBED = 192
N_RAW = 398
N_MODEL = 220


def _json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_artifacts(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    weights = _json(root / "weights" / "linear_head.json")
    scaler = _json(root / "weights" / "scaler.json")
    harm = _json(root / "weights" / "harmonization.json")
    imp = _json(root / "weights" / "imputation.json")
    w = np.asarray(weights["weight"], dtype=np.float64)
    if w.shape != (N_MODEL,) or not np.isfinite(w).all():
        raise ValueError("linear head must contain 220 finite weights")
    b = float(weights["bias"])
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    medians = np.asarray(imp["medians"], dtype=np.float64)
    if mean.shape != (206,) or scale.shape != (206,) or (scale <= 0).any():
        raise ValueError("invalid 206-column scaler")
    if medians.shape != (14,) or not np.isfinite(medians).all():
        raise ValueError("invalid 14-column imputation medians")
    if not np.isfinite(b) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("non-finite model artifact")
    return {"weight": w, "bias": b, "mean": mean, "scale": scale,
            "medians": medians, "harmonization": harm, "imputation": imp}


def _sentinel_mask(acoustic: np.ndarray, policy: str) -> np.ndarray:
    if policy != "core_v1":
        raise ValueError(f"unsupported sentinel policy: {policy}")
    mask = np.zeros(acoustic.shape, dtype=bool)
    mask[:, 0] = acoustic[:, 0] == -200.0
    f0_failure = acoustic[:, 8] == 0.0
    mask[f0_failure, 8:11] = True
    return mask


def transform_features(X: np.ndarray, *, acoustic_available: np.ndarray | None,
                       artifacts: dict[str, Any], feature_layout: str) -> np.ndarray:
    """Apply the frozen sentinel, imputation, scaler and indicator contract."""
    if feature_layout != CANONICAL_LAYOUT:
        raise ValueError("feature_layout must declare the canonical acoustic-first 398-column layout")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != N_RAW:
        raise ValueError("X must have shape (n, 398): [14 acoustic | 192 mean | 192 std]")
    if not np.isfinite(X[:, N_ACOUSTIC:]).all() or np.isinf(X[:, :N_ACOUSTIC]).any():
        raise ValueError("embedding features and infinities must be finite; acoustic NaN is allowed")
    raw_ac = X[:, :N_ACOUSTIC].copy()
    if acoustic_available is None:
        available = np.isfinite(raw_ac)
    else:
        available = np.asarray(acoustic_available)
        if available.shape != raw_ac.shape or not np.isin(available, (0, 1, False, True)).all():
            raise ValueError("acoustic_available must have shape (n, 14) and contain only 0/1")
        available = available.astype(bool) & np.isfinite(raw_ac)
    sentinels = _sentinel_mask(raw_ac, artifacts["imputation"]["policy"])
    trusted = available & ~sentinels
    filled = raw_ac
    filled[~trusted] = np.take(artifacts["medians"], np.nonzero(~trusted)[1])
    selected = np.column_stack([filled, X[:, N_ACOUSTIC:N_ACOUSTIC + N_EMBED]])
    standardized = (selected - artifacts["mean"]) / artifacts["scale"]
    return np.column_stack([standardized, trusted.astype(np.float64)])


def predict(X: np.ndarray, *, acoustic_available: np.ndarray | None = None,
            feature_layout: str = CANONICAL_LAYOUT,
            artifacts: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    artifacts = load_artifacts() if artifacts is None else artifacts
    processed = transform_features(X, acoustic_available=acoustic_available,
                                   artifacts=artifacts, feature_layout=feature_layout)
    z = processed @ artifacts["weight"] + artifacts["bias"]
    harm = artifacts["harmonization"]
    native = z * float(harm["sd_mm"]) + float(harm["mean_mm"])
    return {"normalized_severity_z": z, "severity_mm": native,
            "availability": processed[:, 206:].astype(bool)}


def _load_input(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False), None, CANONICAL_LAYOUT
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            if "X" not in d:
                raise ValueError("NPZ input must contain X")
            X = d["X"]
            availability = d["acoustic_available"] if "acoustic_available" in d else None
            if "feature_layout" in d:
                value = d["feature_layout"][0]
                layout = value.decode() if isinstance(value, bytes) else str(value)
            else:
                layout = ""
        return X, availability, layout
    raise ValueError("input must be .npy or .npz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="raw (n,398) .npy or canonical .npz; labels are ignored")
    ap.add_argument("--output", required=True, type=Path,
                    help="JSON prediction output")
    ap.add_argument("--feature-layout", default=CANONICAL_LAYOUT,
                    help="explicit layout declaration for .npy input")
    args = ap.parse_args()
    X, availability, embedded_layout = _load_input(args.input)
    layout = args.feature_layout if args.input.suffix == ".npy" else embedded_layout
    result = predict(X, acoustic_available=availability, feature_layout=layout)
    payload = {"model": "aria-v0.1-pre", "n": int(len(X)),
               "normalized_severity_z": result["normalized_severity_z"].tolist(),
               "severity_mm": result["severity_mm"].tolist(),
               "availability": result["availability"].astype(int).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"model": payload["model"], "n": payload["n"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
