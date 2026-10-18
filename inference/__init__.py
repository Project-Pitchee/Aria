"""Aria v0.2.3.3 cold-start inference package.

    from inference.predict import load_package, predict_recording, score_segment

    package = load_package()                  # weights/model.json + weights/g_table.json
    out = predict_recording("rec.wav", package, vad_model="SileroVAD.onnx")

``segment_to_score`` (``score_segment``) is the measurement kernel proper:
segment A in, ABI' / y_hat_mm / S out. ``predict_recording`` prepends stage 0 so
a raw recording can be fed in; that path is validated only on PVQD protocol
compound recordings.
"""

from inference.predict import (abi_prime, load_package, nine_features,
                               predict_corpus, predict_recording, resolve_vad_model,
                               score_segment, severity_from_abi, statuses)

__all__ = [
    "load_package", "abi_prime", "severity_from_abi", "nine_features",
    "score_segment", "predict_recording", "predict_corpus", "statuses",
    "resolve_vad_model",
]
