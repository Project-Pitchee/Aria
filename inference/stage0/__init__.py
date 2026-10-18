"""Aria v0.2.1 stage 0 (training side).

Public entry points:

``run_stage0``      one recording -> segment A + 21-field quality report
``SileroVad``       Silero v5 ONNX detector at the SSOT operating point
``validate_report`` Schema v0.2.1 contract check
"""

from .audio import LoadedAudio, load_wav
from .pipeline import Stage0Result, run_stage0
from .report import SCHEMA_FIELDS, SCHEMA_VERSION, validate_report
from .vad import SileroVad

__all__ = [
    "LoadedAudio", "load_wav", "run_stage0", "Stage0Result",
    "SileroVad", "SCHEMA_FIELDS", "SCHEMA_VERSION", "validate_report",
]
