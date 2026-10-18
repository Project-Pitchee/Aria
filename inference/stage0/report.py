"""Quality report Schema v0.2.1: exactly 21 fields, 5 groups (SSOT §2.1).

Field names and order are frozen. ``validate_report`` refuses missing keys and
unknown keys, so a schema drift cannot pass silently. Anything stage 0 computes
that is not a schema field (bypass evidence, gate list, periodicity stats) goes
into a separate diagnostics object that is never merged into the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

try:
    from . import constants as C
    from .gates import RejectDecision
except ImportError:
    import constants as C
    from gates import RejectDecision

SCHEMA_VERSION = "v0.2.1"

SCHEMA_GROUPS: dict[str, tuple[str, ...]] = {
    "G1_collection": ("device_id", "distance_bin", "input_spec"),
    "G2_segmentation": ("T_raw", "T_speech", "T_voiced", "voiced_ratio",
                        "voiced_ok", "bypass_severe", "vowel_coverage",
                        "seg_ver"),
    "G3_quality_fingerprint": ("q_raw", "q_dmg", "d", "snr_db", "dnsmos"),
    "G4_feature_missingness": ("miss_rate", "miss_code", "key3_missing"),
    # SSOT row 21 is one field with two components: reject_code / reject_text.
    "G5_rejection": ("reject", ("reject_code", "reject_text")),
}
# One report key per SSOT row. Row 21 renders as two JSON keys (reject_code,
# reject_text) but is a single schema entry; the slash keeps that visible.
ROW_KEYS = {"reject_code / reject_text": ("reject_code", "reject_text")}
# Rows as listed in SSOT §2.1, in order: 21 rows, one of which expands into
# two JSON keys. ``SCHEMA_FIELDS`` is therefore 21 rows; ``SCHEMA_KEYS`` is the
# 22 JSON keys that must all be present.
SCHEMA_FIELDS = tuple(name for group in SCHEMA_GROUPS.values() for name in group)
SCHEMA_KEYS = tuple(
    key
    for name in SCHEMA_FIELDS
    for key in ((name,) if isinstance(name, str) else ROW_KEYS[" / ".join(name)]))
assert len(SCHEMA_FIELDS) == 21, len(SCHEMA_FIELDS)
assert len(SCHEMA_KEYS) == 22, len(SCHEMA_KEYS)


@dataclass
class Stage0Report:
    """The 21 fields plus the audio A the report describes."""

    report: dict
    segment_a: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def to_json(self, indent: int | None = None) -> str:
        validate_report(self.report)
        return json.dumps(self.report, ensure_ascii=False, indent=indent,
                          allow_nan=False)


def validate_report(report: dict) -> None:
    keys = set(report)
    expected = set(SCHEMA_KEYS)
    missing = expected - keys
    unknown = keys - expected
    if missing:
        raise ValueError(f"report is missing schema fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"report has unknown keys outside Schema v0.2.1: {sorted(unknown)}")
    q_raw = report["q_raw"]
    if not isinstance(q_raw, (list, tuple)) or len(q_raw) != 8:
        raise ValueError("q_raw must have 8 entries")
    q_dmg = report["q_dmg"]
    if not isinstance(q_dmg, (list, tuple)) or len(q_dmg) != 4:
        raise ValueError("q_dmg must have 4 entries")
    if not all(0.0 <= float(v) <= 1.0 for v in q_dmg):
        raise ValueError("q_dmg entries must lie in [0, 1]")
    if not 0.0 <= float(report["d"]) <= 1.0:
        raise ValueError("d must lie in [0, 1]")
    if report["distance_bin"] not in (None, "15cm", "30cm", "50cm"):
        raise ValueError("distance_bin must be null or one of 15cm/30cm/50cm")
    if report["reject"] and not (report["reject_code"] and report["reject_text"]):
        raise ValueError("a rejected recording must carry both reject_code and reject_text")


def build_report(*, device_id: str | None,
                 distance_bin: str | None,
                 input_spec: dict,
                 t_raw_s: float, t_speech_s: float, t_voiced_s: float,
                 voiced_ratio: float, voiced_ok: bool, bypass_severe: bool,
                 vowel_coverage: float,
                 q_raw: np.ndarray, q_dmg: np.ndarray, d: float,
                 snr_db: float, dnsmos: dict | None,
                 miss_rate: float, miss_code: int, key3_missing: bool,
                 decision: RejectDecision) -> dict:
    """Assemble the 21 fields. ``dnsmos`` stays null until offline backfill."""
    report = {
        "device_id": device_id,
        "distance_bin": distance_bin,
        "input_spec": input_spec,
        "T_raw": float(t_raw_s),
        "T_speech": float(t_speech_s),
        "T_voiced": float(t_voiced_s),
        "voiced_ratio": float(voiced_ratio),
        "voiced_ok": bool(voiced_ok),
        "bypass_severe": bool(bypass_severe),
        "vowel_coverage": float(vowel_coverage),
        "seg_ver": C.SEG_VER,
        "q_raw": [float(v) for v in q_raw],
        "q_dmg": [float(v) for v in q_dmg],
        "d": float(d),
        "snr_db": float(snr_db),
        "dnsmos": dnsmos,
        "miss_rate": float(miss_rate),
        "miss_code": int(miss_code),
        "key3_missing": bool(key3_missing),
        "reject": bool(decision.reject),
        "reject_code": decision.reject_code,
        "reject_text": decision.reject_text,
    }
    validate_report(report)
    return report


def field_completeness(report: dict) -> dict:
    """Per-field presence audit for the milestone report.

    A field counts as complete when it is not null and, for numerics, finite.
    ``device_id`` / ``distance_bin`` are legitimately null for PVQD, which
    carries neither; the audit shows that instead of inventing values.
    """
    result = {}
    for group, names in SCHEMA_GROUPS.items():
        for name in names:
            value = report.get(name)
            complete = value is not None
            if isinstance(value, float):
                complete = complete and np.isfinite(value)
            if isinstance(value, (list, tuple)):
                complete = complete and all(
                    isinstance(v, float) and np.isfinite(v) or isinstance(v, (int, str))
                    for v in value)
            result[name] = {"group": group, "complete": bool(complete)}
    return result
