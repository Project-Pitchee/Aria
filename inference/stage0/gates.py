"""Seven stage-0 reject gates (SSOT §3.1.4).

Priority follows the SSOT listing order R1, R2, R3, R4, R5, M_T5, W_窗数; the
first triggered gate supplies ``reject_code`` / ``reject_text`` and every
triggered gate is retained in ``RejectDecision.triggered`` so nothing is hidden.

``bypass_severe`` wins over every gate: a recording that looks like a severe
voice gets the red-flag path, not a "try again" message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from . import constants as C
except ImportError:
    import constants as C

GATE_ORDER = ("R1", "R2", "R3", "R4", "R5", "M_T5", "W_窗数")


@dataclass(frozen=True)
class RejectDecision:
    reject: bool
    reject_code: str | None
    reject_text: str | None
    triggered: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "reject": bool(self.reject),
            "reject_code": self.reject_code,
            "reject_text": self.reject_text,
        }


@dataclass(frozen=True)
class GateInputs:
    t_voiced_s: float
    snr_est_db: float
    voiced_ok: bool
    vowel_coverage: float
    miss_rate: float
    key3_missing: bool
    n_blocks: int
    bypass_severe: bool


def evaluate_gates(inputs: GateInputs) -> RejectDecision:
    if inputs.bypass_severe:
        return RejectDecision(False, None, None, ())

    triggered: list[str] = []
    if inputs.t_voiced_s < C.VOICED_MIN_S:
        triggered.append("R1")
    if not (inputs.snr_est_db >= C.SNR_MIN_DB):   # NaN fails closed
        triggered.append("R2")
    if not inputs.voiced_ok:
        triggered.append("R3")
    if not (inputs.vowel_coverage >= C.VOWEL_COVERAGE_MIN):
        triggered.append("R4")
    if inputs.miss_rate > C.MISS_RATE_MAX:
        triggered.append("R5")
    if inputs.key3_missing:
        triggered.append("M_T5")
    if inputs.n_blocks < C.MIN_BLOCKS:
        triggered.append("W_窗数")

    ordered = tuple(g for g in GATE_ORDER if g in triggered)
    if not ordered:
        return RejectDecision(False, None, None, ())
    return RejectDecision(
        reject=True,
        reject_code=ordered[0],
        reject_text=C.REJECT_TEXT[ordered[0]],
        triggered=ordered,
    )
