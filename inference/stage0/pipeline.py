"""Stage 0 end-to-end: waveform (+ metadata) -> segment A + quality report.

Order follows SSOT §3.1 exactly: segment, estimate the bypass, score quality,
resolve missingness, evaluate gates, build the 21-field report. Nothing here
writes files; callers decide persistence and reruns stay deterministic because
every step is a pure function of the input PCM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from . import constants as C
    from .audio import LoadedAudio
    from .bypass import bypass_evidence
    from .gates import GateInputs, evaluate_gates
    from .missing import (AvailabilityProvider, Stage0Availability, key3_missing,
                          miss_code, miss_rate)
    from .quality import quality_vectors, vowel_like_mask
    from .report import Stage0Report, build_report
    from .segment import SegmentationResult, segment
    from .vad import SileroVad, VadSegment
except ImportError:
    import constants as C
    from audio import LoadedAudio
    from bypass import bypass_evidence
    from gates import GateInputs, evaluate_gates
    from missing import (AvailabilityProvider, Stage0Availability, key3_missing,
                         miss_code, miss_rate)
    from quality import quality_vectors, vowel_like_mask
    from report import Stage0Report, build_report
    from segment import SegmentationResult, segment
    from vad import SileroVad, VadSegment


@dataclass
class Stage0Result:
    report: dict
    segment_a: np.ndarray
    segmentation: SegmentationResult
    diagnostics: dict = field(default_factory=dict)


def run_stage0(audio: LoadedAudio,
               vad_segments: list[VadSegment] | None = None,
               *,
               vad: SileroVad | None = None,
               availability: AvailabilityProvider | None = None,
               device_id: str | None = None,
               distance_bin: str | None = None,
               dnsmos: dict | None = None) -> Stage0Result:
    """Run stage 0 on one recording.

    ``vad_segments`` may be supplied directly (tests, cached runs); otherwise a
    ``vad`` detector is required.
    """
    if vad_segments is None:
        if vad is None:
            raise ValueError("run_stage0 needs either vad_segments or a vad detector")
        vad_segments = vad.segments(audio.pcm)

    seg = segment(audio.pcm, vad_segments)
    evidence = bypass_evidence(seg.x_trim, seg.t_voiced_s)

    # Median RMS over periodic frames, in the same frame grid as the bypass.
    frame = int(round(C.BYPASS_FRAME_S * C.TARGET_SR))
    voiced_frames = _voiced_frame_rms(seg, frame)
    signal_rms = float(np.median(voiced_frames)) if voiced_frames.size else float("nan")

    provider = availability or Stage0Availability()
    availability_mask = np.asarray(
        provider.availability(seg.segment_a, seg.voiced_ratio, seg.t_voiced_s),
        dtype=bool)
    rate = miss_rate(availability_mask)
    code = miss_code(availability_mask)
    key3_missing_flag = key3_missing(availability_mask)

    vowels = vowel_like_mask(seg.segment_a, _voiced_mask_on_a(seg))
    vowel_cov = (float(vowels.sum()) / len(vowels)) if len(vowels) else 0.0

    quality = quality_vectors(
        x_pcm=seg.x_pcm, x_trim=seg.x_trim, segment_a=seg.segment_a,
        voiced_mask=seg.voiced_mask, voiced_ratio=seg.voiced_ratio,
        e_noise=evidence.e_noise, n_missing=int((~availability_mask).sum()),
        signal_rms=signal_rms)

    voiced_ok = bool(seg.t_voiced_s >= C.VOICED_MIN_S
                     and seg.voiced_ratio >= C.VOICED_RATIO_MIN)
    decision = evaluate_gates(GateInputs(
        t_voiced_s=seg.t_voiced_s,
        snr_est_db=quality.snr_est_db,
        voiced_ok=voiced_ok,
        vowel_coverage=vowel_cov,
        miss_rate=rate,
        key3_missing=key3_missing_flag,
        n_blocks=seg.n_blocks,
        bypass_severe=evidence.trigger,
    ))

    report = build_report(
        device_id=device_id,
        distance_bin=distance_bin,
        input_spec=audio.input_spec(),
        t_raw_s=seg.t_raw_s,
        t_speech_s=seg.t_speech_s,
        t_voiced_s=seg.t_voiced_s,
        voiced_ratio=seg.voiced_ratio,
        voiced_ok=voiced_ok,
        bypass_severe=evidence.trigger,
        vowel_coverage=vowel_cov,
        q_raw=quality.q_raw,
        q_dmg=quality.q_dmg,
        d=quality.d,
        snr_db=quality.snr_est_db,
        dnsmos=dnsmos,
        miss_rate=rate,
        miss_code=code,
        key3_missing=key3_missing_flag,
        decision=decision,
    )

    diagnostics = {
        "triggered_gates": list(decision.triggered),
        "subsegments": len(seg.subsegments),
        "dropped_short": seg.dropped_short,
        "n_blocks": seg.n_blocks,
        "e_noise": evidence.e_noise,
        "t_active_s": evidence.t_active_s,
        "snr_active_db": evidence.snr_active_db,
        "availability_provider": getattr(provider, "name", type(provider).__name__),
        "availability_mask": availability_mask.tolist(),
        "snr_components": quality.diagnostics,
        "periodicity_peak_median": (float(np.median(seg.periodicity.peak))
                                    if seg.periodicity.peak.size else None),
        "periodicity_frames": int(seg.periodicity.peak.size),
    }
    return Stage0Result(report=report, segment_a=seg.segment_a,
                        segmentation=seg, diagnostics=diagnostics)


def _voiced_frame_rms(seg: SegmentationResult, frame: int) -> np.ndarray:
    """RMS of the bypass-grid frames whose centre lies in a periodic region."""
    n_frames = len(seg.x_trim) // frame
    if n_frames == 0:
        return np.zeros(0)
    usable = seg.x_trim[:n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(usable ** 2, axis=1))
    centres = (np.arange(n_frames) * frame + frame // 2).astype(int)
    return rms[seg.voiced_mask[centres]]


def _voiced_mask_on_a(seg: SegmentationResult) -> np.ndarray:
    """Rebuild the per-sample periodic mask for the concatenated A.

    A is built from contiguous periodic runs, so its time base is simply all
    True by construction; keeping the function explicit documents that the
    vowel detector and the periodicity test share a time base only via this
    construction, never by index coincidence.
    """
    return np.ones(len(seg.segment_a), dtype=bool)
