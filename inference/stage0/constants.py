"""Aria v0.2.1 stage 0 constant table.

Every number stage 0 uses lives here so the report's ``seg_ver`` can pin one
constant table. Each entry is tagged with where it came from:

``SPEC``
    Written verbatim in the frozen SSOT (docs/m1/aria-v021-complete-scheme.md).
``PLACEHOLDER``
    An engineering placeholder the SSOT explicitly defers or leaves unstated.
    These are NOT calibrated conclusions and must not be quoted as such.
``ENGINEERING``
    Needed to make a spec formula computable (frame length, hop, lag range).
    Not a threshold the product decides on; recorded so results reproduce.

Do not edit a number here without recording it as a deviation in the
milestone report.
"""

from __future__ import annotations

from dataclasses import dataclass

SEG_VER = "stage0-v0.2.1.2"   # .1 Silero context fix; .2 WADA G statistic

# --- Audio ---------------------------------------------------------------
TARGET_SR = 16_000          # SPEC §2: A is 16 kHz mono.

# --- Two-level boundary trim (SPEC §3.1.1) ------------------------------
EDGE_TRIM_S = 0.250               # SPEC: whole-recording head/tail trim.
SUBBOUNDARY_TRIM_S = 0.025        # SPEC: per VAD sub-segment head/tail trim.
                                  # PLACEHOLDER per handoff §2.3 list.

# --- Silero VAD (SPEC §3.1.1, Q11-D1) -----------------------------------
VAD_THRESHOLD = 0.5               # SPEC: speech_prob >= 0.5.
VAD_MIN_SILENCE_S = 0.100         # SPEC: pauses shorter than 100 ms are merged.
VAD_CHUNK_SAMPLES = 512           # ENGINEERING: Silero v5 16 kHz frame.
VAD_CONTEXT_SAMPLES = 64          # ENGINEERING: Silero v5 ONNX left context
                                  # prepended to every chunk (upstream OnnxWrapper).
VAD_MODEL_SHA256 = (
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
)  # ENGINEERING: sha256 of SileroVAD.onnx used for every number in the M1 report.

# --- Periodic voiced confirmation (SPEC §3.1.1 Step 4) ------------------
PERIODICITY_PEAK_MIN = 0.45       # SPEC: normalized autocorrelation peak > 0.45.
PERIOD_FRAME_S = 0.040            # ENGINEERING: one period of a 25 Hz ... (40 ms).
PERIOD_HOP_S = 0.010              # ENGINEERING: 10 ms hop mirrors frame analysis.
F0_MIN_HZ = 60.0                  # ENGINEERING: lag search lower bound.
F0_MAX_HZ = 400.0                 # ENGINEERING: lag search upper bound.
PERIOD_RMS_FLOOR_RATIO = 0.01     # ENGINEERING: frames under 1% of the loudest
                                  # frame RMS are treated as silence (ac on
                                  # pure noise is meaningless).

# --- voiced_ok / gates (SPEC §3.1.1, §3.1.4) ----------------------------
VOICED_MIN_S = 10.0               # SPEC: T_voiced >= 10 s (Q06-D6 option B).
VOICED_RATIO_MIN = 0.30           # SPEC: voiced_ratio >= 0.30.
SNR_MIN_DB = 10.0                 # SPEC: R2 threshold.
VOWEL_COVERAGE_MIN = 0.30         # SPEC: R4 threshold.
MISS_RATE_MAX = 0.50              # SPEC: R5 threshold.
MIN_BLOCKS = 20                   # SPEC: W_窗数, 500 ms analysis blocks.
CVA_BLOCK_S = 0.500               # SPEC §3.4.1: 500 ms blocks.

# --- Severe bypass (SPEC §3.1.2) ----------------------------------------
BYPASS_FRAME_S = 0.050            # SPEC: 50 ms non-overlapping frames.
BYPASS_NOISE_PERCENTILE = 10.0    # SPEC: E_noise = P10 of frame RMS.
BYPASS_ACTIVE_OVER_NOISE = 2.0    # SPEC: +6 dB above noise floor.
BYPASS_BAND_HZ = (300.0, 4000.0)  # SPEC: voice band.
BYPASS_BAND_RATIO_MIN = 0.60      # SPEC: energy fraction in voice band > 0.60.
BYPASS_MIN_CONSECUTIVE = 3        # SPEC: >= 3 consecutive frames.
BYPASS_ACTIVE_MIN_S = 10.0        # SPEC: T_active >= 10 s.
BYPASS_VOICED_MAX_S = 2.0         # SPEC: T_voiced < 2 s.
BYPASS_SNR_MIN_DB = 6.0           # SPEC: SNR_active > 6 dB.
RMS_EPSILON = 1e-12               # ENGINEERING: guards log / divide by zero.

# --- q_raw / q_dmg (SPEC §3.1.3) ----------------------------------------
CLIP_HARD_LIMIT = 0.99            # SPEC: |x| >= 0.99 counts as clipped.
CLIP_FLAT_LIMIT = 0.50            # SPEC: |x| >= 0.50 and flat counts as clipped.
CLIP_FLAT_DELTA = 1e-5            # SPEC: consecutive-sample flatness tolerance.
CLIP_FLAT_LOOKBACK = 2            # SPEC: max_{k=1,2} flatness against t-k.
SPECTRAL_WELCH_SEGMENT = 1024     # ENGINEERING: 64 ms at 16 kHz.
LF_RUMBLE_HZ = 120.0              # SPEC: f < 120 Hz.
TILT_LOW_HZ = (200.0, 1000.0)     # SPEC: tilt denominator band.
TILT_HIGH_HZ = (2000.0, 4000.0)   # SPEC: tilt numerator band.
Q_CLIP_SCALE = 0.05               # SPEC: q_clip = clip(clipping_ratio / 0.05).
Q_REVERB_T60_ONSET_S = 0.40       # SPEC: q_reverb knee.
Q_REVERB_T60_SCALE_S = 0.40       # SPEC: q_reverb scale.
Q_SNR_CLEAN_DB = 20.0             # SPEC: q_snr = clamp((20 - snr) / 10).
Q_SNR_SCALE_DB = 10.0             # SPEC: q_snr scale.

# --- Unstated stage-0 estimators (deviations, see report) ---------------
# T60_est: SSOT writes q_reverb but never defines how T60_est is measured.
T60_WINDOW_S = 0.300              # PLACEHOLDER: Schroeder decay fit window.
T60_MIN_R2 = 0.90                 # PLACEHOLDER: fall back to 0 when fit is poor.
# codec_artifact_score: SSOT writes q_codec but never defines the score.
CODEC_HIGH_BAND_HZ = (6_500.0, 7_800.0)   # PLACEHOLDER
CODEC_MID_BAND_HZ = (300.0, 3_400.0)      # PLACEHOLDER
CODEC_RATIO_REF = 0.02            # PLACEHOLDER: high/mid ratio treated as clean.
# Vowel-like region detection: SSOT gives no detector, only the ratio.
VOWEL_PEAK_PROMINENCE_DB = 2.0    # PLACEHOLDER: intensity-peak prominence.
VOWEL_PEAK_MIN_GAP_S = 0.150      # PLACEHOLDER: syllable-nucleus minimum spacing.
VOWEL_REGION_S = 0.250            # PLACEHOLDER: peak expanded to fixed window.
VOWEL_F0_RANGE_HZ = (60.0, 400.0) # PLACEHOLDER: vowel-like requires periodicity.
VOWEL_F1_RANGE_HZ = (250.0, 950.0)  # PLACEHOLDER: one clear low formant peak.
# WADA-SNR (Kim & Stern 2008): the paper's G statistic, table rebuilt by
# simulation from the paper's Gamma(0.4) speech model.
WADA_GAMMA_SHAPE = 0.4            # SPEC-cited paper constant.
WADA_SIM_SAMPLES = 2_000_000      # ENGINEERING: Monte-Carlo samples per table.
WADA_GRID_DB = (-20.0, 100.0)     # ENGINEERING: paper table range.
WADA_AMPLITUDE_FLOOR = 1e-10      # ENGINEERING: log guard for exact zeros.
SNR_REPORT_MIN_DB = -20.0          # ENGINEERING: WADA table lower end.
SNR_REPORT_MAX_DB = 100.0          # ENGINEERING: WADA table upper end.
DIGITAL_SILENCE_RMS = 1e-6         # ENGINEERING: floor for E_noise so -120 dBFS
                                  # digital silence cannot make every frame active.
WADA_SIM_SEED = 20261017          # ENGINEERING: reproducibility.

# --- Missingness (14 acoustic features, M1 hook only) -------------------
N_ACOUSTIC_FEATURES = 14
KEY3_FEATURES = ("cpps_db", "hnr_db", "f0_family")   # SPEC: CPPs / HNR / F0-or-GNE.
MISS_GROUP_F0 = ("jitter_local", "jitter_rap", "jitter_ppq5",
                 "f0_median", "f0_iqr", "f0_cv", "voiced_fraction")
MISS_GROUP_ENERGY = ("hnr_db", "intensity_median", "intensity_iqr")
MISS_GROUP_SPECTRAL = ("cpps_db", "shimmer_local", "shimmer_apq3", "shimmer_apq5")

# --- Reject copy (not in SSOT; drafted here, flagged in the report) -----
REJECT_TEXT = {
    "R1": "这段录音里能确认的持续发声不够 10 秒，状态参考分算不准。"
          "请在安静的地方自然地说满一分钟以上再录一次。",
    "R2": "录音环境偏吵，声音被背景噪声盖住了，测出来的状态参考会失真。"
          "请换个安静一些的地方，靠近一点再录一次。",
    "R3": "这段录音里能稳定发声的部分偏少，可能说得太短或中途停顿较多。"
          "请连续自然地说一分钟左右再录一次。",
    "R4": "录音里元音样的连续发声偏少，这次的内容分布不利于稳定比较。"
          "请用平时说话的方式连续说一段，不用刻意念字。",
    "R5": "这段录音里多项声学特征无法可靠计算，状态参考分不够可信。"
          "请重录一次，尽量远离噪声源。",
    "M_T5": "这段录音里核心的声学指标没能稳定提取，无法给出状态参考。"
            "请重录一次；如果多次出现，可以换一台设备试试。",
    "W_窗数": "录音里可分析的片段太少，不足以给出稳定的状态参考。"
              "请自然地说满一分钟以上再录一次。",
}


def constant_table() -> dict:
    """Name -> value for everything upper-case in this module (JSON-safe)."""
    import json
    mod = globals()
    out = {}
    for name in sorted(mod):
        if not name.isupper():
            continue
        value = mod[name]
        try:
            json.dumps(value)
        except TypeError:
            continue
        out[name] = value
    return out
