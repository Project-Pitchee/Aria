---
license: apache-2.0
model_type: aria_linear_head
# §2.5 locked benchmark has run; the result (not significant vs. the v1 frozen
# CAPE-V severity head) is in the "锁定 Benchmark 结果" section of model_card.md.
---

# Aria v0.2.3.3

本目录是可直接运行的本地 Hugging Face 模型包：**冷启动测量内核**。

> ⚠️ **§2.5 锁定 Benchmark 已运行（2026-10-18，一次性）。** 结论：**未达显著**——单侧 Wilcoxon p = 0.790005；主终点 median d = +0.693 mm，95% CI [−2.408, +3.012]；v0.2.3.3 rs 0.7164 / MAE 11.971 mm，v1 同集 rs 0.8130 / MAE 10.951 mm（n = 57）。**不得写「v0.2.3.3 显著优于 v1 severity 头」。** 完整方法、分层、拒评披露与合规声明见 [model_card.md](model_card.md)。

Dev（训练口径，非锁定测试）：hold-out rs **0.888**（12 人），CV rs 均值 **0.642**，G3 修订门 R1 PASS。完整方法、局限和已知部署问题见 [model_card.md](model_card.md)。

## Quickstart

依赖：Python 3.10+、numpy、scipy、soundfile（走原始录音再加 onnxruntime）。

```bash
python -m pip install numpy scipy soundfile onnxruntime
```

**已切好的段 A（本包验证过的输入口径）**：

```bash
python inference/predict.py --segment-a segment_a_16k.wav --out out.json
```

**原始录音**（阶段 0 → 段 A → 九特征 → ABI′ → S；需自备 Silero VAD ONNX，SHA-256 `1a153a22…78e3`）：

```bash
python inference/predict.py --audio rec.wav --vad-model /path/SileroVAD.onnx --out out.json
```

**批量**：`requests.jsonl` 每行 `{"audio_path", "audio_sha256", "distance_cm", "speaker_id"}`

```bash
python inference/predict.py --requests requests.jsonl --dataset /path/PVQD \
    --vad-model /path/SileroVAD.onnx --outdir out/
```

输出：`abi_prime`（ABI′，0–10）、`y_hat_mm`（= 10 × ABI′，PVQD CAPE-V 总体严重度 mm）、`s`（量纲链 g，0–100）、九个特征、阶段 0 的 21 字段质量报告与诊断，以及两个口径的 status（`status` = 本包训练口径，`status_product_caliber` = 七门全过）。`predict` 不读取任何标签。

| 路径 | 内容 |
|---|---|
| `weights/model.json` | L3 系数、clip、g 版本、冷启动阈值、dev 指标、包身份 |
| `weights/g_table.json` | 量纲链 g 常数表与验收记录 |
| `inference/predict.py` | 入口（`--audio` / `--segment-a` / `--requests`） |
| `inference/stage0/` | 阶段 0，seg_ver `stage0-v0.2.1.2`（13 模块） |
| `inference/native_features/` | native 九特征引擎（5 模块） |
| `history/v0.1-pre/` | 上一代发布包 v0.1-pre 的卡、权重与入口（归档，含其 §2.5 结论） |

`inference/` 的 18 个模块与冻结预测所用的代码逐字节相同（哈希见 `SHA256SUMS` 与 model_card.md 的「发布包自洽门」）。训练、重拟合与锁定评估代码在独立的 `aria-training` 仓；本包不含训练代码与任何测试标签。

## 适用范围

**已验证范围仅限 PVQD 协议复合录音经阶段 0 产出的段 A。** 任意录音场景不在本包验证范围内——面向任意录音的元音切割前端尚未建立。详见 [model_card.md](model_card.md) 的「适用范围」。
