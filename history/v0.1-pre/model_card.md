# Aria v0.1-pre

> **归档说明（2026-10-18）。** 本文件是 v0.1-pre 发布包的卡，随本仓升级到 v0.2.3.3 移入 `history/v0.1-pre/`（原路径 `model_card.md`；同批移入 `README.md`、`KNOWN_ISSUES.md`、`config.json`、`weights/*.json`、`inference.py`）。除下方 pending 一处按 D1 结果补写外，正文未改。原包仍可从本仓 tag `v0.1-pre` 取得。

> ⚠️ **Test evaluation 已运行（D1，2026-10-18 一次性锁定测试）：未达显著。** 单侧 Wilcoxon signed-rank p = **0.581281**（α = 0.05，W = 829.0），主终点 median d = **−0.063 mm**（95% CI [−1.221, +1.728]，跨 0），配对符号检验 29/58（单侧 p = 0.552158）。**v0.1-pre 在锁定集上未显著优于 v1 冻结 CAPE-V severity 头，不得写「显著优于」。** 描述性指标：v0.1-pre rs = 0.76864 / MAE = 11.465 mm；v1 rs = 0.82127 / MAE = 10.961 mm（n = 58）。本包仍只包含本地预发布权重，不代表测试集结果，也未上传测试标签；完整报告见 VocalHealth-v2 的 `reports/d1-locked-test-result.md`。

Aria 是 VocalHealth v2 的 Tier 1 Ridge 线性头。它把原始 398 列按 `[14 acoustic | 192 embedding mean | 192 embedding std]` 解释，使用 F2 的 14 个声学特征和 192 维 embedding mean，再追加 14 个声学可用性指示变量，得到 220 维输入。权重已折叠为一个线性层；推理输出先是 PVQD CAPE-V 的标准化严重度，再用保存的 train-only 参数逆变换到 mm。

## Dev 结果

- Spearman: **0.7422**
- MAE: **10.98 mm**
- n=36，speaker-grouped split，seed=7
- 训练样本：236 条 clean-only 行（200 train / 36 dev）
- 来源 bundle SHA-256：`4689ea1ffa5130568100baf352e9b64a5eb4ddf1e55f19c46a49a959b09af2de`

## 方法与预处理

声学缺失采用 `core_v1`：`HNR=-200` 标为不可用；`F0 median=0` 同时标记 F0 median/IQR/CV 不可用。不可用值使用 train-only、task-trusted 的逐列中位数填补；指示变量在填补前生成并保持未标准化。严重度 harmonization 使用 PVQD train-only CAPE-V mean/sd，推理时逆变换回 mm。

## 局限

- 增强特征定义曾混用；本 v0.1-pre 只使用 clean-only 重拟合结果，增强样本已排除。
- missingness 与域标签共线，线性头可能学习到采集域/缺失模式而非纯粹的声学严重度。
- narrow 8 kHz 身份削弱约 **22%**；phone 域鲁棒性需要全量重训后再判断。
- 端侧当前存在 sentinel（离线 `-200`、端侧 `0.0`）和维度（离线 220、端侧 206）不一致；详见 `KNOWN_ISSUES.md`。
