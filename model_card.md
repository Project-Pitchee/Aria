# Aria v0.2.3.3 冷启动模型包（正式包）：Model Card

- model_ver **`v0.2.3.3`**（refit L3，g 版本 3）；feature_engine **native**（numpy/scipy 复刻，不依赖 Praat）；seg_ver `stage0-v0.2.1.2`。
- 链路：阶段 0 段 A → native 9 变量 → L3 系数 → clip [0,10] → G6 恒等（ŵ = 0）→ g（[weights/g_table.json](weights/g_table.json)）→ S。常数见 [weights/model.json](weights/model.json)，4 位有效数字。
- 取代 v0.2.3.2（草稿，praat-script）与 v0.2.0.1（breathiness 标签），两者的包存档在 `aria-training/artifacts/m4-model-package-v0.2.3.2/` 与 `aria-training/artifacts/m4-model-package/`。上一代发布包 v0.1-pre 见本仓 tag `v0.1-pre` 与 [history/v0.1-pre/](history/v0.1-pre/model_card.md)。规格修订见 `aria-training/docs/spec-revisions.md`。

> 本卡与 `aria-training/artifacts/m4-model-package-v0.2.3.3/MODEL_CARD.md` 同源同版本。为适应本发布仓的目录布局，卡内指向训练/评估产物的链接改写为 `aria-training` 仓内路径的文字引用，并新增或改写本包专有内容（新增「本包内容」一节与「发布包自洽门」一小节，「复现」改为只列本包命令）。模型本身的数字、门结论与判定一字未改；Limitations 的条目、顺序与结论不变，仅在两条上补了本包专有说明（第 8 条：本包同时输出训练口径与产品口径两个 status；第 12 条：锁定集上未达显著）。本包不含训练与评估代码。

## 本包内容

| 路径 | 内容 |
|---|---|
| `weights/model.json` | L3 系数、clip、g 版本、冷启动阈值、dev 指标与包身份 |
| `weights/g_table.json` | 量纲链 g 的常数表（knots / p / σ）与验收记录 |
| `inference/predict.py` | 入口：`--audio`（原始录音）、`--segment-a`（段 A）、`--requests`（批量）|
| `inference/stage0/` | 阶段 0（seg_ver `stage0-v0.2.1.2`，13 个模块，与冻结预测所用代码逐字节相同）|
| `inference/native_features/` | native 九特征引擎（5 个模块，与冻结预测所用代码逐字节相同）|

依赖：numpy、scipy、soundfile、onnxruntime（只有阶段 0 的 VAD 需要）。阶段 0 的 Silero VAD ONNX 不随包分发，需自备并校验 SHA-256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`（`--vad-model` / `$ARIA_VAD_MODEL` / `inference/models/SileroVAD.onnx`）；只走 `--segment-a` 时无需 VAD。

## 适用范围

本包是**测量内核**，不是完整的录音端产品链路。输入契约是**段 A**（16 kHz mono 波形）：从段 A 往后依次计算九项特征、ABI'、S。段 A 之前的环节——从任意录音中定位并切出持续元音——不属于本包。

**已验证范围仅限 PVQD 协议复合录音经阶段 0 产出的段 A**：拟合池 212 人与 hold-out 12 人的全部样本都是这一口径下的段 A——PVQD 每个文件是持续 /a/、/i/ 加 CAPE-V 六句的复合录音（无人工句切），整文件过现行阶段 0 离线产出段 A（全部周期性区域拼接，训练口径）。**任意录音场景不在本包验证范围内**：面向任意录音的元音切割前端尚未建立，也未经验证；本包没有、也无法据此对任意录音的切分正确性、噪声鲁棒性或端到端输出作出任何声明。

## 变更追溯

| # | 变更 | 从 → 到 | 依据 |
|---|---|---|---|
| 1 | 目标标签 | CAPE-V breathiness / 10 → **CAPE-V 总体严重度均值 / 10** | Ryo 2026-10-18 第 1 项 |
| 2 | G3 CV 极差判据 | ≤ 0.10 → ≤ τ（噪声校准，本池 τ = 0.369） | Ryo 批准提案 A（规格修订 R1） |
| 3 | 「不劣于 L0」 | 严格比较 hold-out → 选级以 CV 为准，hold-out 只核 ≥ 0.70 | Ryo 方案 (iii)（规格修订 R1） |
| 4 | T_diag | 3.44 原文 → **未改，占位，待裁定** | Ryo 第 3 项挂起 |
| 5 | 特征引擎 | praat-script → **native** | 原生线审计 + 本包回归 |

## 开发指标（dev，非锁定测试；训练口径，R1/W 不排除）

| 项 | 主标签：CAPE-V 严重度 | 对照：CAPE-V breathiness（v0.2.0.1） |
|---|---|---|
| 拟合池 / hold-out | 212（normal 84 / mild 79 / moderate 31 / severe 18）/ 12 人 | 同 |
| 选中级 | **L3**（λ = 100，向已发表系数收缩） | L0 |
| CV rs 均值 L0 → L3 | 0.588 → 0.642（Δ +0.054） | 0.635 → 0.636（Δ +0.001） |
| CV 各折 rs（选中级） | 0.490 / 0.677 / 0.676 / 0.697 / 0.668 | 0.519 / 0.621 / 0.631 / 0.540 / 0.866 |
| CV rs 极差（选中级） | 0.207 | 0.348 |
| hold-out rs（L0 对照） | **0.888**（0.895） | 0.860（0.860） |
| hold-out MAE（ABI' 空间） | 1.255 | 2.146 |
| 同一模型对其他标签的 rs（拟合池 / hold-out） | breathiness 0.647 / 0.846；GRBAS-B 0.653 / 0.875 | GRBAS-B 0.664 / 0.883 |

L3 系数：b0 = 12.47；cpps −0.5062，jitter −0.3403，GNE −5.915，Hfno −1.061，HNR-D 0.06068，H1-H2 −0.004161，ShdB 4.759，Shim −0.1942，PSD 10.64。

## 锁定 Benchmark 结果（§2.5，2026-10-18 一次性执行）

比较对象：v1 冻结 CAPE-V severity 头（原生 mm，未重算，只按冻结文件哈希复用）。两侧同在 n = 57 的配对集上计算，零映射、零拟合。预测文件冻结 SHA-256 `3ff22e6a…44d69`；完整方法、终点与合规披露见 `aria-training/docs/v0233-locked-benchmark-result.md`（预测冻结记录：`aria-training/docs/v0233-locked-benchmark-prediction-freeze.md`）。

**判定：未达显著。** 主检验单侧 Wilcoxon signed-rank（方向 d > 0）W = 725，**p = 0.790005**（α = 0.05）；主终点 median d = **+0.693 mm**，95% CI **[−2.408, +3.012]**（跨 0）；配对符号检验 30/57，单侧 p = 0.395683。**因此不得写「v0.2.3.3 在锁定集上显著优于 v1 severity 头」。**

| 侧 | Spearman rs [95% CI] | MAE mm [95% CI] | 中位 \|误差\| mm |
|---|---|---|---|
| v1 CAPE-V severity 头 | 0.8130 [0.6983, 0.8826] | 10.951 [8.645, 13.508] | 8.593 |
| Aria v0.2.3.3（ABI′ × 10） | 0.7164 [0.5338, 0.8429] | 11.971 [9.553, 14.482] | 9.247 |

点估计上 v1 略优（mean d = −1.020 mm；v0.2.3.3 更准 30 人 / v1 更准 27 人 / ties 0），但两个独立 ρ 之差未作显著性检验，两列数字**只作描述**，不构成任何优势或劣势的显著性宣称。按作者 Category 分层同样只是描述：**Normal/Mild 两层两侧相当**（Normal n = 23，MAE 8.649 vs 8.366；Mild n = 17，MAE 7.390 vs 8.322），**Moderate/Severe 两层 v0.2.3.3 描述性偏弱**（Moderate n = 13，22.019 vs 18.532；Severe n = 4，17.883 vs 12.348）；分层子样本很小（Severe 仅 4 人），不作分层结论。

intent-to-screen：原始分配 60 → 成功评分 58 → VAD 拒检 2（PT084、PT134）。v0.2.3.3 侧 **PT097 在正式包（训练口径）下 status = `rejected:R2|R5`**（SNR 9.93 dB < 10、miss_rate > 0.5），不产出预测；配对集因此为 n = 57，不插补、不替换人选。产品口径（七门全过）下同一批 58 人中 22 人可评分、36 人 `rejected:R1`，该表仅描述性、不参与任何终点。

**本结果不改变 v0.2.3.3 的适用范围表述，也不决定是否发布**（v0.2.3.3 按计划上线发布）。

### 发布包自洽门（2026-10-18）

用**本发布包自身的入口**（`inference/predict.py` 批模式）复算同一批 58 人，与冻结预测文件 `3ff22e6a…44d69` 逐人比对：`predictions.csv` **逐字节一致**（同为 SHA-256 `3ff22e6a…44d69`），逐人 ABI′ 最大 |Δ| = **0.0**（容差 1e-6）；status 58/58 一致（scored 57 / `rejected:R2|R5` 1）。同时逐模块校验本包 `inference/` 的 13 + 5 个模块哈希与冻结运行的 `generation_manifest.json` 记录逐字相同，权重哈希与 §2.5 报告页眉所载包身份逐字相同。报告：`aria-training/artifacts/release-selfcheck-v0.2.3.3/selfcheck_report.md`。

## G3 重拟合门：修订后 PASS，原门 FAIL 并排披露

| 判据 | 原门（v0.2.1） | 修订门 R1 | 本包数值 |
|---|---|---|---|
| hold-out rs ≥ 0.70 | 通过 | 通过 | 0.888 |
| 不劣于原系数 | 严格 ≥ L0：**不过**（−0.007） | 不作门，仅披露 | 差 −0.007，bootstrap 95% CI [−0.26, +0.22]，记为噪声内 |
| CV 极差 | ≤ 0.10：**不过** | ≤ τ = 0.369：**通过** | 0.207 |
| **结论** | **FAIL** | **PASS** | |

- τ 的定义：零模型下 5 折 rs 极差的第 95 百分位，零模型取 ρ₀ = 0.70、每折 42.4 人，按 Fisher z 抽样（Bonett–Wright SE），蒙特卡洛 10⁶ 次。观测极差 0.207 落在重新分折噪声分布的第 50 百分位。
- 原结论留档：v0.2.0.1（breathiness 标签）CV 极差 0.348，原门 FAIL。这个结论没有被重判，也没有被覆盖。
- Step 0 对等审计：native 对 praat-script 九项特征 r ≥ 0.95（|r − 1| < 1e-12）。原生线报告 §5.1 记录段 A 对原始录音的特征偏移（H1-H2 0.65、PSD 0.82），属阶段 0 层面的问题。**比较对象口径已裁定**（Ryo 2026-10-18 六项裁决第 4 项）：Step 0 只判引擎对等，已过线；段 A 对原始录音的偏移记为已知限制，归阶段 0 范畴，不在本次引擎对等范围内，不回炉。

## G4 量纲链：通过

参考样本：拟合池中作者类别 Normal 的 84 人，用 L3 打分（1 个值裁剪到 0，无并列）。中位数 1.592，σ̃ = 0.884。三段 C0 连续（接点差 1.9e-9），解析严格单调；纯查表实现与参考实现最大差 1.4e-14。

## 冷启动阈值：未标定、待裁定

> **S_diag 与冷启动红线未标定，不得作为已标定结论引用。**

T_diag 按规格原文 3.44 占位（L0 ABI 诊断阈值）。在严重度标签与 L3 下，ABI' 的含义是「预测 CAPE-V 严重度 / 10」，3.44 对应 34.4 mm，这个阈值在新口径下没有定义。由此算出的数值只是占位：S_diag = g(4.128) = 99.87，T_red^cold = min(90, 99.87) = 90，T_yellow^cold = 75，防倒挂锁不触发。待 Ryo 裁定 T_diag^(3)。本包因此只输出 ABI′、y_hat_mm 与 S 三项测量值，不输出产品档位判定。

## 引擎替换回归（praat-script → native）

对照包：同一标签、同一 R1 规则、同一 g 版本，用 praat-script 特征构建（`aria-training/artifacts/m5-engine-regression/control-praat-script/`，仅作回归对照，不发布）。容差 1e-6（审计误差量级，远低于 4 位有效数字的发布精度）。

| 检查 | 结果 |
|---|---|
| 选中级 / G3 判定 | L3 / PASS，两引擎相同 |
| 发布系数（4 位有效数字） | 完全相同 |
| g 常数表、冷启动阈值 | 完全相同 |
| refit 全部数值（CV、hold-out、λ 曲线等） | 最大差 5.6e-8 |
| dev 指标 | 最大差 2.4e-11 |
| 逐端点 ABI'（2,352 端点，2,167 两边可评分） | 最大 \|Δ\| 5.1e-9 |
| 逐端点 S | 最大 \|Δ\| 5.3e-8，中位数 0 |
| 拟合池 234 段（233 可评分） | ABI' 最大 \|Δ\| 8.9e-10，S 最大 \|Δ\| 6.4e-8 |
| 只一边可评分 | 0 |

单元测试：126/126 通过（`aria-training/artifacts/m5-engine-regression/unit-tests.txt`），包括原生线 18 项，以及 abi_praat 空段 A 修复的 2 项回归测试。

## Limitations 与偏差清单

1. **适用范围限于 PVQD 协议复合录音经阶段 0 产出的段 A**（见上「适用范围」）。本包验证过的输入只有这一口径下的段 A；面向任意录音的元音切割前端未建、未验证，任意录音场景不在本包验证范围内。本项不改变模型数值、门结论与发布状态。
2. **T_diag 未标定**（见上）；冷启动红线不能用于产品结论。
3. G3 按修订门 R1 通过；按原门 0.10 不通过。两者并排披露。
4. hold-out 只有 12 人，rs 的 CI 很宽；L3 与 L0 在 hold-out 上的差在噪声内。
5. 段 A 与原始录音之间的特征偏移（12 人上 ABI r = 0.954）。Step 0 的比较对象已裁定（2026-10-18 第 4 项）：只判引擎对等，已过线；本项偏移记为已知限制，属阶段 0 范畴，不回炉。
6. native 引擎复刻的是 arm64 上 parselmouth 0.4.7（Praat 6.1.38，FMA 舍入）的数值；对 x86 Praat 基准未验证，预期 CPPS 有约 1e-3 dB 量级的平台差。
7. 段 A 是 16 kHz，经 sinc 重采样到 44.1 kHz；Hfno-6000 的 6–10 kHz 分母只有重采样残余。
8. 训练口径：R1/W_窗数 不排除（裁决），产品口径不变。本包同时输出两个口径的 status（`status` = 训练口径，`status_product_caliber` = 七门全过），两个口径下可评分人数差别很大（§2.5 锁定 58 人：57 vs 22）。
9. G6 修正器未激活（ŵ = 0）。
10. 阶段 0 已知偏差：WADA 模拟重建表、q_codec 停用、距离档占位、T60/元音样检测器为占位实现。
11. 个人化参数（σ_floor、σ_CVA、w_base）未出。
12. **§2.5 已运行**（2026-10-18 一次性锁定测试，结果见上「锁定 Benchmark 结果」）；60 名锁定说话人未进入任何拟合、调参或参考集。锁定集上未达显著：v0.2.3.3 与 v1 冻结 severity 头的精度差异不显著。
13. 非医疗产品：输出为嗓音状态参考，不是诊断。

## 复现

只用本包（无训练依赖）：

```bash
python -m pip install numpy scipy soundfile onnxruntime

# 已切好的段 A（本包验证过的输入口径）
python inference/predict.py --segment-a segment_a.wav --out out.json

# 原始录音（阶段 0 + 段 A + 特征 + ABI' + S；需自备 VAD ONNX）
python inference/predict.py --audio rec.wav --vad-model /path/SileroVAD.onnx --out out.json

# 批量：requests.jsonl 每行 {"audio_path","audio_sha256","distance_cm","speaker_id"}
python inference/predict.py --requests requests.jsonl --dataset /path/PVQD \
    --vad-model /path/SileroVAD.onnx --outdir out/
```

训练、重拟合与锁定评估的复现命令在 `aria-training` 仓（`docs/m1/README.md`、`docs/v0233-locked-benchmark-result.md` §9）；本包不含训练代码与任何测试标签。

SHA-256：`weights/model.json` `4415d3fd…56c0`，`weights/g_table.json` `4cc7d521…a385`（见 `SHA256SUMS`）。
