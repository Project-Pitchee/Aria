---
license: other
model_type: aria_linear_head
# v0.1-pre is archived under history/v0.1-pre/; its §2.5 test (D1) has since run
# and landed not significant. See model_card.md in this directory.
---

# Aria v0.1-pre

> **归档说明（2026-10-18）。** 本 README 随本仓升级到 v0.2.3.3 移入 `history/v0.1-pre/`，其 `inference.py` 与 `weights/` 也一并移入本目录。要复跑下面的 Quickstart：把 `inference.py` 与 `weights/` 拷回包根（即把它们放到 `inference/inference.py` 与 `weights/`），或直接调用 `inference.load_artifacts("<本仓根>/history/v0.1-pre")` 并传入显式 root。

本目录是可直接运行的本地 Hugging Face 模型包草案。

> ⚠️ **Test evaluation 已运行（D1，2026-10-18）：未达显著**（单侧 Wilcoxon p = 0.581281；median d = −0.063 mm，95% CI [−1.221, +1.728]；n = 58）。dev 结果之外不得据此声称优于 v1 severity 头。详见 [model_card.md](model_card.md)。

Dev：Spearman **0.7422**，MAE **10.98 mm**（n=36，speaker-grouped，seed=7）。完整方法、局限和已知部署问题见 [model_card.md](model_card.md)。
## Quickstart

依赖只有 Python 3.10+ 和 NumPy：

```bash
python -m pip install numpy
python - <<'PY'
import numpy as np
from inference.inference import predict

# X 必须是 (n, 398)：[14 acoustic | 192 embedding mean | 192 embedding std]
X = np.load("your_canonical_features.npy")
out = predict(X)
print(out["severity_mm"])
PY
```

传入原始声学可用性时使用 `predict(X, acoustic_available=mask)`，其中 `mask` 为 `(n,14)` 的 0/1 数组。`predict` 不读取任何标签。

权重、scaler、harmonization 和填补参数在 `weights/`；产品推理实现只依赖 NumPy。训练与评估代码在独立的 `aria-training` repo。
