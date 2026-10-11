# Aria v0.1-pre

本目录是可直接运行的本地 Hugging Face 模型包草案。它只含推理代码、JSON 权重和配置；不含训练代码、测试标签、legacy bundle 或 `.npz` 数据。**Test evaluation pending.**

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
