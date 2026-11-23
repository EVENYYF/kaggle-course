# Kaggle 机器学习课程项目环境

面向 Apple Silicon Mac 的 Python 3.12 项目环境，覆盖表格、文本和遥感图像分类的常用依赖，并支持 PyTorch MPS 加速。

## 项目结构

- `data/`：原始与处理后的数据（不提交到 Git）
- `configs/`：实验配置
- `src/`：训练、评估及工具代码
- `notebooks/`：探索性分析笔记本
- `runs/`：日志、检查点及实验输出（不提交到 Git）
- `report/`：课程报告材料

## 环境创建与复现

在项目根目录执行：

1. 安装 uv：`brew install uv`
2. 安装 Python 并创建虚拟环境：`uv python install 3.12 && uv venv --python 3.12 .venv`
3. 根据锁定环境恢复依赖：`uv pip install --python .venv/bin/python -r requirements.txt`
4. 验证环境：`.venv/bin/python src/check_env.py`

在终端中激活环境：`source .venv/bin/activate`。

## VS Code 与 Jupyter

项目已通过 `.vscode/settings.json` 将默认解释器设为 `.venv/bin/python`。打开笔记本后，选择 **Kaggle Course — Python 3.12** 内核。

## 遥感分类基线

在项目根目录运行 `.venv/bin/python src/baseline.py --smoke`，只训练第 1 折的第 1 个 epoch；单轮损失、验证准确率和耗时写入 `runs/smoke/fold_1/epochs.csv`。首次运行会生成并固定 `runs/folds.csv`；所有后续实验必须复用此文件，不能重新生成折划分。

查看冒烟结果并确认后，再运行 `.venv/bin/python src/baseline.py` 训练完整的 5 折 × 10 epochs。参数在 `configs/baseline.yaml`；训练脚本在导入 PyTorch 前为当前进程设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`。首次运行可能需要下载 ImageNet 预训练权重。

正式训练把每折逐轮指标、最终模型、验证概率、文件顺序及混淆矩阵写入 `runs/baseline/fold_N/`。`runs/baseline/oof_probs.npy` 与 `oof_files.csv` 的行顺序一致；`runs/baseline/test_probs.npy` 与 `data/raw/sample_submission.csv` 的行顺序一致。`runs/results.csv` 追加每折最终与最佳 epoch 指标，以及五折均值和标准差。正式训练完成后生成并校验项目根目录的 `submission.csv`。这些文件只在完整训练后产生；冒烟结果不会加入正式汇总。

## PyTorch MPS

Apple Silicon 上的 PyTorch 自动支持 MPS。为使暂不支持的 MPS 算子自动回退到 CPU，请仅在当前命令或当前终端会话中设置：

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python src/check_env.py
```

或先执行 `export PYTORCH_ENABLE_MPS_FALLBACK=1`，再运行训练命令。不要将该变量写入系统配置文件。

## 说明

- LightGBM 在 macOS 上依赖 Homebrew 的 `libomp`：`brew install libomp`。
- 当前未安装 `open3d`、`transformers`、`sentence-transformers`，待题目确定后按需安装。
- 不使用 CUDA 相关依赖；请勿安装 `bitsandbytes`、`flash-attn` 等 CUDA 专用包。