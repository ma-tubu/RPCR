# V2A2T 预抽取特征版

本目录从 `V2A2T_Optimization` 完整复制后独立开发。原始链代码保留不动，新增 EBMC CMU-MOSEI 预抽取特征版。

## 输入与链路

```text
MANet [1024]
  -> Visual residual encoder
  -> layer-wise visual-conditioned MoPE audio encoder
Wav2Vec [512]
  -> gated VA fusion [512]
  -> layer-wise VA-conditioned MoPE text encoder
DeBERTa [1024]
  -> regression prediction [-3, 3]
```

特征目录：

```text
datasets/CMUMOSEI
```

官方划分为 16,326 / 1,871 / 4,659，共 22,856 条。

## 关键文件

- `models/feature_v2a2t.py`：特征域 V→A→T 链式 MoPE 模型。
- `utils/ebmc_feature_dataset.py`：EBMC 标注解析、特征校验和紧凑缓存。
- `utils/regression_metrics.py`：与 CURE-MSA 一致的 MOSEI 指标。
- `main_feature_v2a2t.py`：训练、验证、测试、早停和完整 checkpoint。
- `tools/build_ebmc_feature_cache.py`：将 68,568 个 NPY 转为单个张量缓存。
- `tools/run_feature_v2a2t.ps1`：缓存和训练的一键后台任务入口。
- `tools/check_feature_training.ps1`：不依赖 Codex 的本地状态检查。
- `CHANGELOG.md`：全部代码修改记录。

## 指标口径

每轮打印并写入 `metrics.jsonl`：

- 训练目标 loss。
- Acc-2（排除标签 0，负/正）。
- Acc-2-has0（负/非负）。
- Acc-5、Acc-7。
- F1、F1-has0（weighted F1）。
- MAE、Pearson Corr、RMSE。

最佳 checkpoint 只按验证集 MAE 选择，测试集只在训练完成后评估一次。

## 手动运行

```powershell
conda activate mope_env
cd /path/to/RPCR

.\tools\run_feature_v2a2t.ps1 -RunName ebmc_full_seed42
```

如需进行诊断性逐轮测试，可显式开启：

```powershell
.\tools\run_feature_v2a2t.ps1 `
  -RunName ebmc_test_monitor_seed42 `
  -EvalTestEveryEpoch
```

开启后，每轮测试指标写入 `metrics.jsonl` 的 `test_monitor` 字段。该指标不参与反向传播、早停或 `best.pt` 选择；正式方法选择和调参仍应只依据验证集。

查看状态：

```powershell
.\tools\check_feature_training.ps1
```

恢复训练：

```powershell
.\tools\run_feature_v2a2t.ps1 `
  -RunName ebmc_full_seed42 `
  -Resume .\train_log\ebmc_full_seed42\last.pt
```

恢复时使用原来的 `RunName`，使 `best.pt`、`last.pt` 和 `metrics.jsonl` 保持在同一实验目录。

## Checkpoint

`best.pt` 和 `last.pt` 都保存：

- 模型完整参数。
- 优化器与学习率调度器状态。
- AMP GradScaler 状态。
- Python、NumPy、PyTorch 和 CUDA 随机状态。
- 超参数、当前 epoch、最佳验证 MAE 和指标。

## 数据注意事项

- `manet_UTT/Z3fcd1wdzr0_2.npy` 是全零 float64；缓存时统一转为 float32，模型用可学习 missing embedding 处理零向量。
- 全特征版是话语级特征调谐，不能表述为在原始 Swin/Wav2Vec2/BERT 内部逐层注入。
- 原始数据版与官方特征版的样本数和划分不同，结果不能直接视为同一数据协议下的消融。
