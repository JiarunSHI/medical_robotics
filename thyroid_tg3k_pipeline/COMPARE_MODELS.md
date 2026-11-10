# 两阶段模型的本地随机帧对比

运行环境沿用 `D:\anaconda\envs\pytorch\python.exe`。默认权重为项目内 `downloaded_weights/tg3k_best.pt` 与 `downloaded_weights/local_best.pt`，分别是 TG3K 预训练和本地继续训练后的 MobileNetV2 + U-Net，并非两个不同网络结构。

## 一条命令运行

```powershell
Set-Location E:\thyroid_tg3k_pipeline
.\run_step.ps1 -Step compare
```

默认从原有14个测试病例中无放回随机选8个病例，每例选2张原始腺体标注非空帧，共16张。种子为20261017，两个模型使用同一批图像。自动使用CUDA（可用时），否则CPU。不启动训练、不修改权重、原图或原始标注。

```powershell
# 覆盖全部14个测试病例，每例随机选3帧，换一个抽样种子
& 'D:\anaconda\envs\pytorch\python.exe' compare_models.py --cases 14 --frames-per-case 3 --seed 42

# 指定病例（必须在选择的split和已筛选可用名单内）
& 'D:\anaconda\envs\pytorch\python.exe' compare_models.py --case-ids 7 11 --frames-per-case 2

# 从包括空标注的全部原始帧中抽样，仅用于探索；空标注不等同于确认阴性
& 'D:\anaconda\envs\pytorch\python.exe' compare_models.py --frame-pool all --cases 8 --frames-per-case 2
```

支持 `--split train/val/test/all`、`--device cpu/cuda`、`--threshold 0.5`、`--batch-size 4`、`--out 新目录` 以及两个独立的权重路径参数。输出目录已存在时拒绝覆盖；不指定时用时间戳生成。

## 查看结果

输出目录位于 `comparisons/compare_日期_时间/`。双击 `index.html` 查看离线图册，点击图像查看原尺寸PNG。

每张原图对应一张四栏 `comparison.png`：

1. 原始超声图像。
2. 直接从 D 盘原始 gland.seg.nrrd 读取的该帧分割，绿色叠加与边界。
3. TG3K 预训练模型分割，橙色叠加与边界。
4. 本地继续训练模型分割，蓝色叠加与边界。

标题注明 WD 病例号、0起始原图帧号、split和阈值；两种预测分别显示相对原NRRD标注的Dice/IoU。预测先去补边，再双线性恢复原图尺寸，最后阈值化。每个面板保持原图尺寸和坐标，不用模型结果替换标注。

每帧另外输出 `original.png`、`annotation_mask.png`、`tg3k_mask.png`、`local_mask.png` 和两份原尺寸 float32 `*_probability.npy`。掩膜为0/255。

- `metrics.csv`：逐帧Dice、IoU及本地模型相对预训练模型的Dice变化。
- `summary.json`：本次抽样的逐帧均值和病例宏平均。
- `selection.json`：选中原图、帧号、原NRRD路径及其哈希，供复现追溯。
- `run.json`：两个权重的路径、SHA256、阶段、最佳轮数、输入尺寸及抽样参数。
- `complete.json`：全部推理和导出完成标志。

本次随机样本指标不替代完整测试集指标。默认只抽取有原始腺体标注的帧；切换all后，对于空标注帧的重合分数不能用来判断实际无腺体时的误检率。
