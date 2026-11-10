# CS 模型的本地可视化推理

```powershell
Set-Location E:\thyroid_tg3k_pipeline
& 'D:\anaconda\envs\pytorch\python.exe' compare_cs.py
```

默认使用 `downloaded_weights/cmunet_cs_best.pt`，网络是原项目的官方 CMUNet（l=7、k=7、3输入通道、1输出通道）。代码保存在 `vendor/cmunet_cs/`，保留MIT许可，只修改MSAG的导入路径以避免与当前项目的src包冲突。

输入是E盘原始BMP超声图像，D盘 `_cs.seg.nrrd` 是原始分割标注对照，不是模型输入。仅处理原CS标签定义的目标，不把CS输出解释为所有血管结构，也不使用gland标注评分。

严格沿用原CS训练预处理：RGB图像等比例双线性缩放、补边为256×256、除255得到0～1值，不做ImageNet均值/标准差归一化。模型logit经Sigmoid后去除补边、双线性恢复原图尺寸，使用 `probability > 0.5` 得到掩膜。权重按完整state_dict严格加载，并校验target、网络配置、原训练/验证病例列表。

`configs/cs_cases.json` 来自旧项目CS任务实际可用病例，共91例，其中13例为test。与腺体任务的90例名单独立，不用腺体任务资格替代CS资格。

默认从13个测试病例内以原图/原标注特征选样，目标为20张有CS标注帧和8张空标注帧，优先覆盖不同病例。可用样本不足时以实际输出数量为准。每例最多40个有标注候选，标注区间前/内/后各最多10个空标注候选，同例选中帧默认至少相隔20帧。选样不依赖模型预测。共享特征函数里的gland_brightness/gland_texture字段在此表示CS标注区域灰度特征，不代表腺体。

结果在 `comparisons/cs_时间戳/index.html`，每张图有三栏：原图、原始CS NRRD标注（绿）、CMU-Net CS预测（蓝）。原标注为空时明确显示unknown，Dice/IoU不计算。支持在图册切换显示有标注/空标注帧。

每帧额外保存原图、原始CS二值掩膜、预测掩膜、原尺寸float32概率数组。frames.csv/results.json记录病例号、原始BMP帧号、分割指标、选样依据及源文件哈希。summary.json只总结有标注抽样帧，不能替代完整测试结果；与旧项目256×256验证指标的计算分辨率也不同。

可选参数：`--split test/val/train/all`、`--checkpoint 权重文件`、`--device cuda/cpu`、`--out 新目录`、`--positive-candidates-per-case 40`、`--empty-candidates-per-region 10`、`--min-gap 20`。旧输出目录拒绝覆盖。

本程序只推理、保存可视化结果，不训练、不修改原图、NRRD或权重，也不把预测自动写回标注文件。
