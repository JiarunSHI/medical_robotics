# 多特征及空标注帧对比

在Windows PowerShell运行：

```powershell
Set-Location E:\thyroid_tg3k_pipeline
& 'D:\anaconda\envs\pytorch\python.exe' compare_diverse.py
```

默认在原有14个测试病例内选样，保持训练/验证/测试划分不变。每病例从原标注非空帧中等距取最多40个候选；在有标注区间之前、内部空隙、之后，各等距取最多10个空标注候选。近乎全黑或缺乏有效图像内容的候选被排除。

使用原标注面积、外接框宽高比、中心位置、外接框填充率，以及标注区域灰度均值、灰度标准差、与周围区域的灰度差、边界梯度进行排序，优先覆盖不同病例，同病例选中帧默认间隔至少20帧。默认最多选20张有标注帧与8张空标注帧；不足时按实际可选数量输出。不使用模型分数或预测结果来挑样。

这些是可观察的相对图像特征，不是病理类别。“区域纹理”以缩略灰度图的标准差近似；“区域对比”是标注内外灰度均值差；图像上下位置不是以毫米标定的组织深度。“填充率”描述掩膜与外接矩形的面积比，不直接等同于边界是否规则。图册提供特征值和候选分位。

原标注为空标记为unknown，不能直接称为无甲状腺的阴性帧。两模型仍推理并展示预测，标注栏明确注明无标注；Dice/IoU为null或CSV空值，不作评分。未知帧不写回训练清单、不自动补标。

结果位于 `comparisons/diverse_时间戳/`，包含离线图册index.html、四栏对比PNG、原图、原NRRD二值标注、两个模型的掩膜与概率数组。每张图对应原始病例/帧号。candidates.json记录候选特征，selection.json记录选择依据，results.json/frames.csv记录预测与标注状态，run.json记录权重哈希。

参数：`--split test`（可选train/val/all）、`--positive-candidates-per-case 40`、`--empty-candidates-per-region 10`、`--min-gap 20`、`--device cuda`、`--out 新输出目录`。改变split仅改变可视化取样范围，不修改已有数据集划分。特征选样指标不应替代完整测试集评估。
