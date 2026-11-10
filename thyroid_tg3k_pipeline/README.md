# TG3K 预训练 + 本地原始甲状腺标注训练（第三版）

项目目录：`E:\thyroid_tg3k_pipeline`。本版按最新要求只保留本地原始甲状腺标注非空的“图像—标注”对；TG3K 原 unused 的 802 对加入最终测试集。网络和病例划分保持不变。未执行训练或真实图像模型推理。

## 当前流程

1. TG3K 使用已确认的 train 2244 对、val 539 对预训练。
2. 从已筛选的 90 个本地腺体可用病例中，提取原始 gland NRRD 非空的全部帧及对应 BMP。仅去掉空标注帧；不沿用此前少量抽样帧的限制，不使用模型补标。
3. 本地病例划分仍为 train 62、val 14、test 14。同一病例的全部保留帧仍在原集合中。
4. 从 TG3K best.pt 初始化整个 MobileNetV2 + U-Net，用本地训练集继续训练编码器和解码器，用本地验证集选 best.pt。
5. 最终测试集 = 本地 14 个测试病例的原标注非空帧 + TG3K 原 unused 的全部 802 对。两部分按来源分别统计，并提供合并的逐帧均值。

原始 E 盘 BMP 和 D 盘 NRRD 不删除、不改写。“只保留”指新数据清单和导出的训练掩膜仅纳入上述配对。原标注区域不会被模型推理结果替换。

## 数据划分及来源

| 数据来源 | 训练 | 验证 | 最终测试 |
|---|---:|---:|---:|
| TG3K 图像—标注对 | 2244（预训练） | 539（预训练） | 原 unused 的 802 对 |
| 本地可用病例 | 62 | 14 | 14 |

本地各集合的实际帧数见 `data/original_pairs_v3/complete.json`，逐病例帧数见同目录 `case_counts.csv`。`configs/cases.json` 固定了90个病例的路径、原标注 SHA256、原始帧数、split。名单继承 `provenance/manifest.json` 中 gland 任务的病例，归属继承 `provenance/case_split.json`。

TG3K 路径为 `E:\Thyroid Dataset\tg3k`。原方案位于 `provenance/tg3k/`，按连续相同图像尺寸形成15个可观察组，在组内设置验证窗口及间隔。**这15组不是已核实的患者编号。** 原 unused 的802帧是窗口旁的间隔帧；加入测试集后，仍可能与预训练帧相邻。它们是补充测试数据，不应解释为独立患者或完全隔离序列测试。预训练 train/val 清单保持原样，802帧没有进入任何训练或验证集合。

## 运行方式

默认解释器：`D:\anaconda\envs\pytorch\python.exe`，已用于验证的版本为 Python 3.12.11、torch 2.7.1+cu126、torchvision 0.22.1+cu126、NumPy 2.4.4、Pillow 11.0.0。其他机器的依赖见 `requirements.txt`。

```powershell
Set-Location E:\thyroid_tg3k_pipeline

# 检查原图帧号、尺寸、标注哈希；生成TG3K预训练清单
.\run_step.ps1 -Step prepare

# 提取原始阳性标注并生成本地清单、混合测试清单
# 如 data/original_pairs_v3/complete.json 已存在，则数据已准备完成，无需重复执行。
.\run_step.ps1 -Step pairs

# 用户需要训练时再分别运行以下命令
.\run_step.ps1 -Step pretrain
.\run_step.ps1 -Step finetune
.\run_step.ps1 -Step evaluate
```

不再运行 `correct`。`reconcile.py` 和对应旧规则的工具函数仅保留为历史代码，不在第三版入口中使用；旧 `outputs/corrected_*/manifest.csv` 不能用于第三版训练/测试。新清单使用完成标志和哈希校验，拒绝伪标签、病例划分错误和TG3K测试数据混入本地训练。

等价的显式命令如下：

```powershell
$pythonExe = 'D:\anaconda\envs\pytorch\python.exe'
& $pythonExe prepare_pairs.py
& $pythonExe train.py --stage pretrain --run tg3k_v1
& $pythonExe train.py --stage finetune --init runs/tg3k_v1/best.pt --manifest data/original_pairs_v3/local_pairs.csv --run local_original_v3
& $pythonExe evaluate.py --checkpoint runs/local_original_v3/best.pt --manifest data/original_pairs_v3/final_test.csv --run test_original_v3
```

已存在的训练、评估和配对输出目录拒绝覆盖。需要重建配对时使用 `prepare_pairs.py --out data/original_pairs_v4`，并在训练、评估时传入新清单。训练中断不支持断点恢复，已有最佳权重与历史会保留，新训练须换运行名。

## 输出文件

| 文件 | 内容 |
|---|---|
| `data/tg3k.csv` | 原TG3K划分；train/val用于预训练，unused不参与预训练 |
| `data/original_pairs_v3/local_pairs.csv` | 本地原始非空甲状腺配对，含原病例train/val/test归属 |
| `data/original_pairs_v3/masks/WDxxxxxxxx/N.png` | 原尺寸二值甲状腺掩膜，区域与原NRRD一致 |
| `data/original_pairs_v3/final_test.csv` | 本地测试配对 + 802对TG3K原unused数据 |
| `data/original_pairs_v3/case_counts.csv` | 逐病例保留数及排除的空标注帧数 |
| `data/original_pairs_v3/complete.json` | 数据规模、清单哈希、第三版策略与完成标志 |
| `runs/tg3k_v1/best.pt` | 预训练最佳权重（运行训练后生成） |
| `runs/local_original_v3/best.pt` | 本地继续训练的最佳权重（运行训练后生成） |
| `evaluations/test_original_v3/frames.csv` | 含dataset字段的逐帧Dice/IoU |
| `evaluations/test_original_v3/cases.csv` | 本地按病例、TG3K按原尺寸组汇总 |
| `evaluations/test_original_v3/summary.json` | local、tg3k_unused分别统计及合并逐帧均值 |

清单通过绝对路径引用原图，不复制BMP。TG3K测试掩膜引用原文件，不修改原TG3K数据目录。仅新导出本地阳性帧的二值PNG掩膜。不处理cs、结节或良恶性标注。

## 网络和训练设置（保持不变）

MobileNetV2编码器提取stride 2/4/8/16/32的16/24/32/96/1280通道特征，U-Net解码器逐级上采样并拼接跳跃连接，输出一个甲状腺通道。TG3K预训练与本地继续训练使用同一网络结构。默认不下载ImageNet权重，配置项可控制首次预训练初始化方式。

灰度复制为RGB，保持比例缩放并补边到256×256，ImageNet均值/标准差归一化；填充区域不计损失与训练期指标，掩膜最近邻缩放。TG3K的JPEG掩膜使用128阈值去除压缩噪声。最终测试将预测概率还原至原图尺寸，再按0.5阈值计算Dice/IoU。

AdamW，损失0.5×BCE + Dice，预训练学习率1e-4、本地继续训练5e-5，batch size 8、最多100 epochs、20 epochs早停、余弦退火、CUDA混合精度、梯度裁剪、按病例均衡采样。只有训练集做增强。配置见 `configs/project.json`。本版所有配对的source都是provided；旧pseudo_weight配置不会影响这些原标注样本。

## 验证

```powershell
.\run_step.ps1 -Step test
```

检查包含原始标注不变、坐标恢复、NRRD读写、固定病例与TG3K划分、只使用原标注、TG3K原unused仅允许进入测试、不纳入空标注、网络输出与填充区域损失。旧补标工具的测试仅验证保留的历史函数，不代表本版会执行补标。

`reports/preflight.json` 保留已完成的全部72,264张本地BMP的帧号/尺寸检查记录。实际保留的非空标注帧数以第三版配对完成报告为准。无已确认的本地阴性帧，因此本地指标反映有甲状腺标注帧的分割表现。
