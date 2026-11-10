# 跟踪超声单病例三维重建

当前协议：保留带黑边原始整图左上角像素中心为 `(0,0)`，x 向右、y 向下。默认排除左右黑边，只让有效矩形内像素参与重建；裁剪不改变图像坐标系，范围计算和融合均使用原图坐标。本次只排除左右，上下范围不变。使用列向量：

`p_reference_mm = track @ original_ImageToProbe_XML @ diag(sx,sy,1,1) @ [u_full,v_full,0,1]`

裁剪只负责筛选像素。读取裁剪缓冲后，第 `(col,row)` 个像素仍以 `u_full=left+col`、`v_full=top+row` 参与几何运算（跳像素时乘对应采样步长）。例如裁剪范围从原图 `(148,0)` 开始，该点的变换输入仍为 `(148,0)`。不移动图像原点，不增加标定变换，不修改 XML。之前的局部坐标加补偿方式与此在数学上等价；本版改为显式保留原图坐标，避免混淆。

**像素先转 mm，XML 原样保留，不归一化，不修改标定平移。** 默认输入像素间距 0.10×0.10 mm/pixel 是研究用假设，不是已确认的设备标定。输出体素网格三轴平行参考系；原点由所选像素坐标最小值减边界留量确定。体素索引表示中心位置。

## 环境与运行

在项目根目录（包含 `us_reconstruction` 文件夹）执行。支持 Python 3.8+；本项目不依赖 Numba、GPU 或 Slicer。

当前项目位置：`E:\超声三维重建项目`。

```powershell
cd 'E:\超声三维重建项目'
python -m pip install -r us_reconstruction/requirements.txt

# 只检查输入、范围、网格尺寸和内存，不重建
python -m us_reconstruction.reconstruct --case 'E:\超声_姿态匹配数据集_平滑子集\WD00000023' --dry-run

# 全帧、有效矩形内全部像素重建；默认自动去左右黑边，自动生成新输出目录
python -m us_reconstruction.reconstruct --case 'E:\超声_姿态匹配数据集_平滑子集\WD00000023' --pixel-spacing 0.10 0.10 --voxel-spacing 0.5 0.5 0.5 --axis-gamma 1 1 1 --fusion mean

# 手动指定原图列范围（示例：保留148至572列；右端573不包含；请按实际图像设定）
python -m us_reconstruction.reconstruct --case 'E:\超声_姿态匹配数据集_平滑子集\WD00000023' --crop-x 148 573

# 快速预览：所有帧，在裁剪区域内每隔一个像素取样，保留局部坐标步长
python -m us_reconstruction.reconstruct --case 'E:\超声_姿态匹配数据集_平滑子集\WD00000023' --output 'outputs\WD00000023_preview' --voxel-spacing 1 1 1 --pixel-step 2

# 打开本地交互查看器
python -m us_reconstruction.viewer 'outputs\WD00000023_full'

# 生成查看器截图，无需打开窗口
python -m us_reconstruction.viewer 'outputs\WD00000023_full' --snapshot 'outputs\WD00000023_full\preview.png'

python -m unittest us_reconstruction.test_core -v
```

不指定 `--output` 时自动生成带时间戳的新输出目录。程序拒绝写入非空目录，避免旧数据与新参数混合。

## 可调参数

| 参数 | 默认值 | 含义 / 允许范围 |
|---|---|---|
| `--side-crop` | auto | 自动去左右黑边；none 可复现保留全图的旧模式 |
| `--crop-x LEFT RIGHT` | 未指定 | 手动指定原图列范围，左闭右开；覆盖自动检测 |
| `--black-threshold` | 3 | 自动边界检测使用的灰度阈值，0～254；不用于过滤矩形内像素 |
| `--min-column-fraction` | 0.01 | 一列中超过阈值的像素占比下限，(0,1] |
| `--pixel-spacing SX SY` | 0.10 0.10 | 输入图像 mm/pixel；0.001～5，x/y 分别可设 |
| `--voxel-spacing SX SY SZ` | 0.5 0.5 0.5 | 输出体素 mm；每轴 0.05～5；还受内存预算限制 |
| `--interpolation` | linear | linear 八邻域分配 / nearest 最近体素 |
| `--axis-gamma GX GY GZ` | 1 1 1 | 各参考系轴权重指数，0.25～4，仅 linear 有效 |
| `--fusion` | mean | mean / decayed-mean / maximum |
| `--history-decay` | 1 | 0～1，仅 decayed-mean 使用；越小越偏重新帧 |
| `--fill-radius-mm` | 1 | 默认对明确内部空洞进行插值；0 关闭，最大 10 mm |
| `--track-direction` | probe-to-reference | 若 track 存储反向变换，设 reference-to-probe 后求逆 |
| `--frame-step` | 1 | 排序后每 N 帧选一帧 |
| `--max-frames` | 0 | 0 全部；否则限制所选帧数 |
| `--pixel-step` | 1 | 每 N 个像素选一点，1～16；用于预览，改变采样覆盖 |
| `--padding-voxels` | 1 | 每轴两侧边界留量，1～20 个体素 |
| `--chunk-pixels` | 32768 | 每次处理像素数，256～262144 |
| `--max-memory-gb` | 4 | 保守峰值内存预算（GiB）；有 psutil 时也限制到可用内存的 65% |
| `--dry-run` | 关闭 | 仅几何和内存检查 |

查看器 XYZ 滑块移动三个正交截面和斜截面的交点；Yaw/Pitch 控制斜截面角度；Window/Level 控制灰度；鼠标拖动三维视图转动观察角度。三维区域是有界采样的强度点云预览，不是分割表面。`--threshold 30` 控制点云灰度阈值，`--points 12000` 控制点数上限。阈值不影响重建数据。`Observed only` 隐藏未直接观测的位置；斜截面采用保守的插值支持判定。

重建参数变化后需要重新运行重建；截面、角度、窗宽窗位的调整不需要重建。

## 插值与融合的精确定义

连续体素坐标 `q=(p-origin)/spacing`。linear 对每轴使用 `b0=1-frac(q), b1=frac(q)`；调整为 `b_r**gamma / (b0**gamma+b1**gamma)`，再将三轴权重相乘。gamma=1 是标准三线性分配；大于1更偏近点，小于1更均匀；权重总和为1。nearest 使用 `floor(q+0.5)`。

每帧先在局部体素块累计 `S_frame=sum(w*I)`、`W_frame=sum(w)`，确保同一体素的重复索引不会丢失贡献。

- mean：`S += S_frame; W += W_frame; V=S/W`。
- decayed-mean：仅对当前帧触及的体素执行 `S=lambda*S+S_frame; W=lambda*W+W_frame`；每帧只衰减一次，lambda=1 等价 mean。顺序固定为帧号升序，不是按真实时间衰减。
- maximum：在所有正空间权重的贡献中取最大原像素强度；不将强度乘权重。权重仍累积用于覆盖记录。这是本项目定义，不复制 Plus 对 MAXIMUM 的最小权重阈值。

这是参考 Plus/IGSIO 的分配与累计融合结构的 NumPy 实现，不声称与 Plus 定点数实现逐位一致。没有采用 Plus 的 ushort 累积饱和机制。权重体表示算法累计权重，并非统计置信度。

重建全部帧后默认进行内部空洞插值，半径 1 mm。算法先以 `weights>0` 标记直接观测体素；强度为0但有权重的体素不是空洞。从网格外边界沿空体素做26邻接连通判断，能与外部连通的空区域不填补；只将被观测体素包围的非连通空区域作为内部候选。再限制候选到最近原始观测的物理距离不超过半径。

候选值由原始邻域观测的灰度按空间高斯核和累计权重共同归一化得到：`V_fill = G(V*W) / G(W)`，高斯 sigma=radius/2，各轴按体素间距换算，核在各轴2 sigma处截断。一次性使用冻结的原始观测，不以新填充值继续传播。原始观测强度和权重不改写，填补位置单独保存。

这是保守的内部判定：与外部有缝隙连通的区域即使视觉上像内部缺口，也不会填补；较大空洞中距离原始观测超过半径的部分继续保留为空。不会将整个包围盒或凸包都当成组织内部。增大半径只能扩大邻域，不会使外部空区成为内部。使用 `--fill-radius-mm 0` 关闭。

## 输出和 3D Slicer

- `volume.nrrd`：未补洞的 float32 灰度体，数组 `[z,y,x]`。
- `weights.nrrd`：float32 累计权重。
- `observed_mask.nrrd`：uint8，直接观测为1；真实黑像素也属于观测。
- `filled_volume.nrrd` / `filled_mask.nrrd`：默认输出的内部插值结果和新增赋值位置；保留原始 volume 不覆盖。关闭补洞时不生成。
- `metadata.json`：参数、原始标定矩阵、输入 CSV/XML 哈希、帧号、原点/间距/范围、覆盖率、耗时、每帧有效区域，以及原图坐标约定和补洞策略。

把 `filled_volume.nrrd` 拖入 3D Slicer 可查看内部插值结果；`volume.nrrd` 为插值前对照。查看器传入结果目录时默认优先加载 filled_volume；显式传入 volume.nrrd 可看原始结果，`Observed only` 会隐藏插值位置。NRRD 写入毫米单位、间距和原点，参考 XYZ 数值映射为文件 LPS，Slicer 自动转换到内部 RAS。**该映射仅用于显示和数据交换，不代表已配准到患者解剖方向；不要把 R/L/A/P 标签当作本数据的真实解剖方向。**

要自动显示体绘制和三截面，在 Slicer 的 Python 控制台执行（路径可改）：

```python
exec(open("E:/超声三维重建项目/us_reconstruction/load_in_slicer.py", encoding="utf-8").read())
node = load_reconstruction("E:/超声三维重建项目/outputs/WD00000023_run")
```

在 Slicer 的 Volume Rendering 中调透明度/阈值，在 Reformat 中调任意截面。加载辅助函数默认优先打开补洞结果；使用 `load_reconstruction(folder, use_filled=False)` 可看补洞前体数据。

## 范围与限制

当前只处理本数据集的 8-bit BMP（RGB 将转换为灰度）和无表头 frame+12数的 track.csv。检查帧ID重复、缺图、非有限矩阵和近似刚体姿态；不自动修正时间同步、运动形变或错误标定。本项目筛选有效区域时保留原图坐标；输入数据在此前发生过的未知裁剪无法自动恢复。默认像素间距和 XML 对应的原点仍需通过采集资料确认。

自动检测会汇总本次所选帧的逐列非黑像素占比，同分辨率帧共用一个裁剪框，避免单帧阴影使原点跳动。只裁最左和最右有效列之外的区域；矩形内部的零灰度像素、暗区、血管腔仍参与融合，不进行组织分割。没有足够边界证据时要求手动指定 `--crop-x`；边缘标注、噪声或整个序列的阴影可能影响自动判断，可用手动范围覆盖。改变所选帧可能改变自动范围；跨实验需要相同范围时应明确指定 `--crop-x`。

原始 BMP/XML/track 不会改写。网格范围只从裁剪后参与重建的像素计算，输出体素原点仍按参考系极小值和留量生成。旧输出文件不会自动改变，需重新重建才得到去边结果。粗体素网格和插值不能替代真实采样密度；覆盖率是对整个包围盒统计，不是组织重建准确率。

参考：
- https://pluslib.readthedocs.io/en/latest/algorithms/AlgorithmVolumeReconstruction.html
- https://github.com/IGSIO/IGSIO/blob/master/VolumeReconstruction/vtkIGSIOPasteSliceIntoVolumeHelperCommon.h
- https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html
