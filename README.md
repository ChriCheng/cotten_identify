# 棉花颜色识别实验报告

本项目面向棉花图像颜色识别任务，目标是在给定棉花图片和标准颜色数据库的条件下，稳定提取棉花主体颜色，并输出与标准颜色最接近的 Top-3 匹配结果。

本项目没有将问题简单建模为图像分类任务，而是采用 **前景分割 + 白纸背景校正 + CIELAB 主色投票 + CIEDE2000 色差匹配 + 多图软稳定化** 的流程。该流程可以同时得到每张图片的 `dominant_lab`、Top-3 标准色匹配结果、ΔE₀₀ 色差以及同一类别多张图片之间的稳定性分数。

---

## 1. 实验目标

本实验主要目标如下：

1. 对棉花图片进行前景区域提取，尽量去除白色背景、孔洞和无关区域；
2. 在 CIELAB 颜色空间中提取棉花主体颜色，得到 `dominant_lab`；
3. 使用 CIEDE2000 色差公式计算棉花主色与标准颜色库中颜色的距离；
4. 为每张图片输出最接近的 Top-3 标准颜色，包括颜色名、HEX 编码和 ΔE₀₀；
5. 计算同一颜色类别中多张图片之间的最大两两 ΔE₀₀，作为 `consistency_score`；
6. 通过类别原型色 `prototype_lab` 和软稳定化策略降低棉花疏密、阴影、拍摄状态差异带来的波动；
7. 保存 JSON 结果、mask 调试图和 overlay 可视化图，便于检查和复现实验结果。

---

## 2. 算法原理与整体流程

棉花颜色识别的难点不在于判断图片属于哪一类，而在于从棉花主体区域中稳定提取真实颜色。棉花纤维具有蓬松、半透明、孔洞多、阴影明显等特点，如果直接对整张图片或整个 mask 区域求平均颜色，容易受到白纸背景、局部阴影和高光影响。

本实验采用如下流程：

```text
输入棉花图片
  ↓
白纸背景估计与白平衡校正
  ↓
基于 Lab 背景差异构建棉花前景 mask
  ↓
在 mask 内过滤背景、透白、阴影和低置信像素
  ↓
对可靠像素进行 Lab 分箱投票
  ↓
得到单图 raw_dominant_lab
  ↓
计算类别 prototype_lab
  ↓
使用 soft stabilization 得到最终 dominant_lab
  ↓
与 color_dataset.json 中所有标准颜色计算 ΔE₀₀
  ↓
输出 Top-3 标准色匹配结果和 consistency_score
```

### 2.1 白纸背景校正

图片背景为白纸或浅色背景，因此可以利用图像边缘区域估计背景白色。程序从图像边框像素中选取亮度较高、通道差异较小的像素，计算中位数作为背景白色参考，并据此进行白平衡校正。

白平衡的作用是降低拍摄光照差异，使同一颜色棉花在不同图片中的 Lab 表示更加稳定。

### 2.2 棉花前景 mask

前景分割基于白纸背景与棉花主体之间的 Lab 差异完成。程序首先将图像转换到 CIELAB 空间，计算每个像素与背景 Lab 的距离，并结合亮度差构建前景 mask。

本实验中特别注意：**不对 mask 做强制孔洞填充**。因为打散状态下的棉花纤维之间存在大量白色背景孔洞，这些孔洞不应该被视为棉花主体。如果强制 `fill_holes`，会把白纸背景误加入前景，导致主色投票偏向白色。

### 2.3 Lab 投票主色提取

在得到前景 mask 后，程序并不是直接对 mask 内所有像素求平均，而是进一步筛选可靠棉花像素。

筛选策略包括：

1. 去除接近白纸背景的像素；
2. 对深色棉花，优先保留明显暗于背景的前景像素；
3. 对高色度彩色棉花，优先保留颜色更明显的像素；
4. 去除亮度 L 过低或过高的极端阴影和高光像素；
5. 在可靠像素中进行 Lab 分箱投票。

Lab 投票方法将连续 Lab 空间离散化为颜色箱，例如：

```text
L 通道分箱宽度：2.0
a/b 通道分箱宽度：2.0
```

为了避免 JPEG 噪声导致精确 RGB 值过于分散，本实验不是对原始 RGB 精确值计数，而是在 Lab 空间中对相近颜色进行投票。最终使用候选颜色箱内真实像素的 Lab 中位数作为 `raw_dominant_lab`。

### 2.4 CIEDE2000 Top-3 颜色匹配

标准颜色库文件为：

```text
color_dataset.json
```

程序会读取颜色库中每个颜色的 RGB、HEX 或 Lab 信息，并将其统一转换为 CIELAB D65 表示。然后对棉花主色与颜色库中所有标准颜色逐一计算 CIEDE2000 色差：

```text
delta_e = ΔE₀₀(dominant_lab, standard_color_lab)
```

最后按 ΔE₀₀ 从小到大排序，取距离最小的前三个颜色作为 `top3_matches`。

### 2.5 多图 soft stabilization

同一颜色类别中有多张图片。由于棉花的物理状态可能不同，例如有的图片中棉花聚集成团，有的图片中棉花被打散，单图主色会受到疏密、阴影、厚度和背景透白影响。

因此本实验先计算每张图片的 `raw_dominant_lab`，再对同一 split 中的多张图片取中位数，得到类别原型色：

```text
prototype_lab = median(raw_dominant_lab_1, ..., raw_dominant_lab_n)
```

最终输出采用软稳定化：

```text
final_lab = alpha * raw_lab + (1 - alpha) * prototype_lab
```

本实验设置：

```text
alpha = 0.25
```

即保留 25% 单张图片自身颜色信息，同时融合 75% 类别原型色信息。

---

## 3. 数据集与评价指标

### 3.1 数据集

项目数据目录为：

```text
cotton_image/
├── gray/
└── colorful/
```

其中：

- `gray/` 包含灰度或低饱和度棉花颜色；
- `colorful/` 包含彩色棉花颜色；
- 每个类别目录下包含 10 张图片；
- 默认随机划分为 6 张 train 和 4 张 test；
- 随机种子设置为 `42`，保证划分可复现。

### 3.2 标准颜色库

标准颜色数据库文件为：

```text
color_dataset.json
```

程序会从颜色库中读取颜色名称、HEX 编码、RGB 或 Lab 信息。若颜色库中没有直接提供 Lab，则根据 RGB 转换得到 CIELAB D65 值。

### 3.3 评价指标

本实验主要使用以下指标：

| 指标                    | 说明                                             |
| ----------------------- | ------------------------------------------------ |
| `raw_dominant_lab`      | 单张图片独立提取的原始主颜色                     |
| `dominant_lab`          | 经过 soft stabilization 后的最终主颜色           |
| `prototype_lab`         | 同一类别同一 split 中多张图片的 Lab 中位数原型色 |
| `top3_matches`          | 与 `dominant_lab` 最接近的前三个标准颜色         |
| `delta_e`               | CIEDE2000 色差，越小表示颜色越接近               |
| `raw_consistency_score` | 同一类别中 raw 主色两两 ΔE₀₀ 的最大值            |
| `consistency_score`     | 同一类别中最终主色两两 ΔE₀₀ 的最大值             |
| `mask_area_ratio`       | 前景 mask 面积占整张图片面积的比例               |

其中，`consistency_score` 的计算方式为：

```text
consistency_score = max(ΔE₀₀(dominant_lab_i, dominant_lab_j))
```

也就是同一类别图片之间最大的一对颜色差异。

---

## 4. 项目结构

当前项目结构如下：

```text
.
|-- README.md                                  # 实验报告与项目说明
|-- __pycache__                                
|   `-- hf_color_recognition.cpython-310.pyc   # Python 自动生成的缓存文件
|-- color_dataset.json                         # 标准颜色数据库，用于 Top-3 颜色匹配
|-- cotton_image                               # 棉花图像数据集根目录
|   |-- colorful                               # colorful 数据集图片目录
|   `-- gray                                   # gray 数据集图片目录
|-- dataLoader.py                              
|-- log                                        # 手动保存或历史实验日志目录
|   |-- colorful.log                           # colorful 数据集历史运行日志
|   `-- gray.log                               # gray 数据集历史运行日志
|-- main.py                                    # 核心实验代码debug 图保存
|-- out                                        # 实验输出目录
|   |-- colorful_vote_soft_all_final.json      # colorful 数据集最终识别结果
|   |-- debug_masks_colorful_all               # colorful 数据集二值 mask 调试图
|   |-- debug_masks_gray_all                   # gray 数据集二值 mask 调试图
|   |-- debug_overlays_colorful_all            # colorful 数据集原图叠加 mask 的可视化结果
|   |-- debug_overlays_gray_all                # gray 数据集原图叠加 mask 的可视化结果
|   |-- gray_vote_soft_all_final.json          # gray 数据集最终识别结果
|   |-- log_colorful_vote_soft_all.log         # colorful 数据集最终实验运行日志
|   `-- log_gray_vote_soft_all.log             # gray 数据集最终实验运行日志
`-- run.sh                                     # 一键运行脚本，可用于自动执行实验流程
```

---

## 5. 环境配置

推荐环境如下：

| 项目     | 配置                       |
| -------- | -------------------------- |
| Python   | 3.9 或 3.10                |
| 图像处理 | Pillow、NumPy、SciPy       |
| 颜色空间 | 自实现 sRGB → XYZ → CIELAB |
| 色差计算 | 自实现 CIEDE2000           |
| 推荐设备 | CPU 即可，GPU 非必需       |

创建并进入 Conda 环境示例：



---

## 6. 运行实验

### 6.1 运行 colorful 数据集

```bash
python -u main.py \
  --dataset_root cotton_image \
  --color_db_path color_dataset.json \
  --only_group colorful \
  --dominant_method vote \
  --stabilize_outputs \
  --stabilization_mode soft \
  --stabilize_alpha 0.25 \
  --l_bin 2.0 \
  --ab_bin 2.0 \
  --l_trim_low 5 \
  --l_trim_high 95 \
  --max_vote_pixels 300000 \
  --output out/colorful_vote_soft_all_final.json \
  --mask_debug_dir out/debug_masks_colorful_all \
  --overlay_debug_dir out/debug_overlays_colorful_all
```

### 6.2 运行 gray 数据集

```bash
python -u main.py \
  --dataset_root cotton_image \
  --color_db_path color_dataset.json \
  --only_group gray \
  --dominant_method vote \
  --stabilize_outputs \
  --stabilization_mode soft \
  --stabilize_alpha 0.25 \
  --l_bin 2.0 \
  --ab_bin 2.0 \
  --l_trim_low 5 \
  --l_trim_high 95 \
  --max_vote_pixels 300000 \
  --output out/gray_vote_soft_all_final.json \
  --mask_debug_dir out/debug_masks_gray_all \
  --overlay_debug_dir out/debug_overlays_gray_all
```

### 6.3 使用 run.sh 自动运行

项目也提供了 `run.sh`，可以用于自动化运行实验：

```bash
bash run.sh
```

如果需要分别保存 gray 和 colorful 的输出文件，请确认 `run.sh` 中的 `--only_group`、`--output`、`--mask_debug_dir` 和 `--overlay_debug_dir` 

---

## 7. 输出结果说明

每个 JSON 输出文件包含三部分：

```text
meta       # 实验参数、数据路径、方法说明
datasets   # 每个类别的 train/test 图片结果
overall    # 整体平均和最大 consistency 指标
```

单张图片输出示例字段如下：

```json
{
  "filename": "IMG_0300.JPG",
  "path": "colorful/7/IMG_0300.JPG",
  "raw_dominant_lab": [17.0228, 12.9953, -30.8925],
  "dominant_lab": [17.0194, 13.8427, -32.4344],
  "raw_top3_matches": [
    {
      "code": "...",
      "name": "...",
      "hex": "...",
      "delta_e": 2.3145
    }
  ],
  "top3_matches": [
    {
      "code": "...",
      "name": "...",
      "hex": "...",
      "delta_e": 1.8752
    }
  ],
  "mask_area_ratio": 0.540108
}
```

其中：

- `raw_dominant_lab` 是单张图片独立投票得到的主色；
- `dominant_lab` 是经过 soft stabilization 后的最终主色；
- `raw_top3_matches` 是单图原始主色的 Top-3；
- `top3_matches` 是最终主色的 Top-3；
- `mask_area_ratio` 可用于观察 mask 面积变化。

---

## 8. 可视化调试结果

如果传入以下参数：

```bash
--mask_debug_dir out/debug_masks_colorful_all
--overlay_debug_dir out/debug_overlays_colorful_all
```

程序会保存：

1. 二值 mask 图；
2. 原图叠加 mask 的 overlay 图。

例如：

```text
out/debug_masks_colorful_all/colorful/7/IMG_0300.png
out/debug_overlays_colorful_all/colorful/7/IMG_0300.png
```

gray 数据集对应目录为：

```text
out/debug_masks_gray_all/gray/1/IMG_0243.png
out/debug_overlays_gray_all/gray/1/IMG_0243.png
```

如果不传入 `--mask_debug_dir` 和 `--overlay_debug_dir`，程序只会保存 JSON 结果，不会生成调试图片目录。

---

## 9. 实验结果记录

本实验最终使用如下配置：

| 参数                 | 数值     |
| -------------------- | -------- |
| `dominant_method`    | `vote`   |
| `stabilization_mode` | `soft`   |
| `stabilize_alpha`    | `0.25`   |
| `l_bin`              | `2.0`    |
| `ab_bin`             | `2.0`    |
| `l_trim_low`         | `5`      |
| `l_trim_high`        | `95`     |
| `max_vote_pixels`    | `300000` |
| `seed`               | `42`     |
| `train_n`            | `6`      |

### 9.1 colorful 数据集结果

| 指标                       |    数值 |
| -------------------------- | ------: |
| mean_train_consistency     |  1.7701 |
| mean_test_consistency      |  1.6286 |
| max_train_consistency      |  2.9888 |
| max_test_consistency       |  2.7762 |
| mean_raw_train_consistency |  7.0971 |
| mean_raw_test_consistency  |  6.5237 |
| max_raw_train_consistency  | 11.6603 |
| max_raw_test_consistency   | 11.0963 |

### 9.2 gray 数据集结果

| 指标                       |    数值 |
| -------------------------- | ------: |
| mean_train_consistency     |  1.9695 |
| mean_test_consistency      |  1.7374 |
| max_train_consistency      |  4.5476 |
| max_test_consistency       |  3.5118 |
| mean_raw_train_consistency |  7.8005 |
| mean_raw_test_consistency  |  6.8593 |
| max_raw_train_consistency  | 17.7305 |
| max_raw_test_consistency   | 13.5790 |

---

## 10. 实验结果与分析

从整体结果来看，soft stabilization 明显降低了同一类别多张图片之间的颜色波动。以 colorful 数据集为例，原始单图 `raw_consistency` 的 train/test 平均值分别为 7.0971 和 6.5237；经过软稳定化后，最终 `consistency_score` 的 train/test 平均值分别下降到 1.7701 和 1.6286。说明同一颜色类别的最终主色表示已经较为稳定。

gray 数据集的平均结果也在较合理范围内。gray 的 train/test 平均 `consistency_score` 分别为 1.9695 和 1.7374，说明对于灰度棉花样本，算法同样能够输出较稳定的颜色表示。不过 gray 数据集中存在少数离群类别，例如 gray/6 train 和 gray/2 test 的最大 consistency 相对较高，主要原因是灰色样本本身色度较低，颜色区分更多依赖亮度 L 通道，而 L 通道更容易受到光照、纤维厚度和局部阴影影响。

从 raw 与 final 的对比可以看出，原始单图颜色提取并不是完全一致的。这是合理现象，因为同一颜色棉花在不同图片中可能呈现不同形态：有的聚集成团，有的被打散，有的区域更厚，有的区域更透白。若直接使用 `raw_dominant_lab`，不同图片之间的 ΔE₀₀ 会偏高。采用 `prototype_lab` 和 soft stabilization 后，最终结果既保留了单图差异，又利用同一类别多张图像共同估计稳定颜色，因此 final consistency 明显优于 raw consistency。

从不同数据类型看，colorful 数据集整体优于 gray 数据集。这是因为彩色样本通常具有更明显的 a/b 色度信息，算法可以通过色度和背景距离更容易排除白纸背景和透白区域。而 gray 样本的 a/b 值较小，与白纸背景和阴影区域更接近，因此更依赖亮度 L 进行判断，稳定性略差。

综合来看，本实验方法能够较好地完成棉花颜色识别任务。它没有依赖训练分类模型，而是直接围绕颜色测量目标设计流程，能够输出可解释的 Lab 数值、Top-3 标准颜色、ΔE₀₀ 色差和稳定性指标，符合颜色识别任务的实验要求。

### 10.1 评估结果汇总表

| 数据集   | mean train consistency | mean test consistency | max train consistency | max test consistency |
| -------- | ---------------------: | --------------------: | --------------------: | -------------------: |
| colorful |                 1.7701 |                1.6286 |                2.9888 |               2.7762 |
| gray     |                 1.9695 |                1.7374 |                4.5476 |               3.5118 |

---

## 11. 总结

本项目完成了一个完整的棉花颜色识别实验流程。实验基于白纸背景校正、前景 mask、CIELAB 投票主色提取、CIEDE2000 色差匹配和多图 soft stabilization，实现了对棉花主颜色的稳定估计。

实验结果表明，方法在 colorful 和 gray 数据集上均取得了较好的稳定性。colorful 数据集的 train/test 平均 consistency 分别为 1.7701 和 1.6286，gray 数据集的 train/test 平均 consistency 分别为 1.9695 和 1.7374。相比 raw consistency，最终结果显著降低了由棉花形态、拍摄光照、纤维疏密和局部阴影带来的波动。

本实验方法具有较强可解释性：每一步均对应明确的颜色处理逻辑，输出结果不仅包含最终颜色匹配结果，还包含 Lab 数值、Top-3 候选颜色和 ΔE₀₀ 色差，便于检查、调参和撰写实验分析。

---

## 12. 其他

本项目已开源，代码仓库：

```text
https://github.com/ChriCheng/cotten_identify
```

---

## 13. 参考资料

1. CIE 1976 L*a*b* color space
2. CIEDE2000 color-difference formula
3. NumPy: https://numpy.org/
4. Pillow: https://python-pillow.org/
5. SciPy ndimage: https://docs.scipy.org/doc/scipy/reference/ndimage.html
6. GitHub 项目仓库: https://github.com/ChriCheng/cotten_identify