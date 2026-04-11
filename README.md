# 稳定可靠的棉花颜色识别系统

本项目面向“未漂白原棉主颜色识别与标准色匹配”任务，核心目标是：

- 从棉花图像中自动提取主颜色的 CIELAB `Lab` 值；
- 在 `color_dataset.json` 中找到最接近的 Top-3 标准颜色；
- 保证同一颜色棉花多张图像的结果高度一致，组内最大 `ΔE00 < 2.0 ~ 2.5`。

项目采用 **传统颜色科学 + 稳健图像处理** 路线，不依赖神经网络训练，更适合本题这种：

- 背景固定为白纸板；
- 光源与拍摄角度统一；
- 每组颜色样本量较小；
- 更强调一致性、可解释性与稳定性。

---

## 一、算法思路

### 1. 边框白平衡校正
利用图像边框白色纸板区域估计背景 RGB，对三通道做轻度增益修正，降低不同图片之间的整体偏色。

### 2. 棉花区域分割
先将图像转换到 CIELAB 空间，再根据像素与背景 `Lab` 的差异进行前景提取：

- 与背景的色差是否足够大；
- 亮度 `L*` 是否显著低于背景；
- 色度是否显著高于背景。

随后用形态学开闭运算和孔洞填充清理噪声。

### 3. 稳健主色提取
在前景区域中：

- 先在 `a*b*` 平面上剔除离群点；
- 再对 `L*` 做分位数截断，去掉阴影和高光；
- 最终对保留像素求稳健均值，得到单图主颜色 `Lab`。

### 4. 多图稳定化
对同一颜色组的 6 张训练图像先估计一个组级原型 `prototype_lab`。

若某张图的单图结果与原型差异过大（默认 `ΔE00 > 2.0`），则将其结果向原型适度收缩，从而提高多图一致性。

### 5. 标准色匹配
对最终 `Lab` 与 `color_dataset.json` 中全部颜色计算 `ΔE00`，取最小的 Top-3。

---

## 二、项目文件说明

```text
cotton_color_project/
├── README.md                  # 项目说明
├── color_utils.py             # RGB/Lab 转换、ΔE00、颜色库匹配等基础函数
├── robust_cotton_color.py     # 主算法实现
├── main.py                    # 命令行入口
├── generate_report.py         # 根据结果 JSON 自动生成 Markdown 实验报告
└── requirements.txt           # 依赖列表
```

---

## 三、环境依赖

建议 Python 3.10 及以上。

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 四、目录结构要求

你的数据目录可保持如下结构：

```text
colordetect/
├── color_dataset.json
├── cotton_image/
│   ├── colorful/
│   │   ├── 7/
│   │   ├── 8/
│   │   ├── ...
│   └── gray/
│       ├── 1/
│       ├── 2/
│       ├── ...
└── dataLoader.py   # 可选，不强依赖
```

程序会自动递归寻找“包含图片文件的叶子文件夹”，每个叶子文件夹视为一个颜色组。

---

## 五、运行方式

### 1. 处理 6 张训练图像

```bash
python main.py \
  --data-root ./cotton_image \
  --color-db ./color_dataset.json \
  --output-dir ./outputs \
  --subset train
```

### 2. 用 6 张图建原型，测试剩余 4 张

```bash
python main.py \
  --data-root ./cotton_image \
  --color-db ./color_dataset.json \
  --output-dir ./outputs \
  --subset test
```

### 3. 处理全部 10 张图

```bash
python main.py \
  --data-root ./cotton_image \
  --color-db ./color_dataset.json \
  --output-dir ./outputs \
  --subset all
```

---

## 六、输出结果格式

每个颜色组都会输出一个 JSON，例如：

```json
{
  "group": "cotton_image/gray/1",
  "subset": "train",
  "prototype_lab": [28.3, 8.1, 12.7],
  "consistency_score": 1.8,
  "images": [
    {
      "filename": "img_01.jpg",
      "dominant_lab": [28.3, 8.1, 12.7],
      "top3_matches": [
        {"name": "Dark Brown", "hex": "#4A352B", "delta_e": 2.1},
        {"name": "Mocha", "hex": "#5D4037", "delta_e": 3.4},
        {"name": "Espresso", "hex": "#3E2723", "delta_e": 4.7}
      ]
    }
  ]
}
```

同时总目录下会生成：

- `results_train.json`
- `results_test.json`
- `results_all.json`

其中包含所有颜色组的汇总信息。

---

## 七、实验报告自动生成

运行识别后，可根据结果 JSON 自动生成 Markdown 报告：

```bash
python generate_report.py \
  --results ./outputs/results_train.json \
  --output ./实验报告_棉花颜色识别.md
```

然后你可以再将该 Markdown 导出为 PDF 或 Word。

---

## 八、为什么这套方法适合本题

相比训练一个深度模型，这个题目更适合用稳健颜色测量方法，原因有三点：

1. **目标是主颜色估计，不是复杂语义识别。**
2. **图像采集条件已经标准化，背景固定。**
3. **评分强调跨图一致性与可解释性。**

因此，使用 `Lab + ΔE00 + 稳健统计 + 组原型稳定化` 更容易拿到稳定结果，也更容易在实验报告里解释清楚。

---

## 九、可进一步优化的方向

如果你后续想继续提分，可以考虑：

1. 在分割后加入超像素或 GrabCut 做更精细的前景提取；
2. 用边缘区域亮度分布做更强的光照归一化；
3. 将单图颜色改成“主峰聚类中心 + 中位亮度”的组合估计；
4. 对灰色棉花与彩色棉花使用不同阈值策略。

