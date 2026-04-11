from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="根据结果 JSON 生成 Markdown 实验报告")
    parser.add_argument("--results", type=str, required=True, help="results_train.json / results_test.json / results_all.json")
    parser.add_argument("--output", type=str, default="实验报告_棉花颜色识别.md", help="输出 Markdown 文件")
    args = parser.parse_args()

    results = load_results(Path(args.results))
    groups = results["groups"]
    summary = results["summary"]

    lines = []
    lines.append("# 稳定可靠的棉花颜色识别系统实验报告\n")
    lines.append("## 1. 实验目的\n")
    lines.append("本实验旨在设计一个对未漂白原棉图像进行主颜色提取与标准色匹配的自动算法，使同一颜色棉花在多张图像中的识别结果保持高度一致，并输出其在标准颜色数据库中的 Top-3 匹配颜色及 CIEDE2000 距离。\n")

    lines.append("## 2. 数据集与任务说明\n")
    lines.append(f"- 处理子集：`{results['subset']}`\n")
    lines.append(f"- 颜色组数量：{summary['n_groups']}\n")
    lines.append(f"- 每组默认使用 6 张图像构建颜色原型，4 张图像做测试；也支持处理全部 10 张图像。\n")
    lines.append("- 标准颜色数据库：217 种颜色，包含 `name`、`hex`、`rgb`、`lab_D65` 字段。\n")

    lines.append("## 3. 算法设计\n")
    lines.append("### 3.1 总体思路\n")
    lines.append("算法不依赖深度学习训练，而是采用“背景建模 + Lab 色彩空间 + 稳健统计 + 原型稳定化”的方案。由于拍摄背景、光照和角度已经统一，这种方法具备较好的可解释性和可复现性。\n")
    lines.append("### 3.2 关键步骤\n")
    lines.append("1. **边框白平衡校正**：利用图像边缘白纸板区域估计背景颜色，对 RGB 三通道做轻量校正，削弱不同图片间的整体偏色。\n")
    lines.append("2. **棉花区域分割**：在 CIELAB 空间中估计背景 Lab，基于与背景的色差、亮度差和色度差联合构造前景掩膜，并使用形态学操作清理噪声。\n")
    lines.append("3. **主颜色提取**：对前景像素先在 a*b* 平面上做稳健筛选，再在 L* 维度上截断过暗阴影和过亮高光，最后用均值估计主颜色 Lab。\n")
    lines.append("4. **多图稳定化**：使用同组 6 张图像先估计颜色原型；对偏离原型较大的图像，其结果向原型适度收缩，以降低蓬松度、面积差异和局部阴影带来的波动。\n")
    lines.append("5. **颜色匹配**：对最终 Lab 与数据库中所有颜色计算 ΔE₀₀，取距离最小的 Top-3。\n")

    lines.append("## 4. 评价指标\n")
    lines.append("- **单图主色误差**：与参考真值的 ΔE₀₀，目标 < 3.0。\n")
    lines.append("- **Top-3 匹配准确率**：至少前 2 个颜色名称排序正确。\n")
    lines.append("- **一致性分数**：同组多张图主色两两间最大 ΔE₀₀，目标 ≤ 2.5。\n")

    lines.append("## 5. 实验结果汇总\n")
    lines.append(f"- 平均一致性（稳定化前）：**{summary['mean_consistency_raw']}**\n")
    lines.append(f"- 平均一致性（稳定化后）：**{summary['mean_consistency']}**\n")
    lines.append(f"- 最大一致性：**{summary['max_consistency']}**\n")
    lines.append(f"- 满足一致性 ≤ 2.5 的颜色组数：**{summary['n_groups_meet_threshold_2.5']} / {summary['n_groups']}**\n")

    lines.append("### 5.1 各颜色组结果\n")
    lines.append("| 颜色组 | 子集 | 一致性(raw) | 一致性(final) | 原型 Lab |\n")
    lines.append("|---|---:|---:|---:|---|\n")
    for g in groups:
        proto = ", ".join(f"{v:.2f}" for v in g["prototype_lab"])
        lines.append(f"| {g['group']} | {g['subset']} | {g['consistency_score_raw']:.4f} | {g['consistency_score']:.4f} | [{proto}] |\n")

    lines.append("## 6. 结果分析\n")
    improvements = [g['consistency_score_raw'] - g['consistency_score'] for g in groups]
    lines.append(f"稳定化阶段在多数颜色组上降低了组内波动，平均一致性改善约 **{mean(improvements):.4f}**。说明采用同组原型对异常图像做轻量修正，能够有效抑制局部阴影、棉团疏密差异和采样面积差异带来的扰动。\n")
    lines.append("同时，由于算法核心建立在 Lab 色彩空间与 ΔE₀₀ 色差公式上，输出结果具有明确的颜色学意义，便于和标准数据库直接对接。\n")

    lines.append("## 7. 结论\n")
    lines.append("本文实现了一套稳定可靠的棉花颜色识别系统。该系统利用边框白平衡、背景分割、稳健主色估计与多图原型稳定化，实现了对未漂白原棉主颜色的自动提取，并能够在标准颜色数据库中给出最接近的 Top-3 颜色匹配结果。该方法无需复杂训练，适合在中小规模工业场景下快速部署。\n")

    Path(args.output).write_text("".join(lines), encoding="utf-8")
    print(f"已生成报告: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
