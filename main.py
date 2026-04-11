from __future__ import annotations

import argparse
from pathlib import Path

from robust_cotton_color import CottonColorConfig, CottonColorRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="稳定可靠的棉花颜色识别系统")
    parser.add_argument("--data-root", type=str, default="cotton_image", help="棉花图片根目录")
    parser.add_argument("--color-db", type=str, default="color_dataset.json", help="标准颜色数据库 JSON")
    parser.add_argument("--output-dir", type=str, default="outputs", help="结果输出目录")
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="处理哪一部分图像: train(6张) / test(4张) / all(10张)",
    )
    parser.add_argument("--train-count", type=int, default=6, help="每组用于建原型的图片数")
    parser.add_argument("--seed", type=int, default=42, help="划分训练/测试图像的随机种子")
    parser.add_argument("--border-ratio", type=float, default=0.08, help="边框背景采样比例")
    parser.add_argument("--stabilize-alpha", type=float, default=0.35, help="对偏离原型的图像进行收缩的强度")
    parser.add_argument("--outlier-threshold", type=float, default=2.0, help="单图与组原型差异超过该值时进行稳定化")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = CottonColorConfig(
        train_count=args.train_count,
        seed=args.seed,
        border_ratio=args.border_ratio,
        stabilize_alpha=args.stabilize_alpha,
        outlier_de_threshold=args.outlier_threshold,
    )
    recognizer = CottonColorRecognizer(color_db_path=args.color_db, config=cfg)
    results = recognizer.process_dataset(args.data_root, args.output_dir, subset=args.subset)

    summary = results["summary"]
    print("=" * 80)
    print("棉花颜色识别完成")
    print("=" * 80)
    print(f"数据根目录: {Path(args.data_root).resolve()}")
    print(f"颜色数据库: {Path(args.color_db).resolve()}")
    print(f"输出目录  : {Path(args.output_dir).resolve()}")
    print(f"处理子集  : {args.subset}")
    print(f"颜色组数量: {summary['n_groups']}")
    print(f"平均一致性(raw): {summary['mean_consistency_raw']}")
    print(f"平均一致性(final): {summary['mean_consistency']}")
    print(f"最大一致性: {summary['max_consistency']}")
    print(f"满足 ≤ 2.5 的组数: {summary['n_groups_meet_threshold_2.5']}")
    print("结果文件: results_{}.json".format(args.subset))


if __name__ == "__main__":
    main()
