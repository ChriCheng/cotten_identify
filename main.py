#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cotton_color_recognition.py

稳健棉花颜色识别脚本
- 适配目录结构:
    root/
      gray/1,2,3,4,5,6
      colorful/7,8,9,10,11,12,13,14,15,16
- 每个类别目录默认 10 张图，按固定随机种子划分 6 张 train / 4 张 test
- 对每张图输出:
    1) 主颜色 Lab
    2) 颜色库 Top-3 最近颜色及 Delta E 2000
- 对每个类别输出:
    1) train/test 一致性 (类内图像主色最大两两 Delta E00)
    2) 原型色 prototype_lab
    3) 原型色的 Top-3 匹配结果

说明:
1. 本脚本不依赖深度学习训练，而是使用“白底估计 + 白平衡 + 前景分割 + 鲁棒主色提取 + DeltaE00 匹配”。
2. 如果颜色库中没有 lab_D65 字段，则自动由 RGB 计算 Lab。
3. 已添加详细进度打印，便于查看当前处理到哪个类别、哪个 split、哪一张图片。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

import numpy as np
from PIL import Image
from scipy import ndimage


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================
# 基础色彩空间转换
# =========================
def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_rgb_to_xyz(rgb_linear: np.ndarray) -> np.ndarray:
    """
    输入: (..., 3), 值域 [0, 1]
    输出: (..., 3), XYZ (D65)
    """
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    return np.tensordot(rgb_linear, M.T, axes=1)


def _f_lab(t: np.ndarray) -> np.ndarray:
    delta = 6 / 29
    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """
    CIE Lab, D65 2°
    输入/输出 shape: (..., 3)
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)  # D65
    xyz_n = xyz / white
    fx, fy, fz = [_f_lab(xyz_n[..., i]) for i in range(3)]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    输入 RGB 可为:
    - uint8 [0,255]
    - float [0,1] 或 [0,255]
    输出 Lab shape 与输入最后一维保持一致
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_linear = _srgb_to_linear(rgb)
    xyz = _linear_rgb_to_xyz(rgb_linear)
    return xyz_to_lab(xyz)


def hex_to_rgb(hex_str: str) -> np.ndarray:
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        raise ValueError(f"非法 hex 颜色: {hex_str}")
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float64)


# =========================
# Delta E 2000
# =========================
def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    支持广播的 CIEDE2000.
    输入 shape: (..., 3)
    返回 shape: broadcast(...)
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt((C_bar**7) / (C_bar**7 + 25**7 + 1e-12)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((C1p * C2p) == 0, 0, dhp)

    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0

    hp_sum = h1p + h2p
    hp_bar = np.where(
        (C1p * C2p) == 0,
        hp_sum,
        np.where(
            np.abs(h1p - h2p) > 180,
            np.where(hp_sum < 360, (hp_sum + 360) / 2.0, (hp_sum - 360) / 2.0),
            hp_sum / 2.0,
        ),
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(hp_bar - 30))
        + 0.24 * np.cos(np.radians(2 * hp_bar))
        + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
        - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    )

    delta_theta = 30 * np.exp(-(((hp_bar - 275) / 25) ** 2))
    Rc = 2 * np.sqrt((Cp_bar**7) / (Cp_bar**7 + 25**7 + 1e-12))
    Sl = 1 + (0.015 * (Lp_bar - 50) ** 2) / np.sqrt(20 + (Lp_bar - 50) ** 2)
    Sc = 1 + 0.045 * Cp_bar
    Sh = 1 + 0.015 * Cp_bar * T
    Rt = -np.sin(np.radians(2 * delta_theta)) * Rc

    kL = kC = kH = 1.0

    dE = np.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return dE


# =========================
# 颜色库
# =========================
@dataclass
class ColorEntry:
    code: str
    name: str
    hex: str
    rgb: np.ndarray
    lab: np.ndarray


def load_color_database(color_db_path: str | Path) -> List[ColorEntry]:
    path = Path(color_db_path)
    log(f"开始加载颜色库: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    log(f"颜色库原始条目数: {len(raw)}")

    colors: List[ColorEntry] = []
    for i, item in enumerate(raw, 1):
        if "lab_D65" in item and item["lab_D65"] is not None:
            lab = np.array(item["lab_D65"], dtype=np.float64)
        else:
            if "rgb" in item and item["rgb"] is not None:
                rgb = np.array(item["rgb"], dtype=np.float64)
            elif "hex" in item and item["hex"]:
                rgb = hex_to_rgb(item["hex"])
            else:
                continue
            lab = rgb_to_lab(rgb)

        if "rgb" in item and item["rgb"] is not None:
            rgb = np.array(item["rgb"], dtype=np.float64)
        elif "hex" in item and item["hex"]:
            rgb = hex_to_rgb(item["hex"])
        else:
            continue

        colors.append(
            ColorEntry(
                code=str(item.get("code", "")),
                name=str(item.get("name", "")),
                hex=str(item.get("hex", "")),
                rgb=rgb,
                lab=np.array(lab, dtype=np.float64),
            )
        )

        if i % 500 == 0:
            log(f"颜色库加载中: {i}/{len(raw)}")

    if not colors:
        raise ValueError(f"颜色库为空或无法解析: {color_db_path}")

    log(f"颜色库加载完成，可用条目数: {len(colors)}")
    return colors


def top_k_matches(lab: np.ndarray, color_db: List[ColorEntry], k: int = 3) -> List[Dict[str, Any]]:
    db_labs = np.array([c.lab for c in color_db], dtype=np.float64)
    delta = delta_e_ciede2000(db_labs, np.asarray(lab, dtype=np.float64))
    order = np.argsort(delta)[:k]

    out: List[Dict[str, Any]] = []
    for idx in order:
        c = color_db[int(idx)]
        out.append(
            {
                "code": c.code,
                "name": c.name,
                "hex": c.hex,
                "delta_e": round(float(delta[idx]), 4),
            }
        )
    return out


# =========================
# 数据集划分
# =========================
def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def split_image_paths(image_paths: List[Path], seed: int = 42, train_n: int = 6) -> Tuple[List[Path], List[Path]]:
    items = list(image_paths)
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) < train_n:
        raise ValueError(f"图像数量不足 {train_n}: {image_paths}")
    return items[:train_n], items[train_n:]


def discover_classes(dataset_root: str | Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    返回:
    {
      "gray": {"1": [img1,...], ...},
      "colorful": {"7": [img1,...], ...}
    }
    """
    dataset_root = Path(dataset_root)
    log(f"开始扫描数据集目录: {dataset_root.resolve()}")

    out: Dict[str, Dict[str, List[Path]]] = {}

    for dataset_type in ["gray", "colorful"]:
        type_dir = dataset_root / dataset_type
        if not type_dir.exists():
            log(f"跳过不存在目录: {type_dir}")
            continue

        cls_map: Dict[str, List[Path]] = {}
        subdirs = sorted([d for d in type_dir.iterdir() if d.is_dir()], key=lambda p: p.name)

        log(f"发现数据集分组: {dataset_type}，子目录数: {len(subdirs)}")

        for subdir in subdirs:
            imgs = list_images(subdir)
            if imgs:
                cls_map[subdir.name] = imgs
                log(f"  类别 {dataset_type}/{subdir.name}: {len(imgs)} 张图")

        if cls_map:
            out[dataset_type] = cls_map

    if not out:
        raise FileNotFoundError(f"未在 {dataset_root} 下找到有效的 gray/colorful 数据目录")

    total_classes = sum(len(v) for v in out.values())
    total_images = sum(len(imgs) for cls_map in out.values() for imgs in cls_map.values())
    log(f"数据集扫描完成: 共 {total_classes} 个类别，{total_images} 张图片")

    return out


# =========================
# 图像处理
# =========================
def read_image_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _border_pixels(rgb: np.ndarray, border_ratio: float = 0.08) -> np.ndarray:
    h, w, _ = rgb.shape
    bh = max(1, int(round(h * border_ratio)))
    bw = max(1, int(round(w * border_ratio)))

    parts = [
        rgb[:bh, :, :].reshape(-1, 3),
        rgb[-bh:, :, :].reshape(-1, 3),
        rgb[:, :bw, :].reshape(-1, 3),
        rgb[:, -bw:, :].reshape(-1, 3),
    ]
    return np.concatenate(parts, axis=0)


def estimate_background_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    估计白纸背景颜色:
    - 优先使用边框
    - 选择亮度较高且三通道差异小的像素
    """
    border = _border_pixels(rgb).astype(np.float64)
    brightness = border.mean(axis=1)
    channel_range = border.max(axis=1) - border.min(axis=1)

    candidates = border[
        (brightness >= np.percentile(brightness, 70))
        & (channel_range <= np.percentile(channel_range, 60))
    ]
    if len(candidates) < 32:
        candidates = border

    bg = np.median(candidates, axis=0)
    return np.clip(bg, 1.0, 255.0)


def white_balance_with_background(rgb: np.ndarray, target_white: float = 245.0) -> np.ndarray:
    bg = estimate_background_rgb(rgb)
    scale = target_white / bg
    balanced = rgb.astype(np.float64) * scale[None, None, :]
    return np.clip(balanced, 0, 255).astype(np.uint8)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    keep = int(np.argmax(sizes) + 1)
    return labeled == keep


def build_cotton_mask(rgb_balanced: np.ndarray) -> np.ndarray:
    """
    基于“与背景颜色的 Lab 差异 + 形态学”的稳健前景分割.
    """
    rgbf = rgb_balanced.astype(np.float64)
    lab = rgb_to_lab(rgbf)
    bg_rgb = estimate_background_rgb(rgb_balanced)
    bg_lab = rgb_to_lab(bg_rgb)

    dist = np.linalg.norm(lab - bg_lab[None, None, :], axis=-1)

    # 边框处通常主要是背景，用它自适应估计阈值
    border_lab = rgb_to_lab(_border_pixels(rgb_balanced))
    border_dist = np.linalg.norm(border_lab - bg_lab[None, :], axis=-1)

    adaptive_thr = max(3.2, float(np.percentile(border_dist, 98) + 1.2))
    mask = dist > adaptive_thr

    # 兼顾非常浅色棉花：只靠 dist 可能偏弱，再引入亮度差
    L = lab[..., 0]
    bg_L = float(bg_lab[0])
    mask |= (bg_L - L) > 2.2

    # 形态学处理
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)

    mask = largest_connected_component(mask)

    # 如果掩膜过小，退化到宽松阈值
    area_ratio = mask.mean()
    if area_ratio < 0.01:
        loose_mask = dist > max(2.2, adaptive_thr * 0.7)
        loose_mask = ndimage.binary_closing(loose_mask, structure=np.ones((5, 5), dtype=bool))
        loose_mask = ndimage.binary_fill_holes(loose_mask)
        loose_mask = largest_connected_component(loose_mask)
        if loose_mask.mean() > area_ratio:
            mask = loose_mask

    return mask.astype(bool)


def robust_dominant_lab(rgb_balanced: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(rgb_balanced.astype(np.float64))
    pixels = lab[mask]

    if len(pixels) == 0:
        # 极端情况下退化为整图中心区域
        h, w = lab.shape[:2]
        y0, y1 = int(h * 0.2), int(h * 0.8)
        x0, x1 = int(w * 0.2), int(w * 0.8)
        pixels = lab[y0:y1, x0:x1].reshape(-1, 3)

    # 剔除高光与过深阴影，提高不同蓬松度/面积下的一致性
    L = pixels[:, 0]
    low_L, high_L = np.percentile(L, [15, 85])
    keep = (L >= low_L) & (L <= high_L)
    core = pixels[keep] if np.count_nonzero(keep) >= 20 else pixels

    # 再用 a,b 的中位数做鲁棒中心，L 用截尾均值
    L_core = core[:, 0]
    a_core = core[:, 1]
    b_core = core[:, 2]

    start_idx = max(0, int(0.1 * len(L_core)))
    end_idx = max(1, int(0.9 * len(L_core)))
    dominant = np.array(
        [
            np.mean(np.sort(L_core)[start_idx:end_idx]),
            np.median(a_core),
            np.median(b_core),
        ],
        dtype=np.float64,
    )
    return dominant


def process_single_image(image_path: str | Path, color_db: List[ColorEntry]) -> Dict[str, Any]:
    rgb = read_image_rgb(image_path)
    rgb_balanced = white_balance_with_background(rgb)
    mask = build_cotton_mask(rgb_balanced)
    dom_lab = robust_dominant_lab(rgb_balanced, mask)
    matches = top_k_matches(dom_lab, color_db, k=3)

    return {
        "filename": Path(image_path).name,
        "dominant_lab": [round(float(x), 4) for x in dom_lab],
        "top3_matches": matches,
        "mask_area_ratio": round(float(mask.mean()), 6),
    }


# =========================
# 统计与汇总
# =========================
def pairwise_max_delta_e(labs: Iterable[np.ndarray]) -> float:
    labs = [np.asarray(x, dtype=np.float64) for x in labs]
    if len(labs) <= 1:
        return 0.0
    max_de = 0.0
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            de = float(delta_e_ciede2000(labs[i], labs[j]))
            max_de = max(max_de, de)
    return max_de


def compute_prototype_lab(labs: List[np.ndarray]) -> np.ndarray:
    arr = np.array(labs, dtype=np.float64)
    return np.array(
        [
            np.median(arr[:, 0]),
            np.median(arr[:, 1]),
            np.median(arr[:, 2]),
        ],
        dtype=np.float64,
    )


def summarize_split(
    image_paths: List[Path],
    color_db: List[ColorEntry],
    dataset_type: str = "",
    cls_name: str = "",
    split_name: str = "",
) -> Dict[str, Any]:
    image_results = []
    total = len(image_paths)

    log(f"开始处理 {dataset_type}/{cls_name} [{split_name}]，共 {total} 张")

    for i, p in enumerate(image_paths, 1):
        log(f"  [{dataset_type}/{cls_name}][{split_name}] 第 {i}/{total} 张: {p.name}")
        r = process_single_image(p, color_db)
        image_results.append(r)
        log(
            f"  [{dataset_type}/{cls_name}][{split_name}] 完成 {i}/{total}: "
            f"mask_area_ratio={r['mask_area_ratio']}, dominant_lab={r['dominant_lab']}"
        )

    labs = [np.array(r["dominant_lab"], dtype=np.float64) for r in image_results]
    prototype_lab = compute_prototype_lab(labs)
    consistency = pairwise_max_delta_e(labs)
    prototype_top3 = top_k_matches(prototype_lab, color_db, k=3)

    log(
        f"完成 {dataset_type}/{cls_name} [{split_name}] 汇总: "
        f"consistency={round(float(consistency), 4)}, "
        f"prototype_lab={[round(float(x), 4) for x in prototype_lab]}"
    )

    return {
        "images": image_results,
        "prototype_lab": [round(float(x), 4) for x in prototype_lab],
        "prototype_top3": prototype_top3,
        "consistency_score": round(float(consistency), 4),
    }


def process_dataset(
    dataset_root: str | Path,
    color_db_path: str | Path,
    seed: int = 42,
    train_n: int = 6,
) -> Dict[str, Any]:
    log("========== 开始整体处理 ==========")
    color_db = load_color_database(color_db_path)
    classes = discover_classes(dataset_root)

    result: Dict[str, Any] = {
        "meta": {
            "dataset_root": str(Path(dataset_root).resolve()),
            "color_db_path": str(Path(color_db_path).resolve()),
            "seed": seed,
            "train_n": train_n,
            "test_n": None,
            "color_db_size": len(color_db),
            "method": {
                "white_balance": "paper background normalization",
                "segmentation": "Lab distance to estimated white background + morphology",
                "dominant_color": "trimmed-L + median(a,b) robust estimator",
                "matching_metric": "CIEDE2000",
            },
        },
        "datasets": {},
    }

    all_train_cons = []
    all_test_cons = []

    total_class_count = sum(len(cls_map) for cls_map in classes.values())
    done_class_count = 0
    log(f"待处理类别总数: {total_class_count}")

    for dataset_type, cls_map in classes.items():
        result["datasets"][dataset_type] = {}
        log(f"开始处理数据分组: {dataset_type}，共 {len(cls_map)} 个类别")

        for cls_name, img_paths in cls_map.items():
            done_class_count += 1
            log(
                f"======== 类别开始 [{done_class_count}/{total_class_count}] "
                f"{dataset_type}/{cls_name}，原始图像数: {len(img_paths)} ========"
            )

            train_imgs, test_imgs = split_image_paths(img_paths, seed=seed, train_n=train_n)
            result["meta"]["test_n"] = len(test_imgs)

            log(
                f"{dataset_type}/{cls_name} 划分完成: "
                f"train={len(train_imgs)} 张, test={len(test_imgs)} 张"
            )

            train_summary = summarize_split(
                train_imgs,
                color_db,
                dataset_type=dataset_type,
                cls_name=cls_name,
                split_name="train",
            )
            test_summary = summarize_split(
                test_imgs,
                color_db,
                dataset_type=dataset_type,
                cls_name=cls_name,
                split_name="test",
            )

            proto_gap = float(
                delta_e_ciede2000(
                    np.array(train_summary["prototype_lab"], dtype=np.float64),
                    np.array(test_summary["prototype_lab"], dtype=np.float64),
                )
            )

            result["datasets"][dataset_type][cls_name] = {
                "train": train_summary,
                "test": test_summary,
                "prototype_gap_train_vs_test": round(proto_gap, 4),
            }

            all_train_cons.append(train_summary["consistency_score"])
            all_test_cons.append(test_summary["consistency_score"])

            log(
                f"======== 类别完成 {dataset_type}/{cls_name}: "
                f"train_consistency={train_summary['consistency_score']}, "
                f"test_consistency={test_summary['consistency_score']}, "
                f"prototype_gap={round(proto_gap, 4)} ========"
            )

    result["overall"] = {
        "mean_train_consistency": round(float(np.mean(all_train_cons)), 4) if all_train_cons else None,
        "mean_test_consistency": round(float(np.mean(all_test_cons)), 4) if all_test_cons else None,
        "max_train_consistency": round(float(np.max(all_train_cons)), 4) if all_train_cons else None,
        "max_test_consistency": round(float(np.max(all_test_cons)), 4) if all_test_cons else None,
    }

    log("========== 全部处理完成 ==========")
    log(f"总体结果: {json.dumps(result['overall'], ensure_ascii=False)}")

    return result


def save_json(data: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="稳健棉花颜色识别")
    parser.add_argument("--dataset_root", type=str, required=False, default="cotton_image/", help="数据集根目录")
    parser.add_argument("--color_db", type=str, default="color_dataset.json", help="颜色库 json 路径")
    parser.add_argument("--output_json", type=str, default="cotton_color_results.json", help="输出结果 JSON")
    parser.add_argument("--seed", type=int, default=42, help="随机划分种子")
    parser.add_argument("--train_n", type=int, default=6, help="每类前景图中划为训练的张数，默认 6")
    args = parser.parse_args()

    log("程序启动")
    log(
        f"参数: dataset_root={args.dataset_root}, "
        f"color_db={args.color_db}, output_json={args.output_json}, "
        f"seed={args.seed}, train_n={args.train_n}"
    )

    results = process_dataset(
        dataset_root=args.dataset_root,
        color_db_path=args.color_db,
        seed=args.seed,
        train_n=args.train_n,
    )

    log(f"开始保存结果到: {args.output_json}")
    save_json(results, args.output_json)
    log(f"[OK] 结果已保存到: {args.output_json}")
    print(json.dumps(results["overall"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()