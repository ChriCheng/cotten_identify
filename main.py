#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

棉花颜色识别：mask 前景 + Lab 投票主色 + CIEDE2000 Top-3 匹配

推荐运行：
python main.py \
  --dataset_root cotton_image \
  --color_db_path color_dataset.json \
  --only_group colorful \
  --dominant_method vote \
  --output outputs/colorful_vote_results.json

如果需要降低同一类图片由亮度/疏密差异造成的波动，建议使用软稳定化：
python main.py \
  --dataset_root cotton_image \
  --color_db_path color_dataset.json \
  --only_group colorful \
  --dominant_method vote \
  --stabilize_outputs \
  --stabilization_mode soft \
  --stabilize_alpha 0.35 \
  --output outputs/colorful_vote_soft_results.json

说明：
- raw_dominant_lab：单张图片独立提取的主色；
- dominant_lab：最终输出主色；默认不稳定化时等于 raw_dominant_lab；
- 软稳定化时 dominant_lab = alpha * raw_dominant_lab + (1 - alpha) * prototype_lab；
- 硬稳定化时所有图片 dominant_lab 完全等于 prototype_lab，因此 consistency 会变成 0，不建议作为主结果。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 1. RGB / Lab 色彩空间转换
# ============================================================
def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_rgb_to_xyz(rgb_linear: np.ndarray) -> np.ndarray:
    # sRGB -> XYZ, D65
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    return np.tensordot(rgb_linear, matrix.T, axes=1)


def _f_lab(t: np.ndarray) -> np.ndarray:
    delta = 6 / 29
    return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    white_d65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz_n = xyz / white_d65

    fx = _f_lab(xyz_n[..., 0])
    fy = _f_lab(xyz_n[..., 1])
    fz = _f_lab(xyz_n[..., 2])

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    rgb 支持 uint8 [0,255] 或 float [0,1]/[0,255]。
    返回 CIELAB D65。
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.size == 0:
        return np.empty_like(rgb, dtype=np.float64)
    if np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return xyz_to_lab(_linear_rgb_to_xyz(_srgb_to_linear(rgb)))


def hex_to_rgb(hex_str: str) -> np.ndarray:
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"非法 HEX 颜色: {hex_str}")
    return np.array([int(s[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


# ============================================================
# 2. CIEDE2000 色差
# ============================================================
def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """支持广播的 CIEDE2000。输入最后一维必须是 3。"""
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

    dE = np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return dE


# ============================================================
# 3. 颜色库加载与 Top-3 匹配
# ============================================================
@dataclass
class ColorEntry:
    code: str
    name: str
    hex: str
    rgb: np.ndarray
    lab: np.ndarray


def load_color_database(color_db_path: str | Path) -> List[ColorEntry]:
    path = Path(color_db_path)
    log(f"加载颜色库: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    colors: List[ColorEntry] = []
    for item in raw:
        if "rgb" in item and item["rgb"] is not None:
            rgb = np.asarray(item["rgb"], dtype=np.float64)
        elif "hex" in item and item["hex"]:
            rgb = hex_to_rgb(item["hex"])
        else:
            continue

        if "lab_D65" in item and item["lab_D65"] is not None:
            lab = np.asarray(item["lab_D65"], dtype=np.float64)
        else:
            lab = rgb_to_lab(rgb)

        colors.append(
            ColorEntry(
                code=str(item.get("code", "")),
                name=str(item.get("name", "")),
                hex=str(item.get("hex", "")),
                rgb=rgb,
                lab=lab,
            )
        )

    if not colors:
        raise ValueError(f"颜色库为空或无法解析: {color_db_path}")

    log(f"颜色库加载完成: {len(colors)} 个可用颜色")
    return colors


def top_k_matches(lab: np.ndarray, color_db: List[ColorEntry], k: int = 3) -> List[Dict[str, Any]]:
    """dominant_lab 与颜色库逐一计算 ΔE00，按距离升序取 Top-k。"""
    query_lab = np.asarray(lab, dtype=np.float64)
    db_labs = np.asarray([c.lab for c in color_db], dtype=np.float64)
    delta = delta_e_ciede2000(db_labs, query_lab)
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


# ============================================================
# 4. 数据集扫描与划分
# ============================================================
def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def split_image_paths(image_paths: List[Path], seed: int = 42, train_n: int = 6) -> Tuple[List[Path], List[Path]]:
    items = list(image_paths)
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) < train_n:
        raise ValueError(f"图片数量不足 {train_n}: {len(items)}")
    return items[:train_n], items[train_n:]


def discover_classes(dataset_root: str | Path) -> Dict[str, Dict[str, List[Path]]]:
    root = Path(dataset_root)
    out: Dict[str, Dict[str, List[Path]]] = {}

    for group in ["gray", "colorful"]:
        group_dir = root / group
        if not group_dir.exists():
            continue
        cls_map: Dict[str, List[Path]] = {}
        for cls_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            imgs = list_images(cls_dir)
            if imgs:
                cls_map[cls_dir.name] = imgs
                log(f"发现类别 {group}/{cls_dir.name}: {len(imgs)} 张图")
        if cls_map:
            out[group] = cls_map

    if not out:
        raise FileNotFoundError(f"未在 {root} 下找到 gray/colorful 数据目录")
    return out


# ============================================================
# 5. 图像读取、白平衡、mask 分割
# ============================================================
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
    """用边框高亮、低色偏像素估计白纸背景。"""
    border = _border_pixels(rgb).astype(np.float64)
    brightness = border.mean(axis=1)
    channel_range = border.max(axis=1) - border.min(axis=1)

    bright_thr = np.percentile(brightness, 70)
    neutral_thr = np.percentile(channel_range, 60)
    candidates = border[(brightness >= bright_thr) & (channel_range <= neutral_thr)]
    if len(candidates) < 64:
        candidates = border
    return np.clip(np.median(candidates, axis=0), 1.0, 255.0)


def white_balance_with_background(rgb: np.ndarray, target_white: float = 245.0) -> np.ndarray:
    bg = estimate_background_rgb(rgb)
    scale = target_white / bg
    out = rgb.astype(np.float64) * scale[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """
    保留面积足够大的前景连通域。
    不强制只保留最大连通域，因为打散棉花可能由多块主体组成。
    """
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask.astype(bool)

    h, w = mask.shape
    min_area = max(16, int(h * w * min_area_ratio))
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_area]

    if not keep_labels:
        keep_labels = [int(np.argmax(sizes)) + 1]

    return np.isin(labeled, keep_labels)


def _border_mask(shape: Tuple[int, int], border_ratio: float = 0.06) -> np.ndarray:
    h, w = shape
    b = max(8, int(round(min(h, w) * border_ratio)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:b, :] = True
    mask[-b:, :] = True
    mask[:, :b] = True
    mask[:, -b:] = True
    return mask


def _local_std(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = ndimage.gaussian_filter(values, sigma=sigma)
    mean_sq = ndimage.gaussian_filter(values * values, sigma=sigma)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _is_gray_scene(lab: np.ndarray) -> bool:
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    border = _border_mask(chroma.shape)
    return float(np.percentile(chroma[border], 95)) < 8.0 and float(np.percentile(chroma, 95)) < 12.0


def _keep_gray_cotton_components(mask: np.ndarray, residual: np.ndarray, texture: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask.astype(bool)

    h, w = mask.shape
    min_area = max(64, int(h * w * 0.0004))
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    mean_residual = ndimage.mean(residual, labeled, index=np.arange(1, num + 1))
    mean_texture = ndimage.mean(texture, labeled, index=np.arange(1, num + 1))

    candidates = np.where(sizes >= min_area)[0]
    if len(candidates) == 0:
        candidates = np.array([int(np.argmax(sizes))])

    scores = sizes[candidates] * np.maximum(mean_residual[candidates], 0.1) * np.maximum(mean_texture[candidates], 0.1)
    best_label = int(candidates[int(np.argmax(scores))] + 1)
    kept = labeled == best_label

    # Keep nearby fragments around the main cotton body, but reject distant paper noise.
    ys, xs = np.where(kept)
    if len(xs) == 0:
        return kept
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    margin = int(round(max(h, w) * 0.08))
    near_box = (
        (np.arange(w)[None, :] >= max(0, x0 - margin))
        & (np.arange(w)[None, :] <= min(w - 1, x1 + margin))
        & (np.arange(h)[:, None] >= max(0, y0 - margin))
        & (np.arange(h)[:, None] <= min(h - 1, y1 + margin))
    )

    strong = (residual > np.percentile(residual[kept], 35)) & (texture > np.percentile(texture[kept], 25))
    return (kept | (mask & near_box & strong)).astype(bool)


def build_gray_cotton_mask(lab: np.ndarray) -> np.ndarray:
    """灰度棉花：用局部背景扣除和纹理抑制纸面渐变噪音。"""
    L = lab[..., 0].astype(np.float64)
    h, w = L.shape
    short = min(h, w)

    background_sigma = max(24.0, short * 0.035)
    texture_sigma = max(3.0, short * 0.004)
    local_bg = ndimage.gaussian_filter(L, sigma=background_sigma)
    residual = local_bg - L
    texture = _local_std(L, sigma=texture_sigma)

    border = _border_mask((h, w))
    # Gray cotton often touches the image border, so high border percentiles can
    # be foreground rather than paper. Use moderate border stats with caps.
    residual_thr = max(0.6, min(float(np.percentile(residual[border], 90) + 0.25), 2.0))
    texture_thr = max(1.2, min(float(np.percentile(texture[border], 75) + 0.25), 2.0))

    seed = (residual > residual_thr) & (texture > texture_thr)
    softer = (residual > max(0.35, residual_thr * 0.7)) & (texture > max(1.0, texture_thr * 0.75))
    mask = seed | softer

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((13, 13), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)
    mask = remove_small_components(mask, min_area_ratio=0.0004)
    mask = _keep_gray_cotton_components(mask, residual=residual, texture=texture)
    mask = ndimage.binary_dilation(mask, structure=np.ones((5, 5), dtype=bool))
    return mask.astype(bool)


def build_cotton_mask(rgb_balanced: np.ndarray) -> np.ndarray:
    """基于白纸背景差异的前景分割。注意：不填充内部孔洞。"""
    lab = rgb_to_lab(rgb_balanced)
    if _is_gray_scene(lab):
        return build_gray_cotton_mask(lab)

    bg_rgb = estimate_background_rgb(rgb_balanced)
    bg_lab = rgb_to_lab(bg_rgb)

    dist = np.linalg.norm(lab - bg_lab[None, None, :], axis=-1)

    border_lab = rgb_to_lab(_border_pixels(rgb_balanced))
    border_dist = np.linalg.norm(border_lab - bg_lab[None, :], axis=-1)
    thr = max(3.0, float(np.percentile(border_dist, 98) + 1.2))

    mask = dist > thr

    L = lab[..., 0]
    bg_L = float(bg_lab[0])

    # 深色或彩色棉花相对白纸明显更暗
    mask |= (bg_L - L) > 2.5

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))

    # 关键：不要 fill_holes。棉花纤维之间的白色孔洞就是背景。
    mask = remove_small_components(mask, min_area_ratio=0.0008)

    if mask.mean() < 0.01:
        loose = dist > max(2.0, thr * 0.65)
        loose |= (bg_L - L) > 2.0
        loose = ndimage.binary_closing(loose, structure=np.ones((5, 5), dtype=bool))
        loose = remove_small_components(loose, min_area_ratio=0.0005)
        if loose.mean() > mask.mean():
            mask = loose

    return mask.astype(bool)


# ============================================================
# 6. dominant_lab 主色提取
# ============================================================
def _sample_pixels(pixels: np.ndarray, max_pixels: int, seed: int) -> np.ndarray:
    if max_pixels <= 0 or len(pixels) <= max_pixels:
        return pixels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pixels), size=max_pixels, replace=False)
    return pixels[idx]


def select_reliable_lab_pixels(
    lab_pixels: np.ndarray,
    bg_lab: np.ndarray,
    l_trim_low: float = 5.0,
    l_trim_high: float = 95.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    从 mask 内筛出高置信棉花像素。

    改进点：
    1. 先去掉接近白纸背景的像素；
    2. 如果存在足够多“明显比背景暗”的像素，优先使用暗前景像素；
       这对 colorful/8 这类深棕/深灰棉花很关键；
    3. 对高色度彩色棉花，继续使用 chroma 过滤；
    4. 最后再做 L 通道裁剪。
    """
    if len(lab_pixels) == 0:
        return lab_pixels, {
            "valid_pixel_count": 0,
            "background_removed_count": 0,
            "dark_foreground_used": False,
            "dark_foreground_count": 0,
            "chroma_filter_used": False,
            "l_trim_low_value": None,
            "l_trim_high_value": None,
        }

    lab_pixels = np.asarray(lab_pixels, dtype=np.float64)
    bg_lab = np.asarray(bg_lab, dtype=np.float64)

    L = lab_pixels[:, 0]
    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]

    chroma = np.sqrt(a * a + b * b)
    bg_L = float(bg_lab[0])

    # 与背景的 Lab 距离。这里只用于过滤背景，Top-3 仍然用 CIEDE2000。
    dist_to_bg = np.linalg.norm(lab_pixels - bg_lab[None, :], axis=1)
    l_diff = bg_L - L

    # 典型白纸/灰白背景：亮度高、色度低，或者和背景距离很近
    near_white_bg = (
        (dist_to_bg < 8.0)
        | ((L > bg_L - 10.0) & (chroma < 12.0))
        | ((L > 80.0) & (chroma < 8.0))
    )

    # 明显暗于背景的像素。深棕/黑/深蓝棉花主要靠这个条件保留下来。
    dark_foreground = (
        (l_diff > 18.0)
        | ((l_diff > 12.0) & (dist_to_bg > 14.0))
        | ((L < 45.0) & (dist_to_bg > 10.0))
    )

    dark_count = int(np.count_nonzero(dark_foreground))
    dark_min_count = max(200, int(0.02 * len(lab_pixels)))

    dark_foreground_used = False
    chroma_filter_used = False

    if dark_count >= dark_min_count:
        # 对深色棉花，优先只用暗前景像素，避免白纸阴影/灰白孔洞参与投票。
        keep = dark_foreground
        dark_foreground_used = True
    else:
        # 对浅色或普通彩色棉花，先去掉背景。
        keep = ~near_white_bg

        # 如果是明显彩色棉花，再进一步保留高色度像素。
        if np.percentile(chroma, 90) >= 12.0:
            base = chroma[keep] if np.any(keep) else chroma
            chroma_thr = max(8.0, float(np.percentile(base, 45)))
            keep &= chroma >= chroma_thr
            chroma_filter_used = True

    # 如果过滤太狠，退回到“离背景最远”的前 30% 像素。
    if np.count_nonzero(keep) < max(50, int(0.03 * len(lab_pixels))):
        dist_thr = np.percentile(dist_to_bg, 70)
        keep = dist_to_bg >= dist_thr

    candidate = lab_pixels[keep]

    if len(candidate) == 0:
        candidate = lab_pixels

    # 在候选棉花像素中去掉极端阴影/高光
    L2 = candidate[:, 0]
    low_L = np.percentile(L2, l_trim_low)
    high_L = np.percentile(L2, l_trim_high)
    keep2 = (L2 >= low_L) & (L2 <= high_L)

    reliable = candidate[keep2]
    if len(reliable) < max(30, int(0.02 * len(lab_pixels))):
        reliable = candidate
    if len(reliable) == 0:
        reliable = lab_pixels

    info = {
        "valid_pixel_count": int(len(reliable)),
        "background_removed_count": int(np.count_nonzero(near_white_bg)),
        "dark_foreground_used": bool(dark_foreground_used),
        "dark_foreground_count": dark_count,
        "chroma_filter_used": bool(chroma_filter_used),
        "l_trim_low_value": round(float(low_L), 4) if len(candidate) else None,
        "l_trim_high_value": round(float(high_L), 4) if len(candidate) else None,
    }
    return reliable, info


def dominant_lab_by_voting(
    rgb_balanced: np.ndarray,
    mask: np.ndarray,
    l_bin: float = 2.0,
    ab_bin: float = 2.0,
    l_trim_low: float = 5.0,
    l_trim_high: float = 95.0,
    max_vote_pixels: int = 300_000,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Lab 分箱投票主色：
    1. mask 内 RGB -> Lab；
    2. 过滤阴影、高光、低色度透白像素；
    3. 对 Lab 分箱；
    4. 统计出现最多的颜色箱；
    5. 对该颜色箱内真实像素取中位数作为 dominant_lab。
    """
    lab_img = rgb_to_lab(rgb_balanced)
    bg_lab = rgb_to_lab(estimate_background_rgb(rgb_balanced))
    lab_pixels = lab_img[mask]

    fallback_used = False
    if len(lab_pixels) == 0:
        h, w = lab_img.shape[:2]
        lab_pixels = lab_img[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)].reshape(-1, 3)
        fallback_used = True

    reliable, filter_info = select_reliable_lab_pixels(
        lab_pixels,
        bg_lab=bg_lab,
        l_trim_low=l_trim_low,
        l_trim_high=l_trim_high,
    )

    vote_pixels = _sample_pixels(reliable, max_pixels=max_vote_pixels, seed=seed)

    qL = np.floor(vote_pixels[:, 0] / max(l_bin, 1e-6)).astype(np.int32)
    qa = np.floor(vote_pixels[:, 1] / max(ab_bin, 1e-6)).astype(np.int32)
    qb = np.floor(vote_pixels[:, 2] / max(ab_bin, 1e-6)).astype(np.int32)
    bins = np.stack([qL, qa, qb], axis=1)

    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    
    # 先计算可靠像素整体中位数，作为“稳定主色”的参考
    median_lab = np.median(vote_pixels, axis=0)
    
    # 取出现次数最多的 Top-N 个颜色箱
    top_n = min(10, len(unique_bins))
    top_indices = np.argsort(counts)[-top_n:]
    
    best_idx = int(top_indices[-1])
    best_score = float("inf")
    
    for idx in top_indices:
        candidate_bin = unique_bins[idx]
        in_candidate = np.all(bins == candidate_bin[None, :], axis=1)
        candidate_pixels = vote_pixels[in_candidate]
    
        if len(candidate_pixels) == 0:
            continue
        
        candidate_lab = np.median(candidate_pixels, axis=0)
    
        # 和整体中位数的 ΔE00，越小越稳定
        de_to_median = float(delta_e_ciede2000(candidate_lab, median_lab))
    
        # 频数越高越好，所以给高频箱一点奖励
        freq = counts[idx] / max(1, len(vote_pixels))
    
        # 防止选到特别暗的阴影箱
        too_dark_penalty = 0.0
        if candidate_lab[0] < median_lab[0] - 12.0:
            too_dark_penalty = 5.0
    
        score = de_to_median - 2.0 * freq + too_dark_penalty
    
        if score < best_score:
            best_score = score
            best_idx = int(idx)
    
    best_bin = unique_bins[best_idx]
    in_bin = np.all(bins == best_bin[None, :], axis=1)
    dominant_pixels = vote_pixels[in_bin]

    # 如果单个箱太少，合并 Top-3 颜色箱，避免 JPG 噪声导致众数箱过窄。
    if len(dominant_pixels) < 50 and len(unique_bins) >= 3:
        top_idx = np.argsort(counts)[-3:]
        top_bins = unique_bins[top_idx]
        in_top = np.zeros(len(bins), dtype=bool)
        for tb in top_bins:
            in_top |= np.all(bins == tb[None, :], axis=1)
        dominant_pixels = vote_pixels[in_top]

    dominant = np.median(dominant_pixels, axis=0)

    info = {
        "dominant_method": "lab_voting",
        "l_bin": float(l_bin),
        "ab_bin": float(ab_bin),
        "mask_pixel_count": int(len(lab_pixels)),
        "vote_pixel_count": int(len(vote_pixels)),
        "dominant_bin_count": int(counts[best_idx]),
        "dominant_bin_ratio": round(float(counts[best_idx] / max(1, len(vote_pixels))), 6),
        "fallback_used": bool(fallback_used),
        **filter_info,
    }
    return dominant, info


def dominant_lab_by_median(
    rgb_balanced: np.ndarray,
    mask: np.ndarray,
    l_trim_low: float = 5.0,
    l_trim_high: float = 95.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    lab_img = rgb_to_lab(rgb_balanced)
    bg_lab = rgb_to_lab(estimate_background_rgb(rgb_balanced))
    pixels = lab_img[mask]
    fallback_used = False
    if len(pixels) == 0:
        pixels = lab_img.reshape(-1, 3)
        fallback_used = True
    reliable, filter_info = select_reliable_lab_pixels(pixels, bg_lab, l_trim_low, l_trim_high)
    dominant = np.median(reliable, axis=0)
    info = {
        "dominant_method": "lab_median",
        "mask_pixel_count": int(len(pixels)),
        "fallback_used": bool(fallback_used),
        **filter_info,
    }
    return dominant, info


def _pairwise_sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)


def kmeans_lab(pixels: np.ndarray, k: int = 3, max_iter: int = 30, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels, dtype=np.float64)
    n = len(pixels)
    if n == 0:
        raise ValueError("kmeans_lab 收到空像素集")
    k = max(1, min(k, n))

    rng = np.random.default_rng(seed)
    centers = np.empty((k, 3), dtype=np.float64)
    centers[0] = pixels[rng.integers(0, n)]
    for i in range(1, k):
        d2 = np.min(_pairwise_sqdist(pixels, centers[:i]), axis=1)
        prob = d2 / (d2.sum() + 1e-12)
        centers[i] = pixels[rng.choice(n, p=prob)]

    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(max_iter):
        new_labels = np.argmin(_pairwise_sqdist(pixels, centers), axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            members = pixels[labels == ci]
            if len(members) > 0:
                centers[ci] = np.mean(members, axis=0)
            else:
                centers[ci] = pixels[rng.integers(0, n)]
    return centers, labels


def dominant_lab_by_kmeans(
    rgb_balanced: np.ndarray,
    mask: np.ndarray,
    k: int = 3,
    l_trim_low: float = 5.0,
    l_trim_high: float = 95.0,
    max_vote_pixels: int = 300_000,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """保留 KMeans 方案，但选择最接近可靠像素中位数的簇，避免最大簇跳到阴影/透白区。"""
    lab_img = rgb_to_lab(rgb_balanced)
    bg_lab = rgb_to_lab(estimate_background_rgb(rgb_balanced))
    pixels = lab_img[mask]
    fallback_used = False
    if len(pixels) == 0:
        pixels = lab_img.reshape(-1, 3)
        fallback_used = True

    reliable, filter_info = select_reliable_lab_pixels(pixels, bg_lab, l_trim_low, l_trim_high)
    reliable = _sample_pixels(reliable, max_pixels=max_vote_pixels, seed=seed)
    median_lab = np.median(reliable, axis=0)

    if len(reliable) < 20:
        return median_lab, {
            "dominant_method": "kmeans_nearest_median_fallback_median",
            "kmeans_k": 1,
            "fallback_used": bool(fallback_used),
            **filter_info,
        }

    centers, labels = kmeans_lab(reliable, k=k, max_iter=30, seed=seed)
    counts = np.bincount(labels, minlength=len(centers))
    center_de = delta_e_ciede2000(centers, median_lab)
    best_idx = int(np.argmin(center_de))
    dominant = centers[best_idx]

    info = {
        "dominant_method": "kmeans_nearest_median",
        "kmeans_k": int(len(centers)),
        "selected_cluster_ratio": round(float(counts[best_idx] / max(1, len(reliable))), 6),
        "fallback_used": bool(fallback_used),
        **filter_info,
    }
    return dominant, info


def extract_dominant_lab(
    rgb_balanced: np.ndarray,
    mask: np.ndarray,
    method: str,
    kmeans_k: int,
    l_bin: float,
    ab_bin: float,
    l_trim_low: float,
    l_trim_high: float,
    max_vote_pixels: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method == "vote":
        return dominant_lab_by_voting(
            rgb_balanced,
            mask,
            l_bin=l_bin,
            ab_bin=ab_bin,
            l_trim_low=l_trim_low,
            l_trim_high=l_trim_high,
            max_vote_pixels=max_vote_pixels,
            seed=seed,
        )
    if method == "median":
        return dominant_lab_by_median(
            rgb_balanced,
            mask,
            l_trim_low=l_trim_low,
            l_trim_high=l_trim_high,
        )
    if method == "kmeans":
        return dominant_lab_by_kmeans(
            rgb_balanced,
            mask,
            k=kmeans_k,
            l_trim_low=l_trim_low,
            l_trim_high=l_trim_high,
            max_vote_pixels=max_vote_pixels,
            seed=seed,
        )
    raise ValueError(f"未知 dominant_method: {method}")


# ============================================================
# 7. 单图处理、split 汇总、整体处理
# ============================================================
def round_lab(lab: np.ndarray) -> List[float]:
    return [round(float(x), 4) for x in np.asarray(lab, dtype=np.float64)]
def save_debug_images(
    rgb: np.ndarray,
    mask: np.ndarray,
    image_path: Path,
    dataset_root: Path,
    mask_debug_dir: Optional[str],
    overlay_debug_dir: Optional[str],
) -> None:
    try:
        rel = image_path.relative_to(dataset_root)
    except ValueError:
        rel = Path(image_path.name)

    stem_path = rel.with_suffix(".png")

    if mask_debug_dir:
        out_mask = Path(mask_debug_dir) / stem_path
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        mask_img = (mask.astype(np.uint8) * 255)
        Image.fromarray(mask_img, mode="L").save(out_mask)

    if overlay_debug_dir:
        out_overlay = Path(overlay_debug_dir) / stem_path
        out_overlay.parent.mkdir(parents=True, exist_ok=True)

        overlay = rgb.astype(np.float32).copy()
        red = np.array([255, 0, 0], dtype=np.float32)
        alpha = 0.45

        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * red
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        Image.fromarray(overlay, mode="RGB").save(out_overlay)

def process_single_image(
    image_path: Path,
    dataset_root: Path,
    color_db: List[ColorEntry],
    dominant_method: str,
    kmeans_k: int,
    l_bin: float,
    ab_bin: float,
    l_trim_low: float,
    l_trim_high: float,
    max_vote_pixels: int,
    seed: int,
    mask_debug_dir: Optional[str] = None,
    overlay_debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    rgb = read_image_rgb(image_path)
    rgb_balanced = white_balance_with_background(rgb)
    mask = build_cotton_mask(rgb_balanced)
    save_debug_images(
        rgb=rgb,
        mask=mask,
        image_path=image_path,
        dataset_root=dataset_root,
        mask_debug_dir=mask_debug_dir,
        overlay_debug_dir=overlay_debug_dir,
    )
    raw_lab, info = extract_dominant_lab(
        rgb_balanced=rgb_balanced,
        mask=mask,
        method=dominant_method,
        kmeans_k=kmeans_k,
        l_bin=l_bin,
        ab_bin=ab_bin,
        l_trim_low=l_trim_low,
        l_trim_high=l_trim_high,
        max_vote_pixels=max_vote_pixels,
        seed=seed,
    )
    raw_top3 = top_k_matches(raw_lab, color_db, k=3)

    try:
        rel_path = str(image_path.relative_to(dataset_root))
    except ValueError:
        rel_path = str(image_path)

    return {
        "filename": image_path.name,
        "path": rel_path,
        "original_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "processed_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "raw_dominant_lab": round_lab(raw_lab),
        "raw_top3_matches": raw_top3,
        "dominant_lab": round_lab(raw_lab),
        "top3_matches": raw_top3,
        "mask_backend": "color_threshold",
        "mask_area_ratio": round(float(mask.mean()), 6),
        **info,
    }


def pairwise_max_delta_e(labs: Iterable[np.ndarray]) -> float:
    labs = [np.asarray(x, dtype=np.float64) for x in labs]
    if len(labs) <= 1:
        return 0.0
    max_de = 0.0
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            max_de = max(max_de, float(delta_e_ciede2000(labs[i], labs[j])))
    return max_de


def compute_prototype_lab(labs: List[np.ndarray]) -> np.ndarray:
    arr = np.asarray(labs, dtype=np.float64)
    return np.median(arr, axis=0)


def soft_stabilize_lab(raw_lab: np.ndarray, prototype_lab: np.ndarray, alpha: float) -> np.ndarray:
    """
    软稳定化：保留一部分单图颜色，同时向同类 prototype 靠近。

    alpha 表示保留 raw_lab 的比例：
    - alpha=1.0：不稳定化，完全使用单图结果；
    - alpha=0.35：35% 单图结果 + 65% 类别原型色；
    - alpha=0.0：等价硬稳定化，所有图完全一样。
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    raw_lab = np.asarray(raw_lab, dtype=np.float64)
    prototype_lab = np.asarray(prototype_lab, dtype=np.float64)
    return alpha * raw_lab + (1.0 - alpha) * prototype_lab


def summarize_split(
    image_paths: List[Path],
    dataset_root: Path,
    color_db: List[ColorEntry],
    group: str,
    cls_name: str,
    split_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    log(f"开始处理 {group}/{cls_name} [{split_name}]，共 {len(image_paths)} 张")

    images: List[Dict[str, Any]] = []
    for idx, p in enumerate(image_paths, 1):
        log(f"  {group}/{cls_name} [{split_name}] {idx}/{len(image_paths)}: {p.name}")
        item = process_single_image(
            image_path=p,
            dataset_root=dataset_root,
            color_db=color_db,
            dominant_method=args.dominant_method,
            kmeans_k=args.kmeans_k,
            l_bin=args.l_bin,
            ab_bin=args.ab_bin,
            l_trim_low=args.l_trim_low,
            l_trim_high=args.l_trim_high,
            max_vote_pixels=args.max_vote_pixels,
            seed=args.seed + idx,
            mask_debug_dir=args.mask_debug_dir,
            overlay_debug_dir=args.overlay_debug_dir,
        )
        images.append(item)
        log(f"    raw_lab={item['raw_dominant_lab']}, mask={item['mask_area_ratio']}")

    raw_labs = [np.asarray(x["raw_dominant_lab"], dtype=np.float64) for x in images]
    prototype_lab = compute_prototype_lab(raw_labs)
    prototype_top3 = top_k_matches(prototype_lab, color_db, k=3)
    raw_consistency = pairwise_max_delta_e(raw_labs)

    final_labs: List[np.ndarray] = []
    if args.stabilize_outputs:
        if args.stabilization_mode == "hard":
            # 硬稳定化：所有图片直接使用同一个 prototype_lab。
            # 这会让 final_consistency 变成 0，因此只建议作为对照实验。
            for item in images:
                item["dominant_lab"] = round_lab(prototype_lab)
                item["top3_matches"] = prototype_top3
                item["stabilized_delta_e_to_prototype"] = 0.0
                item["stabilization_alpha"] = 0.0
            final_labs = [prototype_lab for _ in images]
        else:
            # 软稳定化：每张图片仍保留一部分 raw_dominant_lab，避免结果过于理想化。
            for item in images:
                raw_lab = np.asarray(item["raw_dominant_lab"], dtype=np.float64)
                final_lab = soft_stabilize_lab(raw_lab, prototype_lab, alpha=args.stabilize_alpha)
                item["dominant_lab"] = round_lab(final_lab)
                item["top3_matches"] = top_k_matches(final_lab, color_db, k=3)
                item["stabilized_delta_e_to_prototype"] = round(
                    float(delta_e_ciede2000(final_lab, prototype_lab)), 4
                )
                item["stabilization_alpha"] = float(args.stabilize_alpha)
                final_labs.append(final_lab)
    else:
        final_labs = [np.asarray(x["dominant_lab"], dtype=np.float64) for x in images]

    consistency = pairwise_max_delta_e(final_labs)

    log(
        f"完成 {group}/{cls_name} [{split_name}]: "
        f"raw_consistency={raw_consistency:.4f}, final_consistency={consistency:.4f}, "
        f"prototype_lab={round_lab(prototype_lab)}"
    )

    return {
        "images": images,
        "prototype_lab": round_lab(prototype_lab),
        "prototype_top3": prototype_top3,
        "raw_consistency_score": round(float(raw_consistency), 4),
        "consistency_score": round(float(consistency), 4),
    }


def process_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    color_db = load_color_database(args.color_db_path)
    classes = discover_classes(dataset_root)

    result: Dict[str, Any] = {
        "meta": {
            "dataset_root": str(dataset_root.resolve()),
            "color_db_path": str(Path(args.color_db_path).resolve()),
            "seed": args.seed,
            "train_n": args.train_n,
            "only_group": args.only_group,
            "only_class": args.only_class,
            "stabilize_outputs": bool(args.stabilize_outputs),
            "stabilization_mode": args.stabilization_mode if args.stabilize_outputs else "none",
            "stabilize_alpha": float(args.stabilize_alpha) if args.stabilize_outputs else None,
            "segmentation_backend": "color_threshold",
            "method": {
                "segmentation": "white paper background threshold in Lab space + morphology",
                "white_balance": "white paper background normalization",
                "dominant_color": (
                    "Lab voting over quantized foreground pixels" if args.dominant_method == "vote" else args.dominant_method
                ),
                "matching_metric": "CIEDE2000",
                "stabilization": (
                    (
                        f"enabled: {args.stabilization_mode}; "
                        f"alpha={args.stabilize_alpha}; "
                        "soft mode uses final_lab = alpha * raw_lab + (1-alpha) * prototype_lab; "
                        "hard mode uses prototype_lab directly"
                    )
                    if args.stabilize_outputs
                    else "disabled: final dominant_lab equals raw_dominant_lab"
                ),
            },
        },
        "datasets": {},
        "overall": {},
    }

    train_cons: List[float] = []
    test_cons: List[float] = []
    raw_train_cons: List[float] = []
    raw_test_cons: List[float] = []

    for group, cls_map in classes.items():
        if args.only_group and group != args.only_group:
            continue
        result["datasets"].setdefault(group, {})

        for cls_name, img_paths in cls_map.items():
            if args.only_class and cls_name != str(args.only_class):
                continue

            log(f"========== 类别 {group}/{cls_name}，图片数 {len(img_paths)} ==========")
            train_imgs, test_imgs = split_image_paths(img_paths, seed=args.seed, train_n=args.train_n)

            train_summary = summarize_split(train_imgs, dataset_root, color_db, group, cls_name, "train", args)
            test_summary = summarize_split(test_imgs, dataset_root, color_db, group, cls_name, "test", args)

            proto_gap = float(
                delta_e_ciede2000(
                    np.asarray(train_summary["prototype_lab"], dtype=np.float64),
                    np.asarray(test_summary["prototype_lab"], dtype=np.float64),
                )
            )

            result["datasets"][group][cls_name] = {
                "train": train_summary,
                "test": test_summary,
                "prototype_gap_train_vs_test": round(proto_gap, 4),
            }

            train_cons.append(float(train_summary["consistency_score"]))
            test_cons.append(float(test_summary["consistency_score"]))
            raw_train_cons.append(float(train_summary["raw_consistency_score"]))
            raw_test_cons.append(float(test_summary["raw_consistency_score"]))

    def mean_or_none(xs: List[float]) -> Optional[float]:
        return round(float(np.mean(xs)), 4) if xs else None

    def max_or_none(xs: List[float]) -> Optional[float]:
        return round(float(np.max(xs)), 4) if xs else None

    result["overall"] = {
        "mean_train_consistency": mean_or_none(train_cons),
        "mean_test_consistency": mean_or_none(test_cons),
        "max_train_consistency": max_or_none(train_cons),
        "max_test_consistency": max_or_none(test_cons),
        "mean_raw_train_consistency": mean_or_none(raw_train_cons),
        "mean_raw_test_consistency": mean_or_none(raw_test_cons),
        "max_raw_train_consistency": max_or_none(raw_train_cons),
        "max_raw_test_consistency": max_or_none(raw_test_cons),
    }
    return result


# ============================================================
# 8. CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cotton color recognition with Lab voting and CIEDE2000 Top-3 matching")
    parser.add_argument("--dataset_root", type=str, default="cotton_image", help="数据集根目录，包含 gray/colorful")
    parser.add_argument("--color_db_path", type=str, default="color_dataset.json", help="颜色数据库 JSON")
    parser.add_argument("--output", type=str, default="outputs/results.json", help="输出 JSON 文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train_n", type=int, default=6, help="每类训练图片数量")
    parser.add_argument("--only_group", type=str, default=None, choices=[None, "gray", "colorful"], help="只处理 gray 或 colorful")
    parser.add_argument("--only_class", type=str, default=None, help="只处理某个类别目录名，例如 7")

    parser.add_argument("--dominant_method", type=str, default="vote", choices=["vote", "median", "kmeans"], help="主色提取方法")
    parser.add_argument("--stabilize_outputs", action="store_true", help="启用最终输出稳定化；默认采用 soft，不会把 consistency 直接压成 0")
    parser.add_argument("--stabilization_mode", type=str, default="soft", choices=["soft", "hard"], help="soft=软稳定化；hard=完全替换为 prototype_lab，会导致 final_consistency=0")
    parser.add_argument("--stabilize_alpha", type=float, default=0.35, help="软稳定化中 raw_lab 的保留比例，final=alpha*raw+(1-alpha)*prototype")

    parser.add_argument("--l_bin", type=float, default=2.0, help="Lab 投票中 L 通道分箱宽度")
    parser.add_argument("--ab_bin", type=float, default=2.0, help="Lab 投票中 a/b 通道分箱宽度")
    parser.add_argument("--l_trim_low", type=float, default=5.0, help="去掉 L 通道最低百分位")
    parser.add_argument("--l_trim_high", type=float, default=95.0, help="去掉 L 通道最高百分位")
    parser.add_argument("--max_vote_pixels", type=int, default=300000, help="最多用于投票/聚类的像素数，<=0 表示不采样")
    parser.add_argument("--kmeans_k", type=int, default=3, help="dominant_method=kmeans 时的聚类数")
    parser.add_argument("--mask_debug_dir", type=str, default=None, help="保存 mask 调试图的目录")
    parser.add_argument("--overlay_debug_dir", type=str, default=None, help="保存 overlay 调试图的目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.stabilize_alpha = float(np.clip(args.stabilize_alpha, 0.0, 1.0))

    log("========== 棉花颜色识别开始 ==========")
    log(f"参数: {vars(args)}")
    result = process_dataset(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log(f"结果已保存: {out_path.resolve()}")
    log(f"overall: {json.dumps(result['overall'], ensure_ascii=False)}")
    log("========== 棉花颜色识别完成 ==========")


if __name__ == "__main__":
    main()
