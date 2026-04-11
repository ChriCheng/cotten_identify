from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np


# =========================
# Color space conversion
# =========================

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(np.clip(rgb, 0.0, None), 1.0 / 2.4) - 0.055)


def rgb01_to_xyz(rgb01: np.ndarray) -> np.ndarray:
    rgb_lin = srgb_to_linear(np.asarray(rgb01, dtype=np.float64))
    # D65, sRGB
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    flat = rgb_lin.reshape(-1, 3)
    xyz = flat @ M.T
    return xyz.reshape(rgb_lin.shape)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    delta = 6 / 29

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype.kind in "ui":
        rgb01 = rgb.astype(np.float64) / 255.0
    else:
        rgb01 = rgb.astype(np.float64)
    return xyz_to_lab(rgb01_to_xyz(rgb01))


def hex_to_rgb01(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0
    return rgb


# =========================
# Delta E
# =========================

def delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    return np.linalg.norm(lab1 - lab2, axis=-1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 implementation.
    Inputs can be shape (..., 3) and broadcast against each other.
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = [lab1[..., i] for i in range(3)]
    L2, a2, b2 = [lab2[..., i] for i in range(3)]

    avg_L = 0.5 * (L1 + L2)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = 0.5 * (C1 + C2)

    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7 + 1e-12)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    avg_Cp = 0.5 * (C1p + C2p)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((C1p * C2p) == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    avg_Lp = 0.5 * (L1 + L2)

    sum_h = h1p + h2p
    abs_diff_h = np.abs(h1p - h2p)
    avg_hp = np.where(
        (C1p * C2p) == 0,
        sum_h,
        np.where(
            abs_diff_h <= 180,
            0.5 * sum_h,
            np.where(sum_h < 360, 0.5 * (sum_h + 360), 0.5 * (sum_h - 360)),
        ),
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30))
        + 0.24 * np.cos(np.radians(2 * avg_hp))
        + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
        - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
    )

    delta_ro = 30 * np.exp(-(((avg_hp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7 + 1e-12))
    Sl = 1 + (0.015 * (avg_Lp - 50) ** 2) / np.sqrt(20 + (avg_Lp - 50) ** 2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -np.sin(np.radians(2 * delta_ro)) * Rc

    dE = np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return dE


# =========================
# Dataset helpers
# =========================

def load_color_database(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("color_dataset.json should be a list of color entries")

    cleaned = []
    for item in data:
        entry = dict(item)
        if "lab_D65" in entry and entry["lab_D65"] is not None:
            lab = np.asarray(entry["lab_D65"], dtype=np.float64)
        elif "hex" in entry and entry["hex"]:
            lab = rgb_to_lab(hex_to_rgb01(entry["hex"]))
        elif "rgb" in entry and entry["rgb"]:
            rgb = np.asarray(entry["rgb"], dtype=np.float64)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            lab = rgb_to_lab(rgb)
        else:
            raise ValueError(f"Color entry missing lab_D65/hex/rgb: {entry}")
        entry["lab_D65"] = np.asarray(lab, dtype=np.float64).tolist()
        cleaned.append(entry)
    return cleaned


def top_k_matches(query_lab: Iterable[float], color_db: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    q = np.asarray(query_lab, dtype=np.float64)
    labs = np.asarray([c["lab_D65"] for c in color_db], dtype=np.float64)
    dists = delta_e2000(labs, q)
    order = np.argsort(dists)[:k]
    result = []
    for idx in order:
        c = color_db[int(idx)]
        result.append(
            {
                "name": c.get("name", f"color_{idx}"),
                "hex": c.get("hex", ""),
                "delta_e": round(float(dists[idx]), 4),
            }
        )
    return result


def pairwise_max_delta_e(labs: Iterable[Iterable[float]]) -> float:
    arr = np.asarray(list(labs), dtype=np.float64)
    if len(arr) <= 1:
        return 0.0
    max_de = 0.0
    for i in range(len(arr)):
        d = delta_e2000(arr[i + 1 :], arr[i])
        if d.size:
            max_de = max(max_de, float(np.max(d)))
    return max_de


def robust_group_prototype(labs: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(labs), dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("No Lab values to aggregate")
    if len(arr) == 1:
        return arr[0]
    # Pick medoid under Delta E 2000
    totals = []
    for i in range(len(arr)):
        totals.append(float(np.sum(delta_e2000(arr, arr[i]))))
    return arr[int(np.argmin(totals))]
