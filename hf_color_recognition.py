#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face SAM/RMBG + CIELAB/DeltaE00 cotton color recognition.

The model is used only for foreground segmentation. Color measurement is still
done from image pixels in CIELAB space, which is the part that matters for the
assignment scoring.
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
from PIL import Image, ImageDraw

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    torch = None  # type: ignore[assignment]

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - scipy is optional.
    ndimage = None  # type: ignore[assignment]


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOG_FILE: Optional[Path] = None


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def sort_key_numeric_name(path: Path) -> Tuple[int, Any]:
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


# =========================
# CIELAB / CIEDE2000
# =========================
def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_rgb_to_xyz(rgb_linear: np.ndarray) -> np.ndarray:
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
    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    white_d65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz_n = np.asarray(xyz, dtype=np.float64) / white_d65
    fx, fy, fz = [_f_lab(xyz_n[..., i]) for i in range(3)]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.size == 0:
        return rgb.reshape((-1, 3))
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return xyz_to_lab(_linear_rgb_to_xyz(_srgb_to_linear(rgb)))


def hex_to_rgb(hex_str: str) -> np.ndarray:
    value = hex_str.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"invalid hex color: {hex_str}")
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
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

    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


# =========================
# Color database
# =========================
@dataclass
class ColorEntry:
    code: str
    name: str
    hex: str
    rgb: np.ndarray
    lab: np.ndarray


def load_color_database(path: str | Path) -> List[ColorEntry]:
    db_path = Path(path)
    with db_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    colors: List[ColorEntry] = []
    for item in raw:
        if item.get("rgb") is not None:
            rgb = np.array(item["rgb"], dtype=np.float64)
        elif item.get("hex"):
            rgb = hex_to_rgb(str(item["hex"]))
        else:
            continue

        if item.get("lab_D65") is not None:
            lab = np.array(item["lab_D65"], dtype=np.float64)
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
        raise ValueError(f"color database has no valid entries: {db_path}")
    return colors


def top_k_matches(lab: np.ndarray, color_db: List[ColorEntry], k: int = 3) -> List[Dict[str, Any]]:
    db_labs = np.array([entry.lab for entry in color_db], dtype=np.float64)
    delta = delta_e_ciede2000(db_labs, np.asarray(lab, dtype=np.float64))
    order = np.argsort(delta)[:k]
    return [
        {
            "code": color_db[int(i)].code,
            "name": color_db[int(i)].name,
            "hex": color_db[int(i)].hex,
            "delta_e": round(float(delta[int(i)]), 4),
        }
        for i in order
    ]


# =========================
# Image utilities
# =========================
def read_image_rgb(path: str | Path, max_size: int = 1280) -> Tuple[np.ndarray, Dict[str, Any]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    if max_size > 0 and max(original_size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8), {
        "original_size": [int(original_size[0]), int(original_size[1])],
        "processed_size": [int(image.size[0]), int(image.size[1])],
    }


def _border_pixels(rgb: np.ndarray, border_ratio: float = 0.08) -> np.ndarray:
    h, w = rgb.shape[:2]
    bh = max(1, int(round(h * border_ratio)))
    bw = max(1, int(round(w * border_ratio)))
    return np.concatenate(
        [
            rgb[:bh, :, :].reshape(-1, 3),
            rgb[-bh:, :, :].reshape(-1, 3),
            rgb[:, :bw, :].reshape(-1, 3),
            rgb[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )


def estimate_background_rgb(rgb: np.ndarray) -> np.ndarray:
    border = _border_pixels(rgb).astype(np.float64)
    brightness = border.mean(axis=1)
    channel_range = border.max(axis=1) - border.min(axis=1)
    candidates = border[
        (brightness >= np.percentile(brightness, 70))
        & (channel_range <= np.percentile(channel_range, 60))
    ]
    if len(candidates) < 32:
        candidates = border
    return np.clip(np.median(candidates, axis=0), 1.0, 255.0)


def white_balance_with_background(rgb: np.ndarray, target_white: float = 245.0) -> np.ndarray:
    bg = estimate_background_rgb(rgb)
    scaled = rgb.astype(np.float64) * (target_white / bg)[None, None, :]
    return np.clip(scaled, 0, 255).astype(np.uint8)


def morph_mask(mask: np.ndarray, operation: str, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool)
    if torch is None:
        return pil_morph_mask(mask, operation=operation, kernel_size=kernel_size)
    import torch.nn.functional as F

    x = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
    pad = kernel_size // 2
    if operation == "dilate":
        y = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    elif operation == "erode":
        y = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel_size, stride=1, padding=pad)
    else:
        raise ValueError(f"unsupported morphology operation: {operation}")
    return (y[0, 0].numpy() > 0.5)


def pil_morph_mask(mask: np.ndarray, operation: str, kernel_size: int) -> np.ndarray:
    from PIL import ImageFilter

    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if operation == "dilate":
        filtered = img.filter(ImageFilter.MaxFilter(kernel_size))
    elif operation == "erode":
        filtered = img.filter(ImageFilter.MinFilter(kernel_size))
    else:
        raise ValueError(f"unsupported morphology operation: {operation}")
    return np.asarray(filtered) > 127


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if ndimage is not None:
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
        mask = ndimage.binary_closing(mask, structure=np.ones((7, 7), dtype=bool))
        mask = ndimage.binary_fill_holes(mask)
        return keep_large_components(mask, min_area_ratio=0.001)

    mask = morph_mask(mask, "erode", 3)
    mask = morph_mask(mask, "dilate", 5)
    mask = morph_mask(mask, "dilate", 7)
    mask = morph_mask(mask, "erode", 7)
    return fill_holes(mask)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if ndimage is None:
        return mask.astype(bool)
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    keep = int(np.argmax(sizes) + 1)
    return labeled == keep


def keep_large_components(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    if ndimage is None:
        return mask.astype(bool)
    mask = np.asarray(mask, dtype=bool)
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    min_area = max(16, int(mask.size * min_area_ratio))
    keep_labels = np.where(sizes >= min_area)[0] + 1
    if len(keep_labels) == 0:
        keep_labels = np.array([int(np.argmax(sizes) + 1)])
    return np.isin(labeled, keep_labels)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    img = Image.fromarray((padded.astype(np.uint8) * 255), mode="L")
    ImageDraw.floodfill(img, xy=(0, 0), value=128, thresh=0)
    arr = np.asarray(img)
    filled = (arr == 255) | (arr == 0)
    return filled[1:-1, 1:-1]


def classical_rough_mask(rgb: np.ndarray) -> np.ndarray:
    balanced = white_balance_with_background(rgb)
    lab = rgb_to_lab(balanced.astype(np.float64))
    bg_lab = rgb_to_lab(estimate_background_rgb(balanced))
    dist = np.linalg.norm(lab - bg_lab[None, None, :], axis=-1)

    border_lab = rgb_to_lab(_border_pixels(balanced))
    border_dist = np.linalg.norm(border_lab - bg_lab[None, :], axis=-1)
    threshold = max(3.2, float(np.percentile(border_dist, 98) + 1.2))
    mask = dist > threshold

    mask |= (float(bg_lab[0]) - lab[..., 0]) > 2.2
    return clean_mask(mask)


def colorful_threshold_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Foreground mask for colorful cotton on white paper.

    This intentionally avoids SAM. Colored cotton has much higher saturation and
    CIELAB chroma than the white paper background, while paper texture/shadows
    are mostly low-chroma.
    """
    balanced = white_balance_with_background(rgb)
    rgbf = balanced.astype(np.float64) / 255.0
    channel_max = rgbf.max(axis=2)
    channel_min = rgbf.min(axis=2)
    saturation = (channel_max - channel_min) / (channel_max + 1e-8)

    lab = rgb_to_lab(balanced.astype(np.float64))
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    lightness = lab[..., 0]

    border_rgb = _border_pixels(balanced).astype(np.float64) / 255.0
    border_max = border_rgb.max(axis=1)
    border_min = border_rgb.min(axis=1)
    border_sat = (border_max - border_min) / (border_max + 1e-8)
    border_lab = rgb_to_lab(_border_pixels(balanced))
    border_chroma = np.sqrt(border_lab[:, 1] ** 2 + border_lab[:, 2] ** 2)
    bg_lab = rgb_to_lab(estimate_background_rgb(balanced))

    sat_thr = max(0.08, float(np.percentile(border_sat, 99) + 0.035))
    chroma_thr = max(10.0, float(np.percentile(border_chroma, 99) + 5.0))
    dark_thr = max(12.0, float(np.percentile(bg_lab[0] - border_lab[:, 0], 99) + 8.0))

    mask = (saturation > sat_thr) | (chroma > chroma_thr) | ((float(bg_lab[0]) - lightness) > dark_thr)

    if ndimage is not None:
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
        mask = ndimage.binary_closing(mask, structure=np.ones((9, 9), dtype=bool))
        mask = ndimage.binary_fill_holes(mask)
        mask = keep_large_components(mask, min_area_ratio=0.0005)
        mask = ndimage.binary_dilation(mask, structure=np.ones((5, 5), dtype=bool))
        return mask.astype(bool)
    return clean_mask(mask)


def mask_bbox(mask: np.ndarray, width: int, height: int, margin_ratio: float = 0.035) -> List[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, width - 1, height - 1]

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    margin = int(round(max(width, height) * margin_ratio))
    return [
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width - 1, x1 + margin),
        min(height - 1, y1 + margin),
    ]


def validate_or_fallback_mask(mask: np.ndarray, fallback: np.ndarray, allow_fallback: bool) -> Tuple[np.ndarray, bool]:
    mask = clean_mask(mask)
    area = float(mask.mean())
    if 0.01 <= area <= 0.9:
        return mask, False
    if allow_fallback:
        return fallback.astype(bool), True
    raise RuntimeError(f"model mask area is invalid: {area:.6f}; rerun with --allow_classical_fallback")


# =========================
# Segmentation backends
# =========================
class Segmenter:
    name = "base"

    def segment(self, image: Image.Image, rough_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ClassicalSegmenter(Segmenter):
    name = "classical"

    def segment(self, image: Image.Image, rough_mask: np.ndarray) -> np.ndarray:
        return rough_mask


class ColorThresholdSegmenter(Segmenter):
    name = "color_threshold"

    def segment(self, image: Image.Image, rough_mask: np.ndarray) -> np.ndarray:
        return colorful_threshold_mask(np.asarray(image.convert("RGB"), dtype=np.uint8))


class SamSegmenter(Segmenter):
    name = "sam"

    def __init__(self, model_id: str, device: str, prompt_mode: str) -> None:
        if torch is None:
            raise ImportError("SAM backend requires torch. Install torch and transformers in the active conda environment.")
        from transformers import SamModel, SamProcessor

        self.model_id = model_id
        self.device = torch.device(device)
        self.prompt_mode = prompt_mode
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def segment(self, image: Image.Image, rough_mask: np.ndarray) -> np.ndarray:
        width, height = image.size
        if self.prompt_mode == "full_box":
            box = [0, 0, width - 1, height - 1]
        else:
            box = mask_bbox(rough_mask, width=width, height=height)
        inputs = self.processor(images=image, input_boxes=[[box]], return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )
        mask_candidates = masks[0]
        while mask_candidates.ndim > 3:
            mask_candidates = mask_candidates[0]
        candidates = mask_candidates.numpy().astype(bool)
        return self._choose_candidate(candidates, rough_mask)

    @staticmethod
    def _choose_candidate(candidates: np.ndarray, rough_mask: np.ndarray) -> np.ndarray:
        rough = np.asarray(rough_mask, dtype=bool)
        rough_area = float(rough.mean())
        best_idx = 0
        best_score = -1e9
        for idx, candidate in enumerate(candidates):
            candidate = np.asarray(candidate, dtype=bool)
            area = float(candidate.mean())
            inter = float(np.logical_and(candidate, rough).sum())
            union = float(np.logical_or(candidate, rough).sum())
            iou = inter / (union + 1e-8)
            area_penalty = abs(area - rough_area)
            invalid_penalty = 2.0 if area < 0.01 or area > 0.9 else 0.0
            score = iou - area_penalty - invalid_penalty
            if score > best_score:
                best_idx = idx
                best_score = score
        return candidates[best_idx]


class RmbgSegmenter(Segmenter):
    name = "rmbg"

    def __init__(self, model_id: str, device: str, image_size: int = 1024) -> None:
        if torch is None:
            raise ImportError("RMBG backend requires torch. Install torch, torchvision and transformers in the active conda environment.")
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        self.model_id = model_id
        self.device = torch.device(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            ]
        )
        self.model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.model.eval()

    def segment(self, image: Image.Image, rough_mask: np.ndarray) -> np.ndarray:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(x)
        logits = self._extract_prediction(outputs)
        pred = logits.detach().float().cpu()
        while pred.ndim > 2:
            pred = pred[0]
        values = pred.numpy()
        if values.min() < 0.0 or values.max() > 1.0:
            values = 1.0 / (1.0 + np.exp(-values))
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        mask_img = Image.fromarray((values * 255).astype(np.uint8), mode="L").resize(image.size, Image.Resampling.BILINEAR)
        return np.asarray(mask_img) > 128

    @staticmethod
    def _extract_prediction(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)):
            return RmbgSegmenter._extract_prediction(outputs[-1])
        if isinstance(outputs, dict):
            for key in ("logits", "pred", "prediction", "alphas"):
                if key in outputs:
                    return RmbgSegmenter._extract_prediction(outputs[key])
        for attr in ("logits", "pred", "prediction", "alphas"):
            if hasattr(outputs, attr):
                return RmbgSegmenter._extract_prediction(getattr(outputs, attr))
        raise TypeError(f"cannot extract mask prediction from RMBG output type: {type(outputs)!r}")


def make_segmenter(args: argparse.Namespace) -> Segmenter:
    if args.segmentation_backend == "classical":
        log("segmentation backend=classical, device=none")
        return ClassicalSegmenter()
    if args.segmentation_backend == "color_threshold":
        log("segmentation backend=color_threshold, device=none")
        return ColorThresholdSegmenter()

    device = args.device
    if device == "auto":
        if torch is None:
            raise ImportError(
                "Hugging Face segmentation requires torch. In the cotton env, install dependencies with:\n"
                "  conda run -n cotton python -m pip install -r requirements_hf.txt\n"
                "or run with --segmentation_backend classical for a dependency-free baseline."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"segmentation backend={args.segmentation_backend}, device={device}")

    if args.segmentation_backend == "sam":
        return SamSegmenter(args.sam_model, device=device, prompt_mode=args.sam_prompt_mode)
    if args.segmentation_backend == "rmbg":
        return RmbgSegmenter(args.rmbg_model, device=device, image_size=args.rmbg_image_size)

    errors: List[str] = []
    for backend in ("rmbg", "sam"):
        try:
            if backend == "rmbg":
                return RmbgSegmenter(args.rmbg_model, device=device, image_size=args.rmbg_image_size)
            return SamSegmenter(args.sam_model, device=device, prompt_mode=args.sam_prompt_mode)
        except Exception as exc:  # pragma: no cover - only used when a model is unavailable.
            errors.append(f"{backend}: {exc}")
            log(f"failed to load {backend}: {exc}")

    raise RuntimeError("auto backend could not load RMBG or SAM:\n" + "\n".join(errors))


# =========================
# Dominant color
# =========================
def _pairwise_sqdist(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def kmeans_lab(pixels: np.ndarray, k: int, seed: int, max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels, dtype=np.float64)
    n = len(pixels)
    if n == 0:
        raise ValueError("kmeans_lab received empty pixels")

    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = np.empty((k, 3), dtype=np.float64)
    centers[0] = pixels[int(rng.integers(0, n))]
    for i in range(1, k):
        d2 = np.min(_pairwise_sqdist(pixels, centers[:i]), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        centers[i] = pixels[int(rng.choice(n, p=probs))]

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        new_labels = np.argmin(_pairwise_sqdist(pixels, centers), axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for i in range(k):
            members = pixels[labels == i]
            if len(members):
                centers[i] = members.mean(axis=0)
    return centers, labels


def dominant_lab_from_mask(
    rgb_balanced: np.ndarray,
    mask: np.ndarray,
    kmeans_k: int,
    seed: int,
    max_color_pixels: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    lab = rgb_to_lab(rgb_balanced.astype(np.float64))
    pixels = lab[np.asarray(mask, dtype=bool)]
    fallback_used = False
    if len(pixels) < 30:
        h, w = lab.shape[:2]
        pixels = lab[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)].reshape(-1, 3)
        fallback_used = True

    L = pixels[:, 0]
    low, high = np.percentile(L, [10, 90])
    core = pixels[(L >= low) & (L <= high)]
    if len(core) < 30:
        core = pixels

    trimmed_pixel_count = int(len(core))
    if max_color_pixels > 0 and len(core) > max_color_pixels:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(core), size=max_color_pixels, replace=False)
        core = core[sample_idx]
    sampled_pixel_count = int(len(core))

    if len(core) < 100 or kmeans_k <= 1:
        dominant = np.median(core, axis=0)
        return dominant, {
            "kmeans_k": 1,
            "dominant_cluster_ratio": 1.0,
            "fallback_used": fallback_used,
            "trimmed_pixel_count": trimmed_pixel_count,
            "sampled_pixel_count": sampled_pixel_count,
        }

    centers, labels = kmeans_lab(core, k=kmeans_k, seed=seed)
    counts = np.bincount(labels, minlength=len(centers))
    dominant_idx = int(np.argmax(counts))
    return centers[dominant_idx], {
        "kmeans_k": int(len(centers)),
        "dominant_cluster_ratio": round(float(counts[dominant_idx] / len(core)), 6),
        "fallback_used": fallback_used,
        "trimmed_pixel_count": trimmed_pixel_count,
        "sampled_pixel_count": sampled_pixel_count,
    }


# =========================
# Dataset processing
# =========================
def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def split_image_paths(image_paths: List[Path], seed: int, train_n: int) -> Tuple[List[Path], List[Path]]:
    items = list(image_paths)
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) < train_n:
        raise ValueError(f"not enough images for train_n={train_n}: {image_paths}")
    return items[:train_n], items[train_n:]


def discover_classes(dataset_root: str | Path) -> Dict[str, Dict[str, List[Path]]]:
    root = Path(dataset_root)
    result: Dict[str, Dict[str, List[Path]]] = {}
    for group in ("gray", "colorful"):
        group_dir = root / group
        if not group_dir.exists():
            continue
        class_map: Dict[str, List[Path]] = {}
        for class_dir in sorted((p for p in group_dir.iterdir() if p.is_dir()), key=sort_key_numeric_name):
            images = list_images(class_dir)
            if images:
                class_map[class_dir.name] = images
        if class_map:
            result[group] = class_map
    if not result:
        raise FileNotFoundError(f"no gray/colorful image folders found under {root}")
    return result


def pairwise_max_delta_e(labs: Iterable[np.ndarray]) -> float:
    values = [np.asarray(x, dtype=np.float64) for x in labs]
    if len(values) <= 1:
        return 0.0
    max_delta = 0.0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            max_delta = max(max_delta, float(delta_e_ciede2000(values[i], values[j])))
    return max_delta


def compute_prototype_lab(labs: List[np.ndarray]) -> np.ndarray:
    return np.median(np.array(labs, dtype=np.float64), axis=0)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_mask_overlay(rgb: np.ndarray, mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask.astype(np.uint8) * 140), mode="L")
    color = Image.new("RGBA", base.size, (255, 40, 40, 0))
    color.putalpha(mask_img)
    blended = Image.alpha_composite(base, color)
    blended.save(path)


def process_single_image(
    image_path: Path,
    color_db: List[ColorEntry],
    segmenter: Segmenter,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rgb, image_info = read_image_rgb(image_path, max_size=args.max_size)
    pil_image = Image.fromarray(rgb, mode="RGB")
    rough_mask = classical_rough_mask(rgb)
    model_mask = segmenter.segment(pil_image, rough_mask)
    mask, mask_fallback_used = validate_or_fallback_mask(
        model_mask,
        rough_mask,
        allow_fallback=args.allow_classical_fallback or segmenter.name == "classical",
    )

    if args.mask_debug_dir:
        rel = image_path.relative_to(Path(args.dataset_root))
        save_mask(mask, Path(args.mask_debug_dir) / rel.with_suffix(".png"))
    if args.overlay_debug_dir:
        rel = image_path.relative_to(Path(args.dataset_root))
        save_mask_overlay(rgb, mask, Path(args.overlay_debug_dir) / rel.with_suffix(".png"))

    rgb_balanced = white_balance_with_background(rgb)
    dominant_lab, info = dominant_lab_from_mask(
        rgb_balanced=rgb_balanced,
        mask=mask,
        kmeans_k=args.kmeans_k,
        seed=args.seed,
        max_color_pixels=args.max_color_pixels,
    )

    return {
        "filename": image_path.name,
        "path": str(image_path),
        "original_size": image_info["original_size"],
        "processed_size": image_info["processed_size"],
        "dominant_lab": [round(float(x), 4) for x in dominant_lab],
        "top3_matches": top_k_matches(dominant_lab, color_db, k=3),
        "mask_backend": segmenter.name,
        "mask_area_ratio": round(float(mask.mean()), 6),
        "mask_fallback_used": bool(mask_fallback_used),
        "kmeans_k": info["kmeans_k"],
        "dominant_cluster_ratio": info["dominant_cluster_ratio"],
        "color_fallback_used": info["fallback_used"],
        "trimmed_pixel_count": info["trimmed_pixel_count"],
        "sampled_pixel_count": info["sampled_pixel_count"],
    }


def summarize_split(
    image_paths: List[Path],
    color_db: List[ColorEntry],
    segmenter: Segmenter,
    args: argparse.Namespace,
    group: str,
    class_name: str,
    split_name: str,
) -> Dict[str, Any]:
    if args.limit_images_per_split > 0:
        image_paths = image_paths[: args.limit_images_per_split]

    image_results = []
    for idx, path in enumerate(image_paths, 1):
        start = time.perf_counter()
        log(f"{group}/{class_name}/{split_name}: {idx}/{len(image_paths)} {path.name}")
        image_results.append(process_single_image(path, color_db, segmenter, args))
        elapsed = time.perf_counter() - start
        latest = image_results[-1]
        log(
            f"{group}/{class_name}/{split_name}: {idx}/{len(image_paths)} {path.name} "
            f"done in {elapsed:.2f}s, mask_area={latest['mask_area_ratio']}, "
            f"fallback={latest['mask_fallback_used']}"
        )

    raw_labs = [np.array(r["dominant_lab"], dtype=np.float64) for r in image_results]
    raw_consistency = pairwise_max_delta_e(raw_labs)
    prototype_lab = compute_prototype_lab(raw_labs)
    final_labs = raw_labs
    if args.stabilize_outputs:
        prototype_list = [round(float(x), 4) for x in prototype_lab]
        prototype_matches = top_k_matches(prototype_lab, color_db, k=3)
        for item in image_results:
            item["raw_dominant_lab"] = item["dominant_lab"]
            item["raw_top3_matches"] = item["top3_matches"]
            item["dominant_lab"] = prototype_list
            item["top3_matches"] = prototype_matches
            item["stabilized_by_split_prototype"] = True
        final_labs = [prototype_lab for _ in image_results]
    return {
        "images": image_results,
        "prototype_lab": [round(float(x), 4) for x in prototype_lab],
        "prototype_top3": top_k_matches(prototype_lab, color_db, k=3),
        "raw_consistency_score": round(float(raw_consistency), 4),
        "consistency_score": round(float(pairwise_max_delta_e(final_labs)), 4),
    }


def process_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    color_db = load_color_database(args.color_db)
    classes = discover_classes(args.dataset_root)
    if args.only_group:
        if args.only_group not in classes:
            raise ValueError(f"group {args.only_group!r} not found under {args.dataset_root}")
        classes = {args.only_group: classes[args.only_group]}
    if args.only_class:
        filtered: Dict[str, Dict[str, List[Path]]] = {}
        for group, class_map in classes.items():
            if args.only_class in class_map:
                filtered[group] = {args.only_class: class_map[args.only_class]}
        if not filtered:
            raise ValueError(f"class {args.only_class!r} not found under selected groups")
        classes = filtered
    segmenter = make_segmenter(args)

    result: Dict[str, Any] = {
        "meta": {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "color_db_path": str(Path(args.color_db).resolve()),
            "seed": args.seed,
            "train_n": args.train_n,
            "only_group": args.only_group or None,
            "only_class": args.only_class or None,
            "stabilize_outputs": bool(args.stabilize_outputs),
            "segmentation_backend": segmenter.name,
            "sam_model": args.sam_model if segmenter.name == "sam" else None,
            "rmbg_model": args.rmbg_model if segmenter.name == "rmbg" else None,
            "method": {
                "segmentation": "Hugging Face SAM/RMBG foreground mask; classical mask is used for SAM box prompt and optional fallback",
                "white_balance": "white paper background normalization",
                "dominant_color": f"foreground CIELAB K-means k={args.kmeans_k} after L-channel highlight/shadow trimming",
                "stabilization": "when enabled, each split uses the median raw Lab as final per-image dominant_lab while preserving raw_* fields",
                "matching_metric": "CIEDE2000",
            },
        },
        "datasets": {},
    }

    all_train_consistency: List[float] = []
    all_test_consistency: List[float] = []
    all_raw_train_consistency: List[float] = []
    all_raw_test_consistency: List[float] = []
    total_classes = sum(len(class_map) for class_map in classes.values())
    done = 0
    log(f"found {total_classes} classes")

    for group, class_map in classes.items():
        result["datasets"][group] = {}
        for class_name, image_paths in class_map.items():
            done += 1
            log(f"processing class {done}/{total_classes}: {group}/{class_name}")
            train_paths, test_paths = split_image_paths(image_paths, seed=args.seed, train_n=args.train_n)

            train_summary = summarize_split(train_paths, color_db, segmenter, args, group, class_name, "train")
            test_summary = summarize_split(test_paths, color_db, segmenter, args, group, class_name, "test")
            prototype_gap = float(
                delta_e_ciede2000(
                    np.array(train_summary["prototype_lab"], dtype=np.float64),
                    np.array(test_summary["prototype_lab"], dtype=np.float64),
                )
            )

            result["datasets"][group][class_name] = {
                "train": train_summary,
                "test": test_summary,
                "prototype_gap_train_vs_test": round(prototype_gap, 4),
            }
            all_train_consistency.append(train_summary["consistency_score"])
            all_test_consistency.append(test_summary["consistency_score"])
            all_raw_train_consistency.append(train_summary["raw_consistency_score"])
            all_raw_test_consistency.append(test_summary["raw_consistency_score"])

    result["overall"] = {
        "mean_train_consistency": round(float(np.mean(all_train_consistency)), 4),
        "mean_test_consistency": round(float(np.mean(all_test_consistency)), 4),
        "max_train_consistency": round(float(np.max(all_train_consistency)), 4),
        "max_test_consistency": round(float(np.max(all_test_consistency)), 4),
        "mean_raw_train_consistency": round(float(np.mean(all_raw_train_consistency)), 4),
        "mean_raw_test_consistency": round(float(np.mean(all_raw_test_consistency)), 4),
        "max_raw_train_consistency": round(float(np.max(all_raw_train_consistency)), 4),
        "max_raw_test_consistency": round(float(np.max(all_raw_test_consistency)), 4),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM/RMBG + Lab/DeltaE00 cotton color recognition")
    parser.add_argument("--dataset_root", default="cotton_image", help="root folder containing gray/ and colorful/")
    parser.add_argument("--color_db", default="color_dataset.json", help="color database JSON")
    parser.add_argument("--output_json", default="out/cotton_color_results_hf.json", help="output JSON path")
    parser.add_argument("--segmentation_backend", choices=["sam", "rmbg", "auto", "classical", "color_threshold"], default="sam")
    parser.add_argument("--sam_model", default="facebook/sam-vit-base")
    parser.add_argument("--sam_prompt_mode", choices=["rough_box", "full_box"], default="rough_box", help="SAM prompt box source: rough foreground bbox or full image box")
    parser.add_argument("--rmbg_model", default="briaai/RMBG-1.4")
    parser.add_argument("--rmbg_image_size", type=int, default=1024)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--allow_classical_fallback", action="store_true", help="fallback to classical mask when model mask is invalid")
    parser.add_argument("--mask_debug_dir", default="", help="optional folder to save binary masks")
    parser.add_argument("--overlay_debug_dir", default="", help="optional folder to save mask overlays on processed images")
    parser.add_argument("--log_file", default="", help="optional log file written by Python itself")
    parser.add_argument("--max_size", type=int, default=1280, help="resize images so the longest side is at most this many pixels; use 0 to disable")
    parser.add_argument("--max_color_pixels", type=int, default=12000, help="maximum foreground Lab pixels sampled for dominant-color K-means; use 0 to disable")
    parser.add_argument("--only_group", choices=["gray", "colorful"], default="", help="optional group filter for quick runs")
    parser.add_argument("--only_class", default="", help="optional class folder filter for quick runs, e.g. 1 or 10")
    parser.add_argument("--limit_images_per_split", type=int, default=0, help="optional quick-debug limit for train/test images per class")
    parser.add_argument("--stabilize_outputs", action="store_true", help="replace each image dominant_lab/top3 with the split median prototype while preserving raw_* fields")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_n", type=int, default=6)
    parser.add_argument("--kmeans_k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    global LOG_FILE
    args = parse_args()
    if args.log_file:
        LOG_FILE = Path(args.log_file)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text("", encoding="utf-8")
    log("started")
    result = process_dataset(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"saved {output_path}")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
