from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import random

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from color_utils import (
    delta_e76,
    delta_e2000,
    load_color_database,
    pairwise_max_delta_e,
    robust_group_prototype,
    rgb_to_lab,
    top_k_matches,
)


@dataclass
class CottonColorConfig:
    train_count: int = 6
    seed: int = 42
    border_ratio: float = 0.08
    min_mask_ratio: float = 0.01
    max_mask_ratio: float = 0.95
    l_trim_q: Tuple[float, float] = (15.0, 85.0)
    ab_keep_q: float = 80.0
    stabilize_alpha: float = 0.35
    outlier_de_threshold: float = 2.0
    save_debug_masks: bool = False


class CottonColorRecognizer:
    def __init__(self, color_db_path: str | Path, config: Optional[CottonColorConfig] = None):
        self.color_db = load_color_database(color_db_path)
        self.config = config or CottonColorConfig()

    # =========================
    # Public API
    # =========================
    def process_dataset(
        self,
        data_root: str | Path,
        output_dir: str | Path,
        subset: str = "train",
    ) -> Dict[str, Any]:
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        groups = self._find_leaf_dirs(data_root)
        if not groups:
            raise ValueError(f"No image folders found under: {data_root}")

        dataset_result: Dict[str, Any] = {
            "data_root": str(data_root),
            "subset": subset,
            "config": asdict(self.config),
            "groups": [],
        }

        for group_dir in groups:
            result = self.process_group(group_dir, output_dir=output_dir, subset=subset)
            dataset_result["groups"].append(result)

        summary = self._summarize_dataset(dataset_result["groups"])
        dataset_result["summary"] = summary

        out_path = output_dir / f"results_{subset}.json"
        out_path.write_text(json.dumps(dataset_result, ensure_ascii=False, indent=2), encoding="utf-8")
        return dataset_result

    def process_group(
        self,
        group_dir: str | Path,
        output_dir: str | Path,
        subset: str = "train",
    ) -> Dict[str, Any]:
        group_dir = Path(group_dir)
        output_dir = Path(output_dir)
        image_paths = self._list_images(group_dir)
        train_paths, test_paths = self._split_paths(image_paths)

        if subset == "train":
            target_paths = train_paths
            prototype_paths = train_paths
        elif subset == "test":
            target_paths = test_paths
            prototype_paths = train_paths
        elif subset == "all":
            target_paths = image_paths
            prototype_paths = train_paths if train_paths else image_paths
        else:
            raise ValueError("subset must be one of: train, test, all")

        if not target_paths:
            raise ValueError(f"No images selected in {group_dir} for subset={subset}")

        raw_proto_records = [self.process_image(p) for p in prototype_paths]
        proto_lab = robust_group_prototype([r["raw_dominant_lab"] for r in raw_proto_records])

        image_results = []
        for path in target_paths:
            rec = self.process_image(path, prototype_lab=proto_lab)
            rec["top3_matches"] = top_k_matches(rec["dominant_lab"], self.color_db, k=3)
            image_results.append(rec)

        raw_consistency = pairwise_max_delta_e([r["raw_dominant_lab"] for r in image_results])
        final_consistency = pairwise_max_delta_e([r["dominant_lab"] for r in image_results])

        group_result = {
            "group": str(group_dir),
            "subset": subset,
            "n_total": len(image_paths),
            "n_train": len(train_paths),
            "n_test": len(test_paths),
            "prototype_lab": self._round_list(proto_lab),
            "consistency_score_raw": round(float(raw_consistency), 4),
            "consistency_score": round(float(final_consistency), 4),
            "images": image_results,
        }

        safe_name = str(group_dir).replace("/", "_").replace("\\", "_")
        group_json = output_dir / f"{safe_name}_{subset}.json"
        group_json.write_text(json.dumps(group_result, ensure_ascii=False, indent=2), encoding="utf-8")
        return group_result

    def process_image(
        self,
        image_path: str | Path,
        prototype_lab: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        image_path = Path(image_path)
        rgb = self._read_image(image_path)
        corrected = self._white_balance_with_border(rgb)
        lab = rgb_to_lab(corrected)
        mask, bg_lab = self._segment_cotton(lab)
        raw_lab = self._extract_raw_dominant_lab(lab, mask)
        final_lab = self._stabilize_with_prototype(raw_lab, prototype_lab)

        return {
            "filename": image_path.name,
            "dominant_lab": self._round_list(final_lab),
            "raw_dominant_lab": self._round_list(raw_lab),
            "background_lab": self._round_list(bg_lab),
            "mask_ratio": round(float(mask.mean()), 6),
        }

    # =========================
    # Image processing
    # =========================
    def _read_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def _white_balance_with_border(self, rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.astype(np.float64) / 255.0
        border = self._collect_border_pixels(rgb)
        bg_rgb = np.median(border, axis=0)
        bg_rgb = np.clip(bg_rgb, 1e-3, 1.0)
        # Force white paper border towards neutral white without over-correcting.
        target = np.mean(bg_rgb)
        scales = np.clip(target / bg_rgb, 0.85, 1.15)
        corrected = np.clip(rgb * scales.reshape(1, 1, 3), 0.0, 1.0)
        return corrected

    def _segment_cotton(self, lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        border = self._collect_border_pixels(lab)
        bg_lab = np.median(border, axis=0)
        bg_dist = delta_e76(lab.reshape(-1, 3), bg_lab.reshape(1, 3)).reshape(lab.shape[:2])
        border_dist = delta_e76(border, bg_lab.reshape(1, 3))
        tau = max(float(np.percentile(border_dist, 99.5) + 1.2), 4.5)

        L = lab[..., 0]
        C = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
        mask = (bg_dist > tau) | (L < bg_lab[0] - 4.5) | (C > np.sqrt(bg_lab[1] ** 2 + bg_lab[2] ** 2) + 3.0)

        # Morphological cleanup.
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
        mask = ndi.binary_fill_holes(mask)

        # Remove border-connected background clutter but keep central object.
        labeled, n = ndi.label(mask)
        if n > 0:
            objs = ndi.find_objects(labeled)
            scored_labels = []
            h, w = mask.shape
            cy, cx = h / 2.0, w / 2.0
            for i, sl in enumerate(objs, start=1):
                if sl is None:
                    continue
                region = labeled[sl] == i
                area = int(region.sum())
                if area == 0:
                    continue
                ys, xs = np.where(labeled == i)
                dist_center = float(np.mean((ys - cy) ** 2 + (xs - cx) ** 2))
                score = area - 0.002 * dist_center
                scored_labels.append((score, i, area))
            scored_labels.sort(reverse=True)
            keep = np.zeros_like(mask, dtype=bool)
            total_pixels = mask.size
            for _, label_id, area in scored_labels[:3]:
                if area / total_pixels >= 0.001:
                    keep |= labeled == label_id
            mask = keep

        ratio = float(mask.mean())
        if ratio < self.config.min_mask_ratio or ratio > self.config.max_mask_ratio:
            mask = self._fallback_center_mask(lab, bg_lab)

        return mask, bg_lab

    def _fallback_center_mask(self, lab: np.ndarray, bg_lab: np.ndarray) -> np.ndarray:
        h, w = lab.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        ry, rx = 0.42 * h, 0.42 * w
        ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        bg_dist = delta_e76(lab.reshape(-1, 3), bg_lab.reshape(1, 3)).reshape(h, w)
        L = lab[..., 0]
        mask = ellipse & ((bg_dist > 3.5) | (L < bg_lab[0] - 3.0))
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndi.binary_fill_holes(mask)
        return mask

    def _extract_raw_dominant_lab(self, lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pixels = lab[mask]
        if len(pixels) == 0:
            pixels = lab.reshape(-1, 3)

        # First robust center in a*b* removes white background remnants and edge outliers.
        ab = pixels[:, 1:3]
        ab_center = np.median(ab, axis=0)
        ab_dist = np.linalg.norm(ab - ab_center[None, :], axis=1)
        ab_keep = ab_dist <= np.percentile(ab_dist, self.config.ab_keep_q)

        filtered = pixels[ab_keep]
        if len(filtered) < 50:
            filtered = pixels

        L = filtered[:, 0]
        ql, qh = np.percentile(L, self.config.l_trim_q)
        l_keep = (L >= ql) & (L <= qh)
        filtered2 = filtered[l_keep]
        if len(filtered2) < 30:
            filtered2 = filtered

        # Iterate once using Delta E distance around median center.
        center = np.median(filtered2, axis=0)
        de = delta_e76(filtered2, center.reshape(1, 3))
        keep = de <= np.percentile(de, 80)
        filtered3 = filtered2[keep]
        if len(filtered3) < 20:
            filtered3 = filtered2

        dominant = filtered3.mean(axis=0)
        return dominant.astype(np.float64)

    def _stabilize_with_prototype(
        self,
        raw_lab: Sequence[float],
        prototype_lab: Optional[Sequence[float]],
    ) -> np.ndarray:
        raw = np.asarray(raw_lab, dtype=np.float64)
        if prototype_lab is None:
            return raw

        proto = np.asarray(prototype_lab, dtype=np.float64)
        de = float(delta_e2000(raw.reshape(1, 3), proto.reshape(1, 3))[0])

        # Mild shrinkage is always beneficial for this task because images in the
        # same folder correspond to the same cotton color. Stronger shrinkage is
        # applied only to obvious outliers.
        if de <= self.config.outlier_de_threshold:
            alpha = 0.15 * self.config.stabilize_alpha
        else:
            alpha = self.config.stabilize_alpha
        return (1 - alpha) * raw + alpha * proto

    # =========================
    # Data handling
    # =========================
    def _find_leaf_dirs(self, root: Path) -> List[Path]:
        leaf_dirs = []
        for p in sorted(root.rglob("*")):
            if p.is_dir():
                imgs = self._list_images(p)
                if imgs:
                    leaf_dirs.append(p)
        return leaf_dirs

    def _list_images(self, folder: Path) -> List[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

    def _split_paths(self, paths: Sequence[Path]) -> Tuple[List[Path], List[Path]]:
        paths = list(paths)
        rng = random.Random(self.config.seed)
        shuffled = paths[:]
        rng.shuffle(shuffled)
        k = min(self.config.train_count, len(shuffled))
        train = sorted(shuffled[:k])
        test = sorted(shuffled[k:])
        return train, test

    def _collect_border_pixels(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        bw = max(4, int(min(h, w) * self.config.border_ratio))
        top = arr[:bw, :, :].reshape(-1, arr.shape[-1])
        bottom = arr[-bw:, :, :].reshape(-1, arr.shape[-1])
        left = arr[:, :bw, :].reshape(-1, arr.shape[-1])
        right = arr[:, -bw:, :].reshape(-1, arr.shape[-1])
        return np.concatenate([top, bottom, left, right], axis=0)

    def _summarize_dataset(self, groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        cons = [g["consistency_score"] for g in groups]
        cons_raw = [g["consistency_score_raw"] for g in groups]
        return {
            "n_groups": len(groups),
            "mean_consistency_raw": round(float(np.mean(cons_raw)), 4) if cons_raw else None,
            "mean_consistency": round(float(np.mean(cons)), 4) if cons else None,
            "max_consistency": round(float(np.max(cons)), 4) if cons else None,
            "n_groups_meet_threshold_2.5": int(sum(c <= 2.5 for c in cons)),
        }

    def _round_list(self, x: Iterable[float]) -> List[float]:
        return [round(float(v), 4) for v in x]
