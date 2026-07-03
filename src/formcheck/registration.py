from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_io import imread, imwrite


@dataclass(frozen=True)
class RegistrationConfig:
    min_inliers: int = 70
    max_rmse: float = 3.5
    ratio_test: float = 0.75
    ransac_reproj_threshold: float = 4.0
    nfeatures: int = 8000
    use_red_suppression: bool = True
    min_area_ratio: float = 0.35
    max_area_ratio: float = 2.5
    max_anisotropy: float = 2.2


@dataclass
class TemplateContext:
    image: np.ndarray
    width: int
    height: int
    keypoints: list
    descriptors: np.ndarray
    config: RegistrationConfig


@dataclass
class RegistrationResult:
    ok: bool
    warped: np.ndarray | None
    homography: np.ndarray | None
    inliers: int
    reproj_rmse: float
    reject_reason: str | None = None


def preprocess(image: np.ndarray, use_red_suppression: bool) -> np.ndarray:
    if use_red_suppression:
        return image[:, :, 2]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _detector(config: RegistrationConfig):
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=config.nfeatures), "sift"
    return cv2.ORB_create(nfeatures=config.nfeatures), "orb"


def load_template(canonical_dir: Path, config: RegistrationConfig | None = None) -> TemplateContext:
    cfg = config or RegistrationConfig()
    candidates = [
        canonical_dir / "template_photo_registration.png",
        canonical_dir / "template_registration.png",
        canonical_dir / "template.png",
    ]
    image = imread(next(path for path in candidates if path.exists()))
    gray = preprocess(image, cfg.use_red_suppression)
    detector, _ = _detector(cfg)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 20:
        raise ValueError("Template has too few features")
    return TemplateContext(
        image=image,
        width=image.shape[1],
        height=image.shape[0],
        keypoints=keypoints,
        descriptors=descriptors,
        config=cfg,
    )


def register(image: np.ndarray, template: TemplateContext) -> RegistrationResult:
    cfg = template.config
    gray = preprocess(image, cfg.use_red_suppression)
    detector, kind = _detector(cfg)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 20:
        return RegistrationResult(False, None, None, 0, float("inf"), "features<20: 图像特征过少")

    if kind == "sift":
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        matches = matcher.knnMatch(descriptors, template.descriptors, k=2)
        good = [m for m, n in matches if m.distance < cfg.ratio_test * n.distance]
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(descriptors, template.descriptors, k=2)
        good = [m for m, n in matches if m.distance < 0.8 * n.distance]
    if len(good) < 8:
        return RegistrationResult(False, None, None, len(good), float("inf"), f"matches={len(good)} < 8: 匹配点过少")

    src = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([template.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    homography, mask = cv2.findHomography(src, dst, method, cfg.ransac_reproj_threshold)
    if homography is None or mask is None:
        return RegistrationResult(False, None, None, 0, float("inf"), "homography_failed: 单应性估计失败")
    inliers = int(mask.ravel().sum())
    projected = cv2.perspectiveTransform(src[mask.astype(bool).ravel()], homography)
    target = dst[mask.astype(bool).ravel()]
    rmse = float(np.sqrt(np.mean(np.sum((projected - target) ** 2, axis=2)))) if inliers else float("inf")
    if inliers < cfg.min_inliers:
        return RegistrationResult(False, None, homography, inliers, rmse, f"inliers={inliers} < {cfg.min_inliers}: 照片可能未拍全或模糊")
    if rmse > cfg.max_rmse:
        return RegistrationResult(False, None, homography, inliers, rmse, f"rmse={rmse:.2f} > {cfg.max_rmse}: 配准误差过大")
    sanity_error = _homography_sanity(image, template, homography, cfg)
    if sanity_error:
        return RegistrationResult(False, None, homography, inliers, rmse, sanity_error)
    warped = cv2.warpPerspective(image, homography, (template.width, template.height), borderValue=(255, 255, 255))
    return RegistrationResult(True, warped, homography, inliers, rmse)


def _homography_sanity(
    image: np.ndarray,
    template: TemplateContext,
    homography: np.ndarray,
    cfg: RegistrationConfig,
) -> str | None:
    h, w = image.shape[:2]
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    area = abs(cv2.contourArea(mapped.astype("float32")))
    template_area = float(template.width * template.height)
    ratio = area / template_area
    if not (cfg.min_area_ratio <= ratio <= cfg.max_area_ratio):
        return f"area_ratio={ratio:.2f}: 单应性退化或表单未完整覆盖"
    if not cv2.isContourConvex(mapped.astype("float32")):
        return "quad_not_convex: 单应性四边形非凸"
    affine = homography[:2, :2]
    _, singular_values, _ = np.linalg.svd(affine)
    if singular_values[-1] <= 1e-8:
        return "singular_homography: 单应性接近奇异"
    anisotropy = float(singular_values[0] / singular_values[-1])
    if anisotropy > cfg.max_anisotropy:
        return f"anisotropy={anisotropy:.2f}: 单应性拉伸过大"
    return None


def save_registration_summary(path: Path, result: RegistrationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ok": result.ok,
        "inliers": result.inliers,
        "reproj_rmse": result.reproj_rmse,
        "reject_reason": result.reject_reason,
        "homography": result.homography.tolist() if result.homography is not None else None,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def register_file(input_path: Path, canonical_dir: Path, out_path: Path) -> RegistrationResult:
    template = load_template(canonical_dir)
    result = register(imread(input_path), template)
    if result.ok and result.warped is not None:
        imwrite(out_path, result.warped)
    return result
