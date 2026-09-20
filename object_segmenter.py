"""Automatic FLUX-object extraction constrained by the user's edit mask.

The CPU-only parts in this module build prompts and composite layers. SAM and
GroundingDINO are imported lazily so ordinary preprocessing/tests do not need
either model installed.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from image_processing import expand_mask, normalize_mask, rgb, shrink_mask


DEFAULT_DINO_ROOT = os.getenv("GROUNDING_DINO_ROOT", "third_party/GroundingDINO")
DEFAULT_DINO_CONFIG = os.getenv(
    "GROUNDING_DINO_CONFIG",
    str(Path(DEFAULT_DINO_ROOT) / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
)
DEFAULT_DINO_CHECKPOINT = os.getenv(
    "GROUNDING_DINO_CHECKPOINT",
    str(Path(DEFAULT_DINO_ROOT) / "weights/groundingdino_swint_ogc.pth"),
)


@dataclass
class AutoPrompt:
    box_xyxy: tuple[float, float, float, float]
    positive_points: list[list[float]]
    negative_points: list[list[float]]
    difference_mask: Image.Image
    difference_preview: Image.Image
    score_map: np.ndarray
    work_scale_x: float
    work_scale_y: float


@dataclass
class SegmentationResult:
    mask: Image.Image | None
    score: float
    quality: float
    kind: str
    logits: np.ndarray | None
    auto_prompt: AutoPrompt
    positive_points: list[list[float]]
    negative_points: list[list[float]]
    accepted: bool
    message: str
    used_dino: bool = False


@dataclass
class CompositeResult:
    final: Image.Image
    object_alpha: Image.Image
    shadow_mask: Image.Image


def full_to_roi_point(
    point: Sequence[float], roi_box: Sequence[int]
) -> list[float] | None:
    """Convert a full-image UI click to ROI coordinates, rejecting outside clicks."""
    if len(point) < 2 or len(roi_box) < 4:
        return None
    left, top, right, bottom = [float(value) for value in roi_box[:4]]
    x, y = float(point[0]) - left, float(point[1]) - top
    if not (0.0 <= x < right - left and 0.0 <= y < bottom - top):
        return None
    return [x, y]


def merge_prompt_history(
    auto_positive: Sequence[Sequence[float]],
    auto_negative: Sequence[Sequence[float]],
    history: Sequence[dict[str, Any]],
) -> tuple[list[list[float]], list[list[float]]]:
    positive = [list(item) for item in auto_positive]
    negative = [list(item) for item in auto_negative]
    for item in history:
        target = positive if int(item.get("label", 0)) == 1 else negative
        target.append([float(value) for value in item["point"][:2]])
    return positive, negative


def _luma(array: np.ndarray) -> np.ndarray:
    return (
        array[..., 0] * 0.2126
        + array[..., 1] * 0.7152
        + array[..., 2] * 0.0722
    )


def _robust_unit(values: np.ndarray, selection: np.ndarray) -> np.ndarray:
    samples = values[selection]
    if samples.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(samples, (20, 95))
    if float(high - low) < 1e-5:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - float(low)) / float(high - low), 0.0, 1.0).astype(
        np.float32
    )


def _components(binary: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected components as flat integer indices."""
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    found: list[np.ndarray] = []
    for start_y, start_x in np.argwhere(binary):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        indices: list[int] = []
        while queue:
            y, x = queue.popleft()
            indices.append(y * width + x)
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= height:
                    continue
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    if 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        found.append(np.asarray(indices, dtype=np.int64))
    return found


def _representative_point(component: np.ndarray, width: int) -> tuple[int, int]:
    """Return an interior distance peak, not merely the component centroid."""
    ys = component // width
    xs = component % width
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    local[ys - y0, xs - x0] = True
    padded = np.pad(local, 1, constant_values=False)
    interior = np.ones_like(local)
    for dy in range(3):
        for dx in range(3):
            interior &= padded[dy:dy + local.shape[0], dx:dx + local.shape[1]]
    boundary_y, boundary_x = np.where(local & ~interior)
    distance = np.full(local.shape, -1, dtype=np.int32)
    queue = deque()
    for y, x in zip(boundary_y, boundary_x):
        distance[y, x] = 0
        queue.append((int(y), int(x)))
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < local.shape[0]
                    and 0 <= nx < local.shape[1]
                    and local[ny, nx]
                    and distance[ny, nx] < 0
                ):
                    distance[ny, nx] = distance[y, x] + 1
                    queue.append((ny, nx))
    peak_y, peak_x = np.where(distance == distance.max())
    center_x, center_y = float(np.mean(xs) - x0), float(np.mean(ys) - y0)
    nearest = np.argmin((peak_x - center_x) ** 2 + (peak_y - center_y) ** 2)
    return int(peak_x[nearest] + x0), int(peak_y[nearest] + y0)


def _spread_points(
    candidates: np.ndarray, count: int, score_map: np.ndarray
) -> list[tuple[int, int]]:
    if len(candidates) == 0 or count <= 0:
        return []
    height, width = score_map.shape
    ys = candidates // width
    xs = candidates % width
    first = int(np.argmin(score_map[ys, xs]))
    chosen = [(int(xs[first]), int(ys[first]))]
    while len(chosen) < count and len(chosen) < len(candidates):
        distance = np.full(len(candidates), np.inf, dtype=np.float32)
        for px, py in chosen:
            distance = np.minimum(distance, (xs - px) ** 2 + (ys - py) ** 2)
        distance -= score_map[ys, xs] * float(width * width + height * height) * 0.05
        next_index = int(np.argmax(distance))
        point = (int(xs[next_index]), int(ys[next_index]))
        if point in chosen:
            break
        chosen.append(point)
    return chosen


def build_auto_prompt(
    background_roi: Image.Image,
    generated_roi: Image.Image,
    edit_mask_roi: Image.Image,
    *,
    max_work_side: int = 512,
) -> AutoPrompt:
    """Build post-FLUX SAM prompts, using the user's painted-mask bbox."""
    background_roi = rgb(background_roi)
    generated_roi = rgb(generated_roi).resize(
        background_roi.size, Image.Resampling.LANCZOS
    )
    edit_mask_roi = normalize_mask(edit_mask_roi, background_roi.size)
    if edit_mask_roi.getbbox() is None:
        raise ValueError("The editing mask is empty. Cannot extract the generated subject.")

    scale = min(1.0, float(max_work_side) / max(background_roi.size))
    work_size = (
        max(16, int(round(background_roi.width * scale))),
        max(16, int(round(background_roi.height * scale))),
    )
    base = np.asarray(
        background_roi.resize(work_size, Image.Resampling.LANCZOS), dtype=np.float32
    )
    generated = np.asarray(
        generated_roi.resize(work_size, Image.Resampling.LANCZOS), dtype=np.float32
    )
    mask_image = edit_mask_roi.resize(work_size, Image.Resampling.NEAREST)
    allowed = np.asarray(mask_image, dtype=np.uint8) >= 128

    # Remove broad exposure/color offsets before change detection. Estimate the
    # bias from the edit-mask inner boundary, which is much more likely to be
    # continuation background than the newly generated object interior.
    residual = generated - base
    bias_eroded = mask_image.filter(ImageFilter.MinFilter(11))
    bias_ring = allowed & (np.asarray(bias_eroded, dtype=np.uint8) < 128)
    bias_samples = residual[bias_ring] if np.count_nonzero(bias_ring) >= 16 else residual[allowed]
    median_bias = np.median(bias_samples, axis=0)
    median_bias = np.clip(median_bias, -24.0, 24.0)
    generated_aligned = np.clip(generated - median_bias, 0.0, 255.0)
    color_difference = np.mean(np.abs(generated_aligned - base), axis=2)

    base_luma = _luma(base)
    generated_luma = _luma(generated_aligned)
    base_gy, base_gx = np.gradient(base_luma)
    generated_gy, generated_gx = np.gradient(generated_luma)
    gradient_difference = np.abs(np.hypot(generated_gx, generated_gy) - np.hypot(base_gx, base_gy))
    color_unit = _robust_unit(color_difference, allowed)
    gradient_unit = _robust_unit(gradient_difference, allowed)
    score_map = (0.60 * color_unit + 0.40 * gradient_unit) * allowed.astype(np.float32)

    samples = score_map[allowed]
    threshold = max(0.18, float(np.percentile(samples, 80)))
    high = (score_map >= threshold) & allowed
    high_image = Image.fromarray((high.astype(np.uint8) * 255), mode="L")
    # 3px open followed by 7px close.
    high_image = high_image.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    high_image = high_image.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    high = (np.asarray(high_image, dtype=np.uint8) >= 128) & allowed

    allowed_pixels = max(1, int(np.count_nonzero(allowed)))
    minimum_area = max(8, int(math.ceil(allowed_pixels * 0.002)))
    components = [item for item in _components(high) if len(item) >= minimum_area]
    width = work_size[0]
    components.sort(
        key=lambda item: float(np.sum(score_map[item // width, item % width])),
        reverse=True,
    )
    selected = components[:3]
    if not selected:
        # Keep the task recoverable even for low-contrast objects: seed SAM from
        # the strongest allowed location and use the user's mask bbox as box.
        flat_allowed = np.flatnonzero(allowed)
        best = int(flat_allowed[np.argmax(score_map.flat[flat_allowed])])
        selected = [np.asarray([best], dtype=np.int64)]

    selected_binary = np.zeros_like(allowed, dtype=bool)
    positive_work: list[tuple[int, int]] = []
    for component in selected:
        selected_binary.flat[component] = True
        positive_work.append(_representative_point(component, width))

    eroded = mask_image.filter(ImageFilter.MinFilter(7))
    inner_ring = allowed & (np.asarray(eroded, dtype=np.uint8) < 128)
    low_limit = float(np.percentile(samples, 30))
    negative_candidates = np.flatnonzero(inner_ring & (score_map <= low_limit))
    negative_work = _spread_points(negative_candidates, 4, score_map)

    scale_x = background_roi.width / work_size[0]
    scale_y = background_roi.height / work_size[1]
    positive = [[(x + 0.5) * scale_x, (y + 0.5) * scale_y] for x, y in positive_work]
    negative = [[(x + 0.5) * scale_x, (y + 0.5) * scale_y] for x, y in negative_work]
    # The yellow localization box is the exact full-resolution bounding box of
    # everything the user painted: leftmost/rightmost and topmost/bottommost
    # non-zero mask pixels. Difference analysis still supplies SAM points, but
    # it no longer changes the box. This function runs only after FLUX returns,
    # so the box is drawn on the generated object image rather than the input.
    mask_bbox_full = edit_mask_roi.getbbox()
    assert mask_bbox_full is not None
    box = tuple(float(value) for value in mask_bbox_full)

    difference_mask = Image.fromarray(
        (selected_binary.astype(np.uint8) * 255), mode="L"
    ).resize(background_roi.size, Image.Resampling.NEAREST)
    preview = generated_roi.copy()
    overlay = Image.new("RGB", preview.size, (24, 210, 92))
    preview = Image.composite(overlay, preview, difference_mask.point(lambda value: int(value * 0.42)))
    draw = ImageDraw.Draw(preview)
    draw.rectangle(
        box,
        outline=(255, 215, 0),
        width=max(3, round(min(preview.size) / 220)),
    )
    radius = max(5, round(min(preview.size) / 100))
    for x, y in positive:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(20, 230, 90), outline="white", width=2)
    for x, y in negative:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(235, 55, 55), outline="white", width=2)
    return AutoPrompt(
        box_xyxy=box,
        positive_points=positive,
        negative_points=negative,
        difference_mask=difference_mask,
        difference_preview=preview,
        score_map=score_map,
        work_scale_x=scale_x,
        work_scale_y=scale_y,
    )


def _candidate_quality(
    candidate: Image.Image,
    sam_score: float,
    prompt: AutoPrompt,
    edit_mask: Image.Image,
) -> tuple[float, bool, dict[str, float]]:
    candidate_array = np.asarray(normalize_mask(candidate, edit_mask.size), dtype=np.uint8) >= 128
    allowed = np.asarray(normalize_mask(edit_mask, edit_mask.size), dtype=np.uint8) >= 128
    area = int(np.count_nonzero(candidate_array))
    allowed_area = max(1, int(np.count_nonzero(allowed)))
    inside = int(np.count_nonzero(candidate_array & allowed))
    containment = inside / max(1, area)
    area_ratio = inside / allowed_area
    high = np.asarray(prompt.difference_mask.resize(edit_mask.size, Image.Resampling.NEAREST), dtype=np.uint8) >= 128
    diff_recall = float(np.count_nonzero(candidate_array & high)) / max(1, int(np.count_nonzero(high)))
    border = np.zeros_like(candidate_array)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    edge_touch = float(np.count_nonzero(candidate_array & border)) / max(1, area)
    quality = (
        0.40 * float(np.clip(sam_score, 0.0, 1.0))
        + 0.30 * float(np.clip(diff_recall, 0.0, 1.0))
        + 0.20 * float(np.clip(containment, 0.0, 1.0))
        + 0.10 * (1.0 - float(np.clip(edge_touch * 20.0, 0.0, 1.0)))
    )
    accepted = (
        0.01 <= area_ratio <= 0.90
        and containment >= 0.95
        and quality >= 0.55
    )
    return quality, accepted, {
        "area_ratio": area_ratio,
        "containment": containment,
        "diff_recall": diff_recall,
        "edge_touch": edge_touch,
    }


def segment_generated_object(
    background_roi: Image.Image,
    generated_roi: Image.Image,
    edit_mask_roi: Image.Image,
    *,
    backend: str = "sam2",
    sam2_path: str,
    sam1_path: str,
    dino_prompt: str = "",
    dino_root: str = DEFAULT_DINO_ROOT,
    dino_config: str = DEFAULT_DINO_CONFIG,
    dino_checkpoint: str = DEFAULT_DINO_CHECKPOINT,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> SegmentationResult:
    prompt = build_auto_prompt(background_roi, generated_roi, edit_mask_roi)
    import sam_segmenter

    try:
        prediction = sam_segmenter.segment_with_prompts(
            generated_roi,
            box_xyxy=prompt.box_xyxy,
            positive_points=prompt.positive_points,
            negative_points=prompt.negative_points,
            backend=backend,
            sam2_path=sam2_path,
            sam1_path=sam1_path,
        )
    except Exception as exc:
        return SegmentationResult(
            mask=None,
            score=0.0,
            quality=0.0,
            kind="unavailable",
            logits=None,
            auto_prompt=prompt,
            positive_points=[list(item) for item in prompt.positive_points],
            negative_points=[list(item) for item in prompt.negative_points],
            accepted=False,
            message=f"SAM is unavailable. The initial composite has been preserved: {exc}",
        )
    result = _choose_prediction(prediction, prompt, edit_mask_roi)
    if result.accepted or not dino_prompt.strip():
        return result

    try:
        sam_segmenter.release_segmenter()
        dino_box = detect_grounding_box(
            generated_roi,
            edit_mask_roi,
            dino_prompt,
            root=dino_root,
            config=dino_config,
            checkpoint=dino_checkpoint,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        prediction = sam_segmenter.segment_with_prompts(
            generated_roi,
            box_xyxy=dino_box,
            positive_points=prompt.positive_points,
            negative_points=prompt.negative_points,
            backend=backend,
            sam2_path=sam2_path,
            sam1_path=sam1_path,
        )
        dino_result = _choose_prediction(prediction, prompt, edit_mask_roi)
        dino_result.used_dino = True
        dino_result.message = "After DINO fallback: " + dino_result.message
        return dino_result if dino_result.quality >= result.quality else result
    except Exception as exc:
        result.message += f"; DINO fallback failed: {exc}"
        return result


def failed_segmentation_result(
    background_roi: Image.Image,
    generated_roi: Image.Image,
    edit_mask_roi: Image.Image,
    message: str,
) -> SegmentationResult:
    """Build a recoverable result when SAM itself cannot be loaded or run."""
    prompt = build_auto_prompt(background_roi, generated_roi, edit_mask_roi)
    return SegmentationResult(
        mask=None,
        score=0.0,
        quality=0.0,
        kind="unavailable",
        logits=None,
        auto_prompt=prompt,
        positive_points=[list(item) for item in prompt.positive_points],
        negative_points=[list(item) for item in prompt.negative_points],
        accepted=False,
        message=f"Automatic SAM segmentation failed. The initial result is preserved. Check model paths and refine: {message}",
    )


def segment_with_dino_fallback(
    generated_roi: Image.Image,
    edit_mask_roi: Image.Image,
    previous: SegmentationResult,
    phrase: str,
    *,
    backend: str,
    sam2_path: str,
    sam1_path: str,
    dino_root: str,
    dino_config: str,
    dino_checkpoint: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> SegmentationResult:
    """Run the optional DINO -> SAM path on demand without re-running FLUX."""
    import sam_segmenter

    sam_segmenter.release_segmenter()
    dino_box = detect_grounding_box(
        generated_roi,
        edit_mask_roi,
        phrase,
        root=dino_root,
        config=dino_config,
        checkpoint=dino_checkpoint,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    prediction = sam_segmenter.segment_with_prompts(
        generated_roi,
        box_xyxy=dino_box,
        positive_points=previous.positive_points,
        negative_points=previous.negative_points,
        previous_mask_logits=previous.logits,
        backend=backend,
        sam2_path=sam2_path,
        sam1_path=sam1_path,
    )
    result = _choose_prediction(prediction, previous.auto_prompt, edit_mask_roi)
    result.positive_points = [list(item) for item in previous.positive_points]
    result.negative_points = [list(item) for item in previous.negative_points]
    result.used_dino = True
    result.message = "After manual DINO fallback: " + result.message
    return result


def _choose_prediction(
    prediction: dict[str, Any], prompt: AutoPrompt, edit_mask: Image.Image
) -> SegmentationResult:
    masks: Sequence[Image.Image] = prediction["masks"]
    scores: Sequence[float] = prediction["scores"]
    logits: Sequence[np.ndarray] = prediction["logits"]
    if not masks:
        return SegmentationResult(
            None, 0.0, 0.0, str(prediction.get("kind", "sam")), None,
            prompt, prompt.positive_points, prompt.negative_points, False,
            "SAM returned no masks.",
        )
    ranked = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        quality, accepted, details = _candidate_quality(mask, score, prompt, edit_mask)
        ranked.append((quality, accepted, index, details))
    # A formally valid candidate wins over an oversized/edge candidate even if
    # the latter reports a slightly higher raw SAM score.
    quality, accepted, best, details = max(
        ranked, key=lambda item: (bool(item[1]), float(item[0]))
    )
    allowed = normalize_mask(edit_mask, edit_mask.size)
    selected = ImageChops.multiply(normalize_mask(masks[best], edit_mask.size), allowed)
    message = (
        f"Automatic SAM mask {'accepted' if accepted else 'needs refinement'} | quality={quality:.3f} | "
        f"area={details['area_ratio']:.3f} | contain={details['containment']:.3f}"
    )
    return SegmentationResult(
        mask=selected,
        score=float(scores[best]),
        quality=float(quality),
        kind=str(prediction.get("kind", "sam")),
        logits=np.asarray(logits[best], dtype=np.float32) if len(logits) > best else None,
        auto_prompt=prompt,
        positive_points=[list(item) for item in prompt.positive_points],
        negative_points=[list(item) for item in prompt.negative_points],
        accepted=bool(accepted),
        message=message,
    )


def refine_segmentation(
    generated_roi: Image.Image,
    edit_mask_roi: Image.Image,
    previous: SegmentationResult,
    positive_points: Sequence[Sequence[float]],
    negative_points: Sequence[Sequence[float]],
    *,
    backend: str,
    sam2_path: str,
    sam1_path: str,
) -> SegmentationResult:
    import sam_segmenter

    prediction = sam_segmenter.segment_with_prompts(
        generated_roi,
        box_xyxy=previous.auto_prompt.box_xyxy,
        positive_points=positive_points,
        negative_points=negative_points,
        previous_mask_logits=previous.logits,
        backend=backend,
        sam2_path=sam2_path,
        sam1_path=sam1_path,
    )
    result = _choose_prediction(prediction, previous.auto_prompt, edit_mask_roi)
    result.positive_points = [list(item) for item in positive_points]
    result.negative_points = [list(item) for item in negative_points]
    result.message = "After point refinement: " + result.message
    return result


def detect_grounding_box(
    image: Image.Image,
    edit_mask: Image.Image,
    phrase: str,
    *,
    root: str,
    config: str,
    checkpoint: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> tuple[float, float, float, float]:
    """Lazy optional GroundingDINO fallback returning one edit-overlapping box."""
    phrase = phrase.strip()
    if not phrase:
        raise ValueError("The DINO detection phrase is empty.")
    root_path = Path(root).expanduser()
    config_path = Path(config).expanduser()
    checkpoint_path = Path(checkpoint).expanduser()
    for name, path in (("DINO source", root_path), ("DINO config", config_path), ("DINO weights", checkpoint_path)):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    import_root = root_path if (root_path / "groundingdino").is_dir() else root_path.parent
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    try:
        import torch
        from groundingdino.util.inference import load_image, load_model, predict
    except ImportError as exc:
        raise RuntimeError("GroundingDINO is not installed or its source path cannot be imported.") from exc

    model = load_model(str(config_path), str(checkpoint_path), device="cuda")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        rgb(image).save(temporary_path)
        image_source, tensor = load_image(str(temporary_path))
        boxes, logits, _phrases = predict(
            model=model,
            image=tensor,
            caption=f"{phrase} .",
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            device="cuda",
        )
    finally:
        temporary_path.unlink(missing_ok=True)
        del model
        _clear_cuda()
    if len(boxes) == 0:
        raise RuntimeError(f"DINO did not detect {phrase!r}.")
    width, height = image.size
    allowed = np.asarray(normalize_mask(edit_mask, image.size), dtype=np.uint8) >= 128
    candidates = []
    for index, box in enumerate(boxes):
        cx, cy, bw, bh = [float(value) for value in box]
        xyxy = (
            max(0.0, (cx - bw / 2) * width),
            max(0.0, (cy - bh / 2) * height),
            min(float(width), (cx + bw / 2) * width),
            min(float(height), (cy + bh / 2) * height),
        )
        x0, y0, x1, y1 = [int(round(value)) for value in xyxy]
        box_area = max(1, (x1 - x0) * (y1 - y0))
        overlap = int(np.count_nonzero(allowed[max(0, y0):min(height, y1), max(0, x0):min(width, x1)])) / box_area
        if overlap >= 0.20:
            confidence = float(logits[index])
            candidates.append((0.60 * confidence + 0.40 * min(1.0, overlap), xyxy))
    del boxes, logits
    _clear_cuda()
    if not candidates:
        raise RuntimeError("The DINO detection box does not overlap the editing region.")
    return max(candidates, key=lambda item: item[0])[1]


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def make_segmentation_preview(
    image: Image.Image,
    mask: Image.Image | None,
    positive_points: Sequence[Sequence[float]],
    negative_points: Sequence[Sequence[float]],
) -> Image.Image:
    preview = rgb(image)
    if mask is not None and mask.getbbox() is not None:
        overlay = Image.new("RGB", preview.size, (22, 210, 90))
        alpha = normalize_mask(mask, preview.size).point(lambda value: int(value * 0.38))
        preview = Image.composite(overlay, preview, alpha)
    draw = ImageDraw.Draw(preview)
    radius = max(5, round(min(preview.size) / 100))
    for points, color in ((positive_points, (20, 230, 90)), (negative_points, (235, 55, 55))):
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
    return preview


def composite_object_layers(
    full_background: Image.Image,
    generated_roi: Image.Image,
    object_mask_roi: Image.Image | None,
    core_mask_full: Image.Image,
    roi_box: tuple[int, int, int, int],
    *,
    object_feather: float = 0.0,
    object_edge_expand: int = 0,
    seam_gaussian_radius: float = 0.0,
    seam_gaussian_strength: float = 0.0,
    shadow_enabled: bool = True,
    shadow_radius: int = 96,
    shadow_strength: float = 0.75,
) -> CompositeResult:
    full_background = rgb(full_background)
    roi_size = (roi_box[2] - roi_box[0], roi_box[3] - roi_box[1])
    generated_roi = rgb(generated_roi).resize(roi_size, Image.Resampling.LANCZOS)
    background_roi = full_background.crop(roi_box)
    core_roi = normalize_mask(core_mask_full, full_background.size).crop(roi_box)
    if object_mask_roi is None or object_mask_roi.getbbox() is None:
        return CompositeResult(
            final=full_background.copy(),
            object_alpha=Image.new("L", roi_size, 0),
            shadow_mask=Image.new("L", roi_size, 0),
        )
    binary_object = ImageChops.multiply(normalize_mask(object_mask_roi, roi_size), core_roi)
    alpha_source = adjust_object_mask(
        binary_object, binary_object.size, int(object_edge_expand)
    )
    # Feather inward only. A symmetric Gaussian has non-zero values outside
    # the SAM mask and visibly pastes a pale generated-image halo around the
    # subject. Multiplying by alpha_source keeps every exterior pixel at zero
    # while retaining a gentle transition on the subject side of the seam.
    alpha = ImageChops.multiply(
        alpha_source.filter(
            ImageFilter.GaussianBlur(max(0.0, float(object_feather)))
        ),
        alpha_source,
    )
    alpha = ImageChops.multiply(alpha, core_roi)
    shadow_mask, shadowed_background = _apply_shadow_field(
        background_roi,
        generated_roi,
        binary_object,
        core_roi,
        enabled=shadow_enabled,
        radius=shadow_radius,
        strength=shadow_strength,
    )
    blended_roi = _gaussian_seam_composite(
        shadowed_background,
        generated_roi,
        alpha,
        radius=seam_gaussian_radius,
        strength=seam_gaussian_strength,
    )
    output = full_background.copy()
    output.paste(blended_roi, roi_box[:2])
    return CompositeResult(output, alpha, shadow_mask)


def adjust_object_mask(
    object_mask: Image.Image,
    size: tuple[int, int],
    edge_adjust: int = 0,
) -> Image.Image:
    """Return the exact binary subject ownership mask used by every stage.

    A negative adjustment removes SAM's commonly over-selected outer pixels;
    those pixels must then be background-edited rather than left protected.
    """
    binary = normalize_mask(object_mask, size)
    edge_adjust = int(edge_adjust)
    adjusted = (
        expand_mask(binary, edge_adjust)
        if edge_adjust > 0
        else shrink_mask(binary, -edge_adjust)
        if edge_adjust < 0
        else binary.copy()
    )
    return adjusted.point(lambda value: 255 if value >= 128 else 0)


def _gaussian_seam_composite(
    background: Image.Image,
    foreground: Image.Image,
    alpha: Image.Image,
    *,
    radius: float,
    strength: float,
) -> Image.Image:
    """Two-band Gaussian blend that softens only the object/background seam.

    A plain feathered alpha can still leave a sharp low-frequency color step,
    especially when FLUX exposure differs slightly from the original image.
    This keeps fine foreground/background texture on its own side while using
    a wider Gaussian alpha for the low-frequency colors. Pixels whose original
    alpha is exactly zero are restored byte-for-byte from the background.
    """
    background = rgb(background)
    foreground = rgb(foreground).resize(background.size, Image.Resampling.LANCZOS)
    alpha = alpha.convert("L").resize(background.size, Image.Resampling.BILINEAR)
    radius = max(0.0, min(32.0, float(radius)))
    strength = max(0.0, min(1.0, float(strength)))
    if radius < 0.25 or strength <= 0:
        return Image.composite(foreground, background, alpha)

    background_array = np.asarray(background, dtype=np.float32)
    foreground_array = np.asarray(foreground, dtype=np.float32)
    alpha_array = np.asarray(alpha, dtype=np.float32) / 255.0
    background_low = np.asarray(
        background.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32
    )
    foreground_low = np.asarray(
        foreground.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32
    )
    low_alpha = np.asarray(
        alpha.filter(ImageFilter.GaussianBlur(max(0.5, radius * 0.75))),
        dtype=np.float32,
    ) / 255.0

    standard = (
        foreground_array * alpha_array[..., None]
        + background_array * (1.0 - alpha_array[..., None])
    )
    low_band = (
        foreground_low * low_alpha[..., None]
        + background_low * (1.0 - low_alpha[..., None])
    )
    detail_band = (
        (foreground_array - foreground_low) * alpha_array[..., None]
        + (background_array - background_low) * (1.0 - alpha_array[..., None])
    )
    multiband = low_band + detail_band
    blended = standard * (1.0 - strength) + multiband * strength
    blended = np.where(
        alpha_array[..., None] > 0.0,
        blended,
        background_array,
    )
    return Image.fromarray(
        np.clip(np.rint(blended), 0, 255).astype(np.uint8), mode="RGB"
    )


def _apply_shadow_field(
    background: Image.Image,
    generated: Image.Image,
    object_mask: Image.Image,
    core_mask: Image.Image,
    *,
    enabled: bool,
    radius: int,
    strength: float,
) -> tuple[Image.Image, Image.Image]:
    empty = Image.new("L", background.size, 0)
    if not enabled or object_mask.getbbox() is None or float(strength) <= 0:
        return empty, background.copy()
    radius = max(1, min(256, int(radius)))
    strength = max(0.0, min(1.0, float(strength)))
    bbox = object_mask.getbbox()
    assert bbox is not None
    lower_limit = bbox[1] + int(round((bbox[3] - bbox[1]) * 0.45))
    band = ImageChops.subtract(expand_mask(object_mask, radius), object_mask)
    band = ImageChops.multiply(band, core_mask)
    allowed = np.asarray(band, dtype=np.uint8) >= 128
    rows = np.arange(background.height)[:, None]
    allowed &= rows >= lower_limit
    base = np.asarray(background, dtype=np.float32)
    generated_array = np.asarray(generated, dtype=np.float32)
    base_luma = _luma(base)
    generated_luma = _luma(generated_array)
    darkness = base_luma - generated_luma
    base_chroma = base - base_luma[..., None]
    generated_chroma = generated_array - generated_luma[..., None]
    chroma_change = np.mean(np.abs(base_chroma - generated_chroma), axis=2)
    seed = allowed & (darkness > 8.0) & (chroma_change < 24.0)
    near_object = np.asarray(
        ImageChops.subtract(expand_mask(object_mask, min(8, radius)), object_mask),
        dtype=np.uint8,
    ) >= 128
    connected_seed = np.zeros_like(seed)
    for component in _components(seed):
        ys = component // background.width
        xs = component % background.width
        if np.any(near_object[ys, xs]):
            connected_seed.flat[component] = True
    seed = connected_seed
    raw_alpha = np.clip((darkness - 8.0) / 48.0, 0.0, 1.0) * seed.astype(np.float32) * strength
    alpha_image = Image.fromarray(
        np.clip(np.rint(raw_alpha * 255.0), 0, 255).astype(np.uint8), mode="L"
    ).filter(ImageFilter.GaussianBlur(5.0))
    alpha_image = ImageChops.multiply(alpha_image, band)
    alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    ratio = np.clip(generated_luma / np.maximum(base_luma, 1.0), 0.55, 1.0)
    shadowed = base * (1.0 - alpha[..., None] * (1.0 - ratio[..., None]))
    return alpha_image, Image.fromarray(
        np.clip(np.rint(shadowed), 0, 255).astype(np.uint8), mode="RGB"
    )
