"""SAM1/SAM2 point-prompt segmentation for extracting a reference object."""

from __future__ import annotations

import gc
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from image_processing import crop_reference_cutout, normalize_mask, rgb


DEFAULT_SAM2_PATH = os.getenv("SAM2_MODEL_PATH", "models/sam2/sam2-hiera-large")
DEFAULT_SAM1_PATH = os.getenv("SAM1_CHECKPOINT_PATH", "models/sam/sam_vit_h_4b8939.pth")

_SEGMENTER: Any | None = None
_SEGMENTER_KIND: str | None = None
_SEGMENTER_KEY: tuple[str, str, str] | None = None
_CURRENT_IMAGE_KEY: str | None = None
_LOCK = threading.Lock()


def _empty_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_segmenter() -> None:
    global _SEGMENTER, _SEGMENTER_KIND, _SEGMENTER_KEY, _CURRENT_IMAGE_KEY
    with _LOCK:
        _SEGMENTER = _SEGMENTER_KIND = _SEGMENTER_KEY = _CURRENT_IMAGE_KEY = None
    _empty_cuda_cache()


def _image_key(image: Image.Image) -> str:
    thumb = rgb(image).resize((64, 64), Image.Resampling.BILINEAR)
    return hashlib.sha1(thumb.tobytes() + str(image.size).encode()).hexdigest()


def _find_file(folder: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _build_local_sam2(model_dir: Path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.build_sam import _load_checkpoint
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = _find_file(model_dir, ("*.pt", "*.pth"))
    config = _find_file(model_dir, ("*.yaml", "*.yml"))
    if checkpoint is None or config is None:
        raise FileNotFoundError(f"{model_dir} must contain both a SAM2 YAML config and .pt/.pth weights.")
    started = time.perf_counter()
    print(f"[SAM2] config={config}", flush=True)
    print(f"[SAM2] checkpoint={checkpoint}", flush=True)
    overrides = [
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config.parent.resolve())):
        cfg = compose(config_name=config.stem, overrides=overrides)
    OmegaConf.resolve(cfg)
    print("[SAM2] Building the model...", flush=True)
    model = instantiate(cfg.model, _recursive_=True)
    print("[SAM2] Reading checkpoint...", flush=True)
    _load_checkpoint(model, str(checkpoint.resolve()))
    print("[SAM2] Moving model to CUDA...", flush=True)
    model.to("cuda").eval()
    print(f"[SAM2] Model ready in {time.perf_counter() - started:.1f}s", flush=True)
    return SAM2ImagePredictor(model)


def load_segmenter(backend: str, sam2_path: str, sam1_path: str):
    global _SEGMENTER, _SEGMENTER_KIND, _SEGMENTER_KEY, _CURRENT_IMAGE_KEY
    backend = (backend or "auto").lower()
    key = (backend, sam2_path.strip(), sam1_path.strip())
    with _LOCK:
        if _SEGMENTER is not None and _SEGMENTER_KEY == key:
            print(f"[SAM] Using cached {_SEGMENTER_KIND}", flush=True)
            return _SEGMENTER, _SEGMENTER_KIND
        _SEGMENTER = _SEGMENTER_KIND = _SEGMENTER_KEY = _CURRENT_IMAGE_KEY = None
        _empty_cuda_cache()
        errors: list[str] = []
        if backend in {"auto", "sam2"}:
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                path = Path(sam2_path).expanduser()
                print(f"[SAM] Loading SAM2: {sam2_path}", flush=True)
                _SEGMENTER = _build_local_sam2(path) if path.is_dir() else SAM2ImagePredictor.from_pretrained(sam2_path, device="cuda")
                _SEGMENTER_KIND = "sam2"
            except Exception as exc:
                print(f"[SAM] SAM2 loading failed: {exc}", flush=True)
                errors.append(f"SAM2: {exc}")
                if backend == "sam2":
                    raise RuntimeError("；".join(errors)) from exc
        if _SEGMENTER is None and backend in {"auto", "sam1"}:
            try:
                from segment_anything import SamPredictor, sam_model_registry

                print(f"[SAM] Loading SAM1: {sam1_path}", flush=True)
                started = time.perf_counter()
                sam = sam_model_registry["vit_h"](checkpoint=sam1_path)
                sam.to(device="cuda").eval()
                _SEGMENTER, _SEGMENTER_KIND = SamPredictor(sam), "sam1"
                print(f"[SAM] SAM1 ready in {time.perf_counter() - started:.1f}s", flush=True)
            except Exception as exc:
                errors.append(f"SAM1: {exc}")
                raise RuntimeError("；".join(errors)) from exc
        if _SEGMENTER is None:
            raise ValueError(f"Unknown segmentation backend: {backend}")
        _SEGMENTER_KEY = key
        return _SEGMENTER, _SEGMENTER_KIND


def segment_reference(
    image: Image.Image,
    points: Sequence[Sequence[float]],
    labels: Sequence[int],
    backend: str = "auto",
    sam2_path: str = DEFAULT_SAM2_PATH,
    sam1_path: str = DEFAULT_SAM1_PATH,
) -> tuple[Image.Image, float, str]:
    if not points or not any(int(label) == 1 for label in labels):
        raise ValueError("At least one foreground point is required.")
    positive = [point for point, label in zip(points, labels) if int(label) == 1]
    negative = [point for point, label in zip(points, labels) if int(label) == 0]
    prediction = segment_with_prompts(
        image,
        positive_points=positive,
        negative_points=negative,
        backend=backend,
        sam2_path=sam2_path,
        sam1_path=sam1_path,
    )
    best = int(np.argmax(prediction["scores"]))
    return prediction["masks"][best], float(prediction["scores"][best]), str(prediction["kind"])


def segment_with_prompts(
    image: Image.Image,
    *,
    box_xyxy: Sequence[float] | None = None,
    positive_points: Sequence[Sequence[float]] = (),
    negative_points: Sequence[Sequence[float]] = (),
    previous_mask_logits: np.ndarray | None = None,
    backend: str = "auto",
    sam2_path: str = DEFAULT_SAM2_PATH,
    sam1_path: str = DEFAULT_SAM1_PATH,
) -> dict[str, Any]:
    """Run SAM1/SAM2 with an optional box, positive/negative points and logits."""
    global _CURRENT_IMAGE_KEY
    if box_xyxy is None and not positive_points:
        raise ValueError("SAM requires at least one foreground point or a bounding box.")
    predictor, kind = load_segmenter(backend, sam2_path, sam1_path)
    image = rgb(image)
    key = _image_key(image)
    points = [list(item) for item in positive_points] + [list(item) for item in negative_points]
    labels = [1] * len(positive_points) + [0] * len(negative_points)
    coords = np.asarray(points, dtype=np.float32).reshape(-1, 2) if points else None
    point_labels = np.asarray(labels, dtype=np.int32).reshape(-1) if labels else None
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(4) if box_xyxy is not None else None
    mask_input = None
    if previous_mask_logits is not None:
        mask_input = np.asarray(previous_mask_logits, dtype=np.float32)
        if mask_input.ndim == 2:
            mask_input = mask_input[None, ...]
    import torch

    if key != _CURRENT_IMAGE_KEY:
        print(f"[SAM] Encoding image: {image.width}x{image.height}", flush=True)
        encode_started = time.perf_counter()
        image_array = np.array(image, dtype=np.uint8, copy=True)
        with torch.inference_mode():
            predictor.set_image(image_array)
        _CURRENT_IMAGE_KEY = key
        print(f"[SAM] Image encoded in {time.perf_counter() - encode_started:.1f}s", flush=True)
    point_count = 0 if coords is None else len(coords)
    print(
        f"[SAM] Predicting mask with box={'ON' if box is not None else 'OFF'}, "
        f"{point_count} prompt points...",
        flush=True,
    )
    predict_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    ):
        masks, scores, low_res_logits = predictor.predict(
            point_coords=coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=True,
        )
    print(f"[SAM] Mask predicted in {time.perf_counter() - predict_started:.2f}s", flush=True)
    mask_images = [
        normalize_mask(
            Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L"),
            image.size,
        )
        for mask in masks
    ]
    return {
        "masks": mask_images,
        "scores": [float(value) for value in np.asarray(scores).reshape(-1)],
        "logits": [np.asarray(value, dtype=np.float32) for value in np.asarray(low_res_logits)],
        "kind": str(kind),
    }


def make_reference_cutout(
    image: Image.Image,
    mask: Image.Image,
    padding_ratio: float = 0.10,
    feather_radius: float = 0.0,
) -> Image.Image:
    return crop_reference_cutout(image, mask, padding_ratio, feather_radius)


def draw_points(
    image: Image.Image, points: Sequence[Sequence[float]], labels: Sequence[int]
) -> Image.Image:
    preview = rgb(image)
    draw = ImageDraw.Draw(preview)
    radius = max(6, round(min(preview.size) / 90))
    for (x, y), label in zip(points, labels):
        color = "#20c866" if int(label) == 1 else "#ed3f3f"
        x, y = int(x), int(y)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=3)
    return preview


def segmentation_preview(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = rgb(image)
    overlay = Image.new("RGB", image.size, (25, 205, 95))
    alpha = normalize_mask(mask, image.size).point(lambda value: int(value * 0.38))
    return Image.composite(overlay, image, alpha)
