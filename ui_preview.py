"""Small, file-backed previews for the Base64-only Gradio 3 image outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps


DEFAULT_PREVIEW_MAX_SIDE = 1280


def preview_size(
    size: tuple[int, int], max_side: int = DEFAULT_PREVIEW_MAX_SIDE
) -> tuple[int, int]:
    width, height = (max(1, int(size[0])), max(1, int(size[1])))
    max_side = max(64, int(max_side))
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def display_preview(
    image: Image.Image,
    max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
    *,
    mask: bool = False,
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L" if mask else "RGB")
    size = preview_size(image.size, max_side)
    if size == image.size:
        return image.copy()
    resampling = Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
    return image.resize(size, resampling)


def display_point_to_full(
    point: Sequence[float],
    full_size: tuple[int, int],
    display_size: tuple[int, int],
) -> list[float]:
    """Map a Gradio click on a reduced preview back to full image pixels."""
    if len(point) < 2:
        raise ValueError("A point must contain x and y coordinates.")
    full_width, full_height = full_size
    display_width, display_height = display_size
    x_scale = (full_width - 1) / max(1, display_width - 1)
    y_scale = (full_height - 1) / max(1, display_height - 1)
    x = max(0.0, min(float(full_width - 1), float(point[0]) * x_scale))
    y = max(0.0, min(float(full_height - 1), float(point[1]) * y_scale))
    return [x, y]


def save_preview(
    image: Image.Image,
    path: str | Path,
    *,
    max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
    mask: bool = False,
) -> str:
    """Save a fast display asset; Gradio 3 preserves the file's compact bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = display_preview(image, max_side=max_side, mask=mask)
    if mask:
        path = path.with_suffix(".png")
        preview.save(path, format="PNG", compress_level=1)
    else:
        path = path.with_suffix(".jpg")
        preview.save(
            path,
            format="JPEG",
            quality=88,
            subsampling=1,
            optimize=False,
        )
    return str(path)


def save_full_png(image: Image.Image, path: str | Path) -> str:
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    ImageOps.exif_transpose(image).convert("RGB").save(
        path,
        format="PNG",
        compress_level=2,
    )
    return str(path)
