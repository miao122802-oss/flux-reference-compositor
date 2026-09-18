"""Pure-PIL preprocessing helpers for reference-guided inpainting."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


def rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def normalize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    mask = ImageOps.exif_transpose(mask).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda value: 255 if value >= 128 else 0)


def paint_mask_green(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Replace the selected pixels with the LoRA's required pure green color."""
    image = rgb(image)
    mask = normalize_mask(mask, image.size)
    green = Image.new("RGB", image.size, (0, 255, 0))
    return Image.composite(green, image, mask)


def draw_user_mask_bbox(
    generated: Image.Image,
    user_mask: Image.Image,
    *,
    expand_left: int = 0,
    expand_right: int = 0,
    expand_top: int = 0,
    expand_bottom: int = 0,
    color: tuple[int, int, int] = (255, 215, 0),
) -> Image.Image:
    """Draw the user-adjustable painted-mask bbox on a post-FLUX image."""
    generated = rgb(generated)
    user_mask = normalize_mask(user_mask, generated.size)
    box = mask_bbox_with_padding(
        user_mask,
        expand_left=expand_left,
        expand_right=expand_right,
        expand_top=expand_top,
        expand_bottom=expand_bottom,
    )
    if box is None:
        return generated.copy()
    left, top, right, bottom = box
    preview = generated.copy()
    line_width = max(3, round(min(preview.size) / 220))
    ImageDraw.Draw(preview).rectangle(
        (left, top, max(left, right - 1), max(top, bottom - 1)),
        outline=color,
        width=line_width,
    )
    return preview


def mask_bbox_with_padding(
    mask: Image.Image,
    *,
    expand_left: int = 0,
    expand_right: int = 0,
    expand_top: int = 0,
    expand_bottom: int = 0,
) -> tuple[int, int, int, int] | None:
    """Return the painted-mask bbox with independent outward padding."""
    bbox = mask.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return (
        max(0, left - max(0, int(expand_left))),
        max(0, top - max(0, int(expand_top))),
        min(mask.width, right + max(0, int(expand_right))),
        min(mask.height, bottom + max(0, int(expand_bottom))),
    )


def expand_mask(mask: Image.Image, pixels: int) -> Image.Image:
    pixels = max(0, int(pixels))
    if pixels == 0:
        return mask.copy()
    if pixels >= 12:
        # Pillow's MaxFilter cost grows with the square of the kernel width;
        # a 48 px expansion becomes a 97x97 filter and is very slow on 4K
        # masks. User masks are binary here, so a summed-area table gives the
        # identical square dilation in linear time.
        binary = np.asarray(normalize_mask(mask, mask.size), dtype=np.uint8) >= 128
        padded = np.pad(binary.astype(np.uint8), pixels, mode="constant")
        integral = np.pad(
            np.cumsum(
                np.cumsum(padded, axis=0, dtype=np.int32),
                axis=1,
                dtype=np.int32,
            ),
            ((1, 0), (1, 0)),
            mode="constant",
        )
        kernel = pixels * 2 + 1
        counts = (
            integral[kernel:, kernel:]
            - integral[:-kernel, kernel:]
            - integral[kernel:, :-kernel]
            + integral[:-kernel, :-kernel]
        )
        return Image.fromarray((counts > 0).astype(np.uint8) * 255, mode="L")
    return mask.filter(ImageFilter.MaxFilter(pixels * 2 + 1))


def feather_mask(mask: Image.Image, radius: float) -> Image.Image:
    radius = max(0.0, float(radius))
    return mask.copy() if radius == 0 else mask.filter(ImageFilter.GaussianBlur(radius))


def inward_feather_mask(mask: Image.Image, radius: float) -> Image.Image:
    """Feather an edit mask inward while keeping its outer contour at zero.

    Blurring a binary mask directly leaves roughly 50% alpha on the contour,
    which can reveal a rectangular ROI whenever the generated background has a
    small exposure difference.  Eroding before the blur moves the Gaussian
    transition fully inside the edit region.  Pixels outside the binary mask
    always remain exactly zero.
    """
    binary = normalize_mask(mask, mask.size)
    radius = max(0.0, float(radius))
    if radius == 0 or binary.getbbox() is None:
        return binary.copy()
    # Three sigmas puts the original contour below one alpha level for a
    # Gaussian kernel, so the first edited pixel is effectively pure Target.
    inset = max(1, int(round(radius * 3.0)))
    inner = shrink_mask(binary, inset)
    if inner.getbbox() is None:
        # Very thin edit regions cannot accommodate the requested transition.
        # Preserve their binary ownership instead of erasing them completely.
        return binary.copy()
    softened = inner.filter(ImageFilter.GaussianBlur(radius))
    return ImageChops.multiply(softened, binary)


def final_background_alpha(
    core_mask: Image.Image,
    old_subject_cleanup_mask: Image.Image,
    radius: float,
) -> Image.Image:
    """Build a hard edit region with an outward-only cosine transition.

    Both the user edit mask and the dilated old-subject cleanup mask are one
    continuous ownership region. Every pixel inside it stays 100% generated;
    otherwise an inward feather revives Target pixels and draws a rectangular
    cloud/wall break or the old subject contour. Softness is allowed only in a
    small ring *outside* the union, where first-pass context fades to Target.
    A blurred hard mask is not used because its first exterior pixel is near
    50% alpha, producing a thin line at the 255-to-128 ownership jump.
    """
    size = core_mask.size
    core = normalize_mask(core_mask, size).point(
        lambda value: 255 if value >= 128 else 0
    )
    old_cleanup = normalize_mask(old_subject_cleanup_mask, size).point(
        lambda value: 255 if value >= 128 else 0
    )
    radius = max(0.0, float(radius))
    effective = ImageChops.lighter(core, old_cleanup).point(
        lambda value: 255 if value >= 128 else 0
    )
    if radius <= 0 or effective.getbbox() is None:
        return effective
    # Build one-pixel outward rings. Alpha starts continuously near 255 just
    # outside the hard region and follows a raised-cosine curve to exactly 0.
    # This is an outward-only distance ramp: no Target pixel is mixed inside.
    width = max(1, int(round(radius)))
    hard = np.asarray(effective, dtype=np.uint8) >= 128
    reached = hard.copy()
    alpha = np.zeros(hard.shape, dtype=np.uint8)
    alpha[hard] = 255
    expanded_image = effective
    for distance in range(1, width + 1):
        expanded_image = expanded_image.filter(ImageFilter.MaxFilter(3))
        expanded = np.asarray(expanded_image, dtype=np.uint8) >= 128
        ring = expanded & ~reached
        if np.any(ring):
            phase = min(1.0, float(distance) / float(width))
            value = int(round(255.0 * 0.5 * (1.0 + np.cos(np.pi * phase))))
            alpha[ring] = value
        reached = expanded
    return Image.fromarray(alpha, mode="L")


def fill_mask_from_surroundings(
    image: Image.Image,
    mask: Image.Image,
    radius: float = 48.0,
) -> Image.Image:
    """Replace a masked subject with a smooth estimate from surrounding pixels.

    This is a neutral conditioning plate, not the final inpainted background.
    Normalized convolution prevents the hidden subject's own colors from
    bleeding into the estimate supplied to the consistency-edit LoRA.
    """
    image = rgb(image)
    mask = normalize_mask(mask, image.size)
    if mask.getbbox() is None:
        return image.copy()
    values = np.asarray(image, dtype=np.float32)
    hidden = np.asarray(mask, dtype=np.uint8) >= 128
    known = (~hidden).astype(np.float32)
    sigma = max(2.0, min(128.0, float(radius)))
    denominator = _gaussian_blur_float(known, sigma)
    numerator = _gaussian_blur_float(values * known[..., None], sigma)
    estimate = numerator / np.maximum(denominator[..., None], 1e-4)
    result = np.where(hidden[..., None], estimate, values)
    return Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype(np.uint8), mode="RGB"
    )


def match_reference_background_colors(
    reference: Image.Image,
    generated: Image.Image,
    background_mask: Image.Image,
    *,
    reference_valid_mask: Image.Image | None = None,
    generated_estimation_image: Image.Image | None = None,
    radius: float = 64.0,
    strength: float = 0.9,
    max_delta: float = 72.0,
) -> tuple[Image.Image, dict]:
    """Lock generated background color/exposure to a reference color field.

    Only a heavily blurred RGB difference field is transferred, so reference
    edges and subjects are not copied. `generated_estimation_image` may be a
    subject-free temporary plate used solely to estimate that field; corrected
    pixels are always based on `generated`, and the plate is never composited.
    Pixels outside `background_mask` remain byte-for-byte generated pixels.
    """
    reference = rgb(reference)
    generated = rgb(generated).resize(reference.size, Image.Resampling.LANCZOS)
    # Unlike ownership/SAM masks, this application mask is allowed to contain
    # a continuous 0..255 outer transition. Do not call normalize_mask() here:
    # it intentionally thresholds to binary and would recreate the seam.
    mask = ImageOps.exif_transpose(background_mask).convert("L")
    if mask.size != reference.size:
        mask = mask.resize(reference.size, Image.Resampling.BILINEAR)
    mask_values = np.asarray(mask, dtype=np.float32) / 255.0
    selected = mask_values > 0.0
    if not np.any(selected) or float(strength) <= 0:
        return generated.copy(), {
            "applied": False,
            "error_before": 0.0,
            "error_after": 0.0,
        }
    ref_values = np.asarray(reference, dtype=np.float32)
    gen_values = np.asarray(generated, dtype=np.float32)
    estimation_values = (
        np.asarray(
            rgb(generated_estimation_image).resize(
                reference.size, Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        if generated_estimation_image is not None
        else gen_values
    )
    sigma = max(4.0, min(256.0, float(radius)))
    if reference_valid_mask is None:
        reference_valid = np.ones(mask.size[::-1], dtype=np.float32)
    else:
        reference_valid = (
            np.asarray(
                normalize_mask(reference_valid_mask, reference.size),
                dtype=np.uint8,
            )
            >= 128
        ).astype(np.float32)
    # Estimate the Target color field exclusively from genuine, unmasked
    # Target pixels. Normalized convolution extrapolates those colors across
    # the removed old subject without ever treating its artificial fill plate
    # as real evidence.
    reference_denominator = _gaussian_blur_float(reference_valid, sigma)
    reference_numerator = _gaussian_blur_float(
        ref_values * reference_valid[..., None], sigma
    )
    ref_low = reference_numerator / np.maximum(
        reference_denominator[..., None], 1e-4
    )
    unsupported = reference_denominator <= 1e-4
    if np.any(unsupported):
        valid_samples = ref_values[reference_valid > 0.5]
        fallback = (
            np.median(valid_samples, axis=0)
            if valid_samples.size
            else np.median(ref_values.reshape(-1, 3), axis=0)
        )
        ref_low[unsupported] = fallback
    # Estimate the generated background field from a subject-free plate when
    # supplied, while applying the correction to the untouched first-pass
    # FLUX pixels. This prevents subject colors from tinting nearby background
    # without ever exposing the synthetic fill plate in the final image.
    gen_low = _gaussian_blur_float(estimation_values, sigma)
    limit = abs(float(max_delta))
    delta = np.clip(ref_low - gen_low, -limit, limit)
    amount = max(0.0, min(1.0, float(strength)))
    corrected = np.clip(gen_values + delta * amount, 0, 255)
    # A binary mask keeps the old behaviour. A soft, outward-only transition
    # mask makes the color correction decay through the exact same band as the
    # final background paste, avoiding a hard rectangular delta cutoff.
    alpha = mask_values[..., None]
    result = corrected * alpha + gen_values * (1.0 - alpha)
    measurable = selected & (reference_denominator > 1e-4)
    if not np.any(measurable):
        measurable = selected
    before = float(np.mean(np.abs(ref_low[measurable] - gen_low[measurable])))
    result_low = _gaussian_blur_float(result, sigma)
    after = float(np.mean(np.abs(ref_low[measurable] - result_low[measurable])))
    return Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype(np.uint8), mode="RGB"
    ), {
        "applied": True,
        "radius": sigma,
        "strength": amount,
        "max_delta": limit,
        "error_before": before,
        "error_after": after,
    }


def shrink_mask(mask: Image.Image, pixels: int) -> Image.Image:
    """Morphologically erode a binary mask using Pillow-only operations."""
    pixels = max(0, int(pixels))
    mask = normalize_mask(mask, mask.size)
    if pixels == 0:
        return mask.copy()
    return ImageOps.invert(expand_mask(ImageOps.invert(mask), pixels))


def _box_blur_axis(array: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Fast floating-point box blur without SciPy/OpenCV."""
    radius = max(0, int(radius))
    values = np.asarray(array, dtype=np.float32)
    if radius == 0:
        return values.copy()
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float32)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate(
        [np.zeros(zero_shape, dtype=np.float32), cumulative], axis=axis
    )
    window = radius * 2 + 1
    tail = [slice(None)] * cumulative.ndim
    head = [slice(None)] * cumulative.ndim
    tail[axis] = slice(window, None)
    head[axis] = slice(None, -window)
    return (cumulative[tuple(tail)] - cumulative[tuple(head)]) / float(window)


def _gaussian_blur_float(array: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate a Gaussian with three separable box blurs on float arrays."""
    sigma = max(0.0, float(sigma))
    values = np.asarray(array, dtype=np.float32)
    if sigma < 0.35:
        return values.copy()
    passes = 3
    ideal_width = float(np.sqrt(12.0 * sigma * sigma / passes + 1.0))
    lower_width = int(np.floor(ideal_width))
    if lower_width % 2 == 0:
        lower_width -= 1
    lower_width = max(1, lower_width)
    upper_width = lower_width + 2
    numerator = 12.0 * sigma * sigma - passes * lower_width * lower_width
    numerator -= 4.0 * passes * lower_width + 3.0 * passes
    lower_passes = int(round(numerator / (-4.0 * lower_width - 4.0)))
    lower_passes = max(0, min(passes, lower_passes))
    widths = [lower_width] * lower_passes + [upper_width] * (passes - lower_passes)
    result = values
    for width in widths:
        radius = (width - 1) // 2
        result = _box_blur_axis(result, radius, axis=1)
        result = _box_blur_axis(result, radius, axis=0)
    return result


def harmonize_roi_boundary_colors(
    source_roi: Image.Image,
    generated_roi: Image.Image,
    mask_roi: Image.Image,
    *,
    enabled: bool = True,
    luminance_only: bool = False,
    ring_width: int = 24,
    strength: float = 0.85,
    fade_radius: float = 96.0,
    interior_strength: float = 0.25,
    gain_limit: float = 1.40,
    bias_limit: float = 48.0,
) -> tuple[Image.Image, dict]:
    """Apply a spatially varying low-frequency color field at the seam.

    A single RGB gain/bias fails when one mask boundary crosses sky, clouds,
    walls and shadows. This implementation samples the *source outside* each
    local boundary segment, compares it with the generated image just inside
    that segment, and diffuses the local correction field inward. It therefore
    keeps corrections for distant materials separate and also avoids learning
    colors from the old object hidden under the edit mask.
    """
    source_roi = rgb(source_roi)
    generated_roi = rgb(generated_roi).resize(source_roi.size, Image.Resampling.LANCZOS)
    mask_roi = normalize_mask(mask_roi, source_roi.size)
    empty_info = {
        "applied": False,
        "sample_strategy": "none",
        "sample_pixels": 0,
        "gain": [1.0, 1.0, 1.0],
        "bias": [0.0, 0.0, 0.0],
        "edge_error_before": 0.0,
        "edge_error_after": 0.0,
        "field_abs_mean": 0.0,
        "field_abs_max": 0.0,
        "work_size": [0, 0],
    }
    strength = max(0.0, min(1.0, float(strength)))
    if not enabled or strength == 0 or mask_roi.getbbox() is None:
        return generated_roi.copy(), empty_info

    ring_width = max(2, min(128, int(ring_width)))
    bias_limit = max(0.0, float(bias_limit))
    if bias_limit < 0.5:
        empty_info["sample_strategy"] = "zero_bias_limit"
        return generated_roi.copy(), empty_info

    # Build the correction field at a bounded working resolution. The result is
    # deliberately low frequency, so processing it at <=640 px is both faster
    # and more stable than estimating per-pixel corrections at full resolution.
    work_scale = min(1.0, 640.0 / max(source_roi.size))
    work_size = (
        max(16, int(round(source_roi.width * work_scale))),
        max(16, int(round(source_roi.height * work_scale))),
    )
    source_work = source_roi.resize(work_size, Image.Resampling.LANCZOS)
    generated_work = generated_roi.resize(work_size, Image.Resampling.LANCZOS)
    mask_work = mask_roi.resize(work_size, Image.Resampling.NEAREST)
    ring_work = max(2.0, ring_width * work_scale)
    mask_binary = np.asarray(mask_work, dtype=np.uint8) >= 128
    boundary_soft = np.asarray(
        mask_work.filter(ImageFilter.GaussianBlur(max(0.75, ring_work / 3.0))),
        dtype=np.uint8,
    )
    inner_select = mask_binary & (boundary_soft < 254)
    outer_select = (~mask_binary) & (boundary_soft > 1)

    lowpass_radius = max(0.8, ring_work / 3.0)
    source_low = np.asarray(
        source_work.filter(ImageFilter.GaussianBlur(lowpass_radius)), dtype=np.float32
    )
    generated_low = np.asarray(
        generated_work.filter(ImageFilter.GaussianBlur(lowpass_radius)), dtype=np.float32
    )

    # Extrapolate the original background from the local outer ring to the
    # adjacent inner ring. This is crucial when source pixels under the mask are
    # an old object rather than the background we want to match.
    outer_weight = outer_select.astype(np.float32)
    outer_sigma = max(1.25, ring_work * 0.85)
    outer_denominator = _gaussian_blur_float(outer_weight, outer_sigma)
    outer_numerator = _gaussian_blur_float(
        source_low * outer_weight[..., None], outer_sigma
    )
    expected_source = outer_numerator / np.maximum(outer_denominator[..., None], 1e-4)
    valid_outer = outer_denominator > max(1e-4, float(outer_denominator.max()) * 0.002)
    sample_select = inner_select & valid_outer
    sample_strategy = "spatial_outer_boundary_field"
    if int(np.count_nonzero(sample_select)) < 64:
        # A mask touching every ROI edge has no usable outer ring. Pairing with
        # the source at the same coordinates is still preferable to skipping the
        # fix completely, though less reliable for object-replacement masks.
        expected_source = source_low
        sample_select = inner_select
        sample_strategy = "spatial_paired_inner_fallback"

    sample_count = int(np.count_nonzero(sample_select))
    if sample_count < 64:
        empty_info["sample_strategy"] = "insufficient_boundary_pixels"
        empty_info["sample_pixels"] = sample_count
        empty_info["work_size"] = list(work_size)
        return generated_roi.copy(), empty_info

    raw_delta = expected_source - generated_low
    if luminance_only:
        luma_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        luma_delta = raw_delta @ luma_weights
        raw_delta = np.repeat(luma_delta[..., None], 3, axis=2)

    # Large mismatches are usually a generated foreground touching the mask
    # boundary, not a background exposure shift. They are down-weighted before
    # diffusion so that a yellow statue cannot tint a neighboring blue sky.
    mismatch = np.max(np.abs(raw_delta), axis=2)
    confidence_scale = max(12.0, bias_limit * 1.25)
    confidence = 1.0 / (1.0 + np.power(mismatch / confidence_scale, 4.0))
    seed_weight = sample_select.astype(np.float32) * confidence.astype(np.float32)
    seed_delta = np.clip(raw_delta, -bias_limit, bias_limit)

    fade_radius = max(float(ring_width) * 1.5, float(fade_radius))
    field_sigma = max(1.5, fade_radius * work_scale)
    field_denominator = _gaussian_blur_float(seed_weight, field_sigma)
    field_numerator = _gaussian_blur_float(
        seed_delta * seed_weight[..., None], field_sigma
    )
    field = field_numerator / np.maximum(field_denominator[..., None], 1e-5)
    field = np.clip(field, -bias_limit, bias_limit)

    boundary_density = field_denominator[sample_select]
    density_peak = float(np.percentile(boundary_density, 90)) if boundary_density.size else 0.0
    influence = np.clip(field_denominator / max(density_peak, 1e-6), 0.0, 1.0)
    influence = np.sqrt(influence)
    interior_strength = max(0.0, min(1.0, float(interior_strength)))
    mask_float_work = mask_binary.astype(np.float32)
    correction_weight_work = mask_float_work * strength * (
        interior_strength + (1.0 - interior_strength) * influence
    )
    corrected_work = generated_low + field * correction_weight_work[..., None]

    # Upsample only the smooth field and its confidence; original generated
    # detail is retained at full resolution.
    encoded_field = np.clip(
        np.rint((field / (2.0 * bias_limit) + 0.5) * 255.0), 0, 255
    ).astype(np.uint8)
    field_image = Image.fromarray(encoded_field, mode="RGB").resize(
        source_roi.size, Image.Resampling.BICUBIC
    )
    field_full = (
        np.asarray(field_image, dtype=np.float32) / 255.0 - 0.5
    ) * (2.0 * bias_limit)
    influence_image = Image.fromarray(
        np.clip(np.rint(influence * 255.0), 0, 255).astype(np.uint8), mode="L"
    ).resize(source_roi.size, Image.Resampling.BILINEAR)
    influence_full = np.asarray(influence_image, dtype=np.float32) / 255.0
    mask_float = np.asarray(mask_roi, dtype=np.float32) / 255.0
    correction_weight = mask_float * strength * (
        interior_strength + (1.0 - interior_strength) * influence_full
    )
    generated_float = np.asarray(generated_roi, dtype=np.float32)
    corrected = generated_float + field_full * correction_weight[..., None]
    corrected_image = Image.fromarray(
        np.clip(np.rint(corrected), 0, 255).astype(np.uint8), mode="RGB"
    )

    expected_samples = expected_source[sample_select]
    generated_samples = generated_low[sample_select]
    corrected_samples = corrected_work[sample_select]
    edge_error_before = float(np.mean(np.abs(expected_samples - generated_samples)))
    edge_error_after = float(np.mean(np.abs(expected_samples - corrected_samples)))
    median_bias = np.median(seed_delta[sample_select], axis=0).astype(np.float32)
    info = {
        "applied": True,
        "sample_strategy": sample_strategy,
        "sample_pixels": sample_count,
        "gain": [1.0, 1.0, 1.0],
        "bias": [float(value) for value in median_bias],
        "edge_error_before": edge_error_before,
        "edge_error_after": edge_error_after,
        "field_abs_mean": float(np.mean(np.abs(field[sample_select]))),
        "field_abs_max": float(np.max(np.abs(field[sample_select]))),
        "work_size": list(work_size),
        "spatial_radius": float(fade_radius),
    }
    return corrected_image, info


def mask_bbox_with_context(
    mask: Image.Image,
    context_ratio: float = 0.45,
    min_context: int = 48,
    *,
    expand_left: int = 0,
    expand_right: int = 0,
    expand_top: int = 0,
    expand_bottom: int = 0,
) -> tuple[int, int, int, int]:
    bbox = mask_bbox_with_padding(
        mask,
        expand_left=expand_left,
        expand_right=expand_right,
        expand_top=expand_top,
        expand_bottom=expand_bottom,
    )
    if bbox is None:
        raise ValueError("Mask is empty.")
    left, top, right, bottom = bbox
    span = max(right - left, bottom - top)
    padding = max(int(min_context), int(round(span * max(0.0, context_ratio))))
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(mask.width, right + padding),
        min(mask.height, bottom + padding),
    )


def aligned_size(
    size: tuple[int, int], max_side: int = 1024, multiple: int = 16
) -> tuple[int, int]:
    width, height = size
    max_side = max(multiple, int(max_side))
    scale = min(1.0, max_side / max(width, height))
    width = max(multiple, int(round(width * scale / multiple)) * multiple)
    height = max(multiple, int(round(height * scale / multiple)) * multiple)
    return width, height


def resize_pair_for_model(
    image: Image.Image,
    mask: Image.Image,
    max_side: int,
    multiple: int = 16,
    *,
    binary_mask: bool = False,
) -> tuple[Image.Image, Image.Image]:
    size = aligned_size(image.size, max_side=max_side, multiple=multiple)
    if image.size == size:
        resized_mask = mask.copy()
    else:
        mask_resampling = (
            Image.Resampling.NEAREST if binary_mask else Image.Resampling.BILINEAR
        )
        resized_mask = mask.resize(size, mask_resampling)
    if binary_mask:
        resized_mask = normalize_mask(resized_mask, size)
    resized_image = image.copy() if image.size == size else image.resize(
        size, Image.Resampling.LANCZOS
    )
    return resized_image, resized_mask


def limit_reference(
    image: Image.Image, max_side: int = 1024, multiple: int = 16
) -> Image.Image:
    size = aligned_size(image.size, max_side=max_side, multiple=multiple)
    return image.copy() if size == image.size else image.resize(size, Image.Resampling.LANCZOS)


def crop_reference_cutout(
    image: Image.Image,
    mask: Image.Image,
    padding_ratio: float = 0.10,
    feather_radius: float = 0.0,
) -> Image.Image:
    image = rgb(image)
    mask = normalize_mask(mask, image.size)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Reference mask is empty.")
    left, top, right, bottom = bbox
    padding = int(round(max(right - left, bottom - top) * max(0.0, padding_ratio)))
    box = (
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding),
        min(image.height, bottom + padding),
    )
    foreground = image.crop(box)
    alpha = feather_mask(mask.crop(box), feather_radius)
    return Image.composite(foreground, Image.new("RGB", foreground.size, "white"), alpha)


def despill_green_boundary(
    source_roi: Image.Image,
    generated_roi: Image.Image,
    mask_roi: Image.Image,
    *,
    enabled: bool = True,
    width: int = 10,
    strength: float = 0.9,
    threshold: float = 6.0,
) -> tuple[Image.Image, dict]:
    """Remove residual pure-green LoRA input color from the inner mask edge."""
    source_roi = rgb(source_roi)
    generated_roi = rgb(generated_roi).resize(source_roi.size, Image.Resampling.LANCZOS)
    mask_roi = normalize_mask(mask_roi, source_roi.size)
    info = {"applied": False, "pixels": 0, "mean_excess": 0.0}
    width = max(1, min(48, int(width)))
    strength = max(0.0, min(1.0, float(strength)))
    if not enabled or strength == 0 or mask_roi.getbbox() is None:
        return generated_roi.copy(), info

    mask_binary = np.asarray(mask_roi, dtype=np.uint8) >= 128
    boundary_soft = np.asarray(
        mask_roi.filter(ImageFilter.GaussianBlur(max(0.75, width / 3.0))),
        dtype=np.uint8,
    )
    inner_boundary = mask_binary & (boundary_soft < 254)
    generated = np.asarray(generated_roi, dtype=np.float32)
    source = np.asarray(
        source_roi.filter(ImageFilter.GaussianBlur(max(1.0, width / 2.0))),
        dtype=np.float32,
    )
    generated_excess = generated[..., 1] - np.maximum(generated[..., 0], generated[..., 2])
    source_excess = source[..., 1] - np.maximum(source[..., 0], source[..., 2])
    spill_excess = np.maximum(0.0, generated_excess - np.maximum(0.0, source_excess))
    activation = np.clip((spill_excess - float(threshold)) / 32.0, 0.0, 1.0)
    weight = activation * inner_boundary.astype(np.float32) * strength
    selected = weight > 0.01
    if int(np.count_nonzero(selected)) == 0:
        return generated_roi.copy(), info

    corrected = generated * (1.0 - weight[..., None]) + source * weight[..., None]
    corrected_image = Image.fromarray(
        np.clip(np.rint(corrected), 0, 255).astype(np.uint8), mode="RGB"
    )
    info = {
        "applied": True,
        "pixels": int(np.count_nonzero(selected)),
        "mean_excess": float(np.mean(spill_excess[selected])),
    }
    return corrected_image, info


def despill_green_seam_band(
    source_roi: Image.Image,
    generated_roi: Image.Image,
    mask_roi: Image.Image,
    *,
    enabled: bool = True,
    width: int = 20,
    strength: float = 1.0,
    threshold: float = 8.0,
) -> tuple[Image.Image, dict]:
    """Remove thin green-key or complementary magenta residue at the seam."""
    source_roi = rgb(source_roi)
    generated_roi = rgb(generated_roi).resize(source_roi.size, Image.Resampling.LANCZOS)
    core = normalize_mask(mask_roi, source_roi.size).point(
        lambda value: 255 if value >= 128 else 0
    )
    info = {"applied": False, "pixels": 0, "mean_excess": 0.0}
    width = max(2, min(64, int(width)))
    strength = max(0.0, min(1.0, float(strength)))
    if not enabled or strength == 0 or core.getbbox() is None:
        return generated_roi.copy(), info

    outer = np.asarray(expand_mask(core, width), dtype=np.uint8) >= 128
    mask_binary = np.asarray(core, dtype=np.uint8) >= 128
    eroded = ImageOps.invert(
        expand_mask(ImageOps.invert(core), max(2, width // 2))
    )
    seam_band = outer & ~(np.asarray(eroded, dtype=np.uint8) >= 128)
    generated = np.asarray(generated_roi, dtype=np.float32)
    local = np.asarray(generated_roi.filter(ImageFilter.MedianFilter(5)), dtype=np.float32)
    source = np.asarray(
        source_roi.filter(ImageFilter.GaussianBlur(max(1.0, width / 6.0))),
        dtype=np.float32,
    )
    generated_excess = generated[..., 1] - np.maximum(generated[..., 0], generated[..., 2])
    source_excess = source[..., 1] - np.maximum(source[..., 0], source[..., 2])
    spill_excess = np.maximum(0.0, generated_excess - np.maximum(0.0, source_excess))
    # A green-screen fringe is not always green in the decoded result. The
    # model and later color matching can turn it into a thin complementary
    # magenta/purple line. Detect both directions of the green-vs-red/blue
    # chroma axis, but only when they are stronger than the source and the
    # local 5x5 median. That leaves broad, intentional subject colors alone.
    key_axis = generated[..., 1] - 0.5 * (generated[..., 0] + generated[..., 2])
    source_axis = source[..., 1] - 0.5 * (source[..., 0] + source[..., 2])
    local_axis = local[..., 1] - 0.5 * (local[..., 0] + local[..., 2])
    key_spike = np.maximum(
        0.0,
        np.abs(key_axis) - np.maximum(np.abs(source_axis), np.abs(local_axis)),
    )
    green_activation = np.clip((spill_excess - float(threshold)) / 24.0, 0.0, 1.0)
    key_activation = np.clip((key_spike - float(threshold)) / 20.0, 0.0, 1.0)
    activation = np.maximum(green_activation, key_activation)
    weight = np.sqrt(activation) * seam_band.astype(np.float32) * strength
    selected = weight > 0.01
    if int(np.count_nonzero(selected)) == 0:
        return generated_roi.copy(), info

    # Inside the requested edit area, correct only the green-vs-magenta chroma
    # axis toward the local median. Keeping the RGB sum stable preserves edge
    # luminance and fine texture instead of replacing the contour with a
    # spatially smoothed pixel. Outside the Mask, the original Target remains
    # the safest repair source and cannot reveal an edited subject interior.
    chroma_delta = local_axis - key_axis
    chroma_repair = generated.copy()
    chroma_repair[..., 0] -= chroma_delta / 3.0
    chroma_repair[..., 1] += chroma_delta * (2.0 / 3.0)
    chroma_repair[..., 2] -= chroma_delta / 3.0
    repair = np.where(mask_binary[..., None], chroma_repair, source)
    corrected = generated * (1.0 - weight[..., None]) + repair * weight[..., None]
    corrected_image = Image.fromarray(
        np.clip(np.rint(corrected), 0, 255).astype(np.uint8), mode="RGB"
    )
    info = {
        "applied": True,
        "pixels": int(np.count_nonzero(selected)),
        "mean_excess": float(np.mean(spill_excess[selected])),
    }
    return corrected_image, info


def _resize_rgb_array(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(
        np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB"
    )
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def _rgb_gaussian_pyramid(image: Image.Image, levels: int) -> list[np.ndarray]:
    current = rgb(image)
    pyramid = [np.asarray(current, dtype=np.float32)]
    for _ in range(max(0, int(levels))):
        if min(current.size) <= 16:
            break
        next_size = (max(1, (current.width + 1) // 2), max(1, (current.height + 1) // 2))
        current = current.filter(ImageFilter.GaussianBlur(1.0)).resize(
            next_size, Image.Resampling.LANCZOS
        )
        pyramid.append(np.asarray(current, dtype=np.float32))
    return pyramid


def _laplacian_pyramid(gaussian: list[np.ndarray]) -> list[np.ndarray]:
    layers: list[np.ndarray] = []
    for index in range(len(gaussian) - 1):
        height, width = gaussian[index].shape[:2]
        expanded = _resize_rgb_array(gaussian[index + 1], (width, height))
        layers.append(gaussian[index] - expanded)
    layers.append(gaussian[-1])
    return layers


def _multiband_blend_roi(
    source_roi: Image.Image,
    generated_roi: Image.Image,
    allowed: Image.Image,
    *,
    levels: int,
    background_similarity_threshold: float = 32.0,
    background_restore_strength: float = 0.85,
) -> Image.Image:
    """Restore source-like background while preserving visibly new objects."""
    source_roi = rgb(source_roi)
    generated_roi = rgb(generated_roi).resize(source_roi.size, Image.Resampling.LANCZOS)
    allowed = normalize_mask(allowed, source_roi.size)
    levels = max(1, min(7, int(levels)))

    allowed_float = np.asarray(allowed, dtype=np.float32) / 255.0
    # Combine direct color difference with a low-frequency comparison. A real
    # generated object normally differs clearly from the old target and remains
    # opaque; generated sky/wall/cloud pixels close to the source are restored
    # toward the original background instead of hiding the seam with feathering.
    protect_radius = max(2.0, float(2 ** max(1, levels - 2)))
    source_array = np.asarray(source_roi, dtype=np.float32)
    generated_array = np.asarray(generated_roi, dtype=np.float32)
    source_low = np.asarray(
        source_roi.filter(ImageFilter.GaussianBlur(protect_radius)), dtype=np.float32
    )
    generated_low = np.asarray(
        generated_roi.filter(ImageFilter.GaussianBlur(protect_radius)), dtype=np.float32
    )
    direct_difference = np.mean(np.abs(generated_array - source_array), axis=2)
    low_difference = np.mean(np.abs(generated_low - source_low), axis=2)
    difference = np.maximum(direct_difference, low_difference)
    threshold = max(1.0, float(background_similarity_threshold))
    transition = max(8.0, threshold)
    protection = np.clip((difference - threshold) / transition, 0.0, 1.0)
    protection = protection * protection * (3.0 - 2.0 * protection)
    restore_strength = max(0.0, min(1.0, float(background_restore_strength)))
    generated_share = 1.0 - restore_strength * (1.0 - protection)
    adaptive_alpha = allowed_float * generated_share
    alpha_current = Image.fromarray(
        np.clip(np.rint(adaptive_alpha * 255.0), 0, 255).astype(np.uint8), mode="L"
    )

    source_gaussian = _rgb_gaussian_pyramid(source_roi, levels)
    generated_gaussian = _rgb_gaussian_pyramid(generated_roi, len(source_gaussian) - 1)
    source_laplacian = _laplacian_pyramid(source_gaussian)
    generated_laplacian = _laplacian_pyramid(generated_gaussian)

    alpha_pyramid: list[np.ndarray] = []
    for layer in source_gaussian:
        height, width = layer.shape[:2]
        if alpha_current.size != (width, height):
            alpha_current = alpha_current.filter(ImageFilter.GaussianBlur(1.0)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
        alpha_pyramid.append(np.asarray(alpha_current, dtype=np.float32) / 255.0)

    blended_layers = []
    for source_layer, generated_layer, alpha in zip(
        source_laplacian, generated_laplacian, alpha_pyramid
    ):
        blended_layers.append(
            generated_layer * alpha[..., None] + source_layer * (1.0 - alpha[..., None])
        )

    reconstructed = blended_layers[-1]
    for index in range(len(blended_layers) - 2, -1, -1):
        height, width = blended_layers[index].shape[:2]
        reconstructed = _resize_rgb_array(reconstructed, (width, height)) + blended_layers[index]

    # No final edge feather: restore all disallowed pixels exactly and let color
    # similarity decide how much generated background remains inside the Mask.
    reconstructed = np.where(
        allowed_float[..., None] > 0.5, reconstructed, source_array
    )
    return Image.fromarray(
        np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8), mode="RGB"
    )


def composite_roi(
    original: Image.Image,
    generated_roi: Image.Image,
    core_mask: Image.Image,
    roi_box: tuple[int, int, int, int],
    blend_expand: int = 0,
    texture_blend_enabled: bool = True,
    texture_blend_levels: int = 4,
    background_similarity_threshold: float = 32.0,
    background_restore_strength: float = 0.85,
) -> Image.Image:
    original = rgb(original)
    roi_size = (roi_box[2] - roi_box[0], roi_box[3] - roi_box[1])
    generated_roi = rgb(generated_roi).resize(roi_size, Image.Resampling.LANCZOS)
    source_roi = original.crop(roi_box)
    mask_roi = normalize_mask(core_mask, original.size).crop(roi_box)
    allowed = expand_mask(mask_roi, blend_expand)
    if texture_blend_enabled and int(texture_blend_levels) > 0:
        blended = _multiband_blend_roi(
            source_roi,
            generated_roi,
            allowed,
            levels=texture_blend_levels,
            background_similarity_threshold=background_similarity_threshold,
            background_restore_strength=background_restore_strength,
        )
    else:
        blended = Image.composite(generated_roi, source_roi, allowed)
    output = original.copy()
    output.paste(blended, roi_box[:2])
    return output


def composite_roi_with_outward_feather(
    original: Image.Image,
    generated_roi: Image.Image,
    core_mask: Image.Image,
    roi_box: tuple[int, int, int, int],
    expand_pixels: int = 48,
    feather_pixels: float = 32.0,
    hard_protect_pixels: int = 8,
    seam_color_match: bool = True,
    seam_color_strength: float = 0.9,
    green_cleanup: bool = True,
    green_cleanup_width: int = 20,
) -> Image.Image:
    """Match the generated edge background, then feather only the outer ring."""
    original = rgb(original)
    roi_size = (roi_box[2] - roi_box[0], roi_box[3] - roi_box[1])
    generated_roi = rgb(generated_roi).resize(roi_size, Image.Resampling.LANCZOS)
    source_roi = original.crop(roi_box)
    core = normalize_mask(core_mask, original.size).point(
        lambda value: 255 if value >= 128 else 0
    )
    expand = max(0, int(expand_pixels))
    feather = min(float(expand), max(0.0, float(feather_pixels)))
    limit = expand_mask(core, expand)
    hard_protect = expand_mask(core, min(expand, max(0, int(hard_protect_pixels))))
    generated_roi, _ = despill_green_seam_band(
        source_roi,
        generated_roi,
        core.crop(roi_box),
        enabled=green_cleanup,
        width=green_cleanup_width,
    )
    if seam_color_match and expand > 0:
        raw_generated_roi = generated_roi.copy()
        color_matched_roi, _ = harmonize_roi_boundary_colors(
            source_roi,
            generated_roi,
            limit.crop(roi_box),
            enabled=True,
            ring_width=max(12, min(96, int(round(max(feather, 8.0) * 1.5)))),
            strength=max(0.0, min(1.0, float(seam_color_strength))),
            fade_radius=max(48.0, float(expand) * 2.0),
            interior_strength=0.0,
            bias_limit=48.0,
        )
        # Color matching belongs to the generated background ring, never the
        # protected subject core. Ramp it in away from the core so there is no
        # second hard boundary at the original user mask.
        protect_expand = min(expand, max(1, int(round(max(feather, 2.0) * 0.5))))
        protect = expand_mask(core, protect_expand).filter(
            ImageFilter.GaussianBlur(max(1.0, feather * 0.5))
        )
        correction_alpha = ImageChops.subtract(limit, protect).crop(roi_box)
        generated_roi = Image.composite(
            color_matched_roi,
            raw_generated_roi,
            correction_alpha,
        )
        generated_roi = Image.composite(
            raw_generated_roi,
            generated_roi,
            hard_protect.crop(roi_box),
        )
    if feather > 0:
        solid_expand = max(0, int(round(expand - feather)))
        alpha = expand_mask(core, solid_expand).filter(
            ImageFilter.GaussianBlur(max(0.5, feather / 2.0))
        )
        alpha = ImageChops.multiply(alpha, limit)
        alpha = ImageChops.lighter(alpha, hard_protect)
    else:
        alpha = limit
    alpha_roi = alpha.crop(roi_box)
    blended = Image.composite(generated_roi, source_roi, alpha_roi)
    output = original.copy()
    output.paste(blended, roi_box[:2])
    return output


def bbox_text(box: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in box)
