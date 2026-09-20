"""FLUX Reference Compositor: reference-guided local image replacement."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import html
import importlib
import inspect
import io
from pathlib import Path
import shutil
import threading
import time
import traceback
import uuid

import gradio as gr
from PIL import Image, ImageChops, ImageOps

import flux_inpaint
import object_segmenter
import sam_segmenter
import ui_preview
from image_processing import (
    composite_roi_with_outward_feather,
    draw_user_mask_bbox,
)
from mask_editor import EDITOR_CSS, EDITOR_HTML, MASK_EDITOR_JS


_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flux2-klein")
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

TARGET_DRAW_MODE = "Draw Mask"
TARGET_SAM_MODE = "Segment Target with SAM"


def _generation_active() -> bool:
    with _JOBS_LOCK:
        return any(job.get("state") in {"queued", "running"} for job in _JOBS.values())


def _update_job(job_id: str, *, message: str | None = None, progress: float | None = None, **values) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        if message is not None:
            job["message"] = message
        if progress is not None:
            job["progress"] = max(0.0, min(100.0, float(progress)))
        job.update(values)


def _prune_finished_jobs(keep: int = 10) -> None:
    removed = []
    with _JOBS_LOCK:
        finished = [
            key for key, value in _JOBS.items()
            if value.get("state") in {"completed", "failed"}
        ]
        for old_id in finished[:-max(1, int(keep))]:
            removed.append(_JOBS.pop(old_id, None))
    if removed:
        output_root = _OUTPUT_ROOT.resolve()
        for job in removed:
            if not job:
                continue
            artifact_dir = job.get("artifact_dir")
            if not artifact_dir:
                continue
            candidate = Path(artifact_dir).resolve()
            if candidate.parent == output_root:
                shutil.rmtree(candidate, ignore_errors=True)
        removed.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _progress_html(progress: float, message: str, state: str = "running") -> str:
    value = max(0.0, min(100.0, float(progress)))
    safe_message = html.escape(message or "Waiting for task status...")
    state_class = " failed" if state == "failed" else (" completed" if state == "completed" else "")
    return (
        f'<div class="generation-progress{state_class}">'
        f'<div class="generation-progress-head"><span>{safe_message}</span><strong>{value:.0f}%</strong></div>'
        f'<div class="generation-progress-track"><div class="generation-progress-fill" '
        f'style="width:{value:.2f}%"></div></div></div>'
    )


def _metrics_html(metrics: dict | None = None) -> str:
    if not metrics:
        return (
            '<div class="task-metrics empty"><div class="task-metrics-title">Task Metrics</div>'
            '<div class="task-metrics-empty">Image sizes, scale, timing, and peak GPU memory appear after generation.</div></div>'
        )

    def size(name: str) -> str:
        width, height = metrics[name]
        return f"{int(width)} × {int(height)}"

    scale = f"{metrics['output_scale_x']:.3f}× / {metrics['output_scale_y']:.3f}×"
    cards = [
        ("Target image", size("source_input_size"), "Input dimensions"),
        ("Reference", size("reference_input_size"), f"Model input: {size('reference_model_size') }"),
        ("FLUX ROI", size("model_input_size"), f"Original ROI: {size('roi_source_size')}"),
        ("Final output", size("output_size"), f"Width/height scale: {scale}"),
        ("Total elapsed time", f"{metrics['end_to_end_seconds']:.2f} s", f"FLUX processing: {metrics['flux_total_seconds']:.2f} s"),
        ("Peak GPU memory", f"{metrics['peak_reserved_gib']:.2f} GiB", f"allocated {metrics['peak_allocated_gib']:.2f} GiB"),
    ]
    card_html = "".join(
        f'<div class="metric-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>'
        f'<small>{html.escape(note)}</small></div>'
        for label, value, note in cards
    )
    breakdown_parts = [
        f"Preprocessing: {metrics['preprocess_seconds']:.2f}s",
        f"Model preparation: {metrics['pipeline_prepare_seconds']:.2f}s",
    ]
    if metrics.get("background_diffusion_seconds", 0.0) > 0:
        breakdown_parts.append(
            f"FLUX background: {metrics['background_diffusion_seconds']:.2f}s"
        )
    breakdown_parts.extend(
        [
            f"FLUX subject: {metrics.get('object_diffusion_seconds', metrics['diffusion_seconds']):.2f}s",
            f"Resize and composite: {metrics['postprocess_seconds']:.2f}s",
        ]
    )
    if metrics.get("segmentation_enabled", False):
        breakdown_parts.append(
            f"Subject segmentation: {metrics.get('segmentation_seconds', 0.0):.2f}s"
        )
    else:
        breakdown_parts.append("Waiting for generated subject selection")
    breakdown = "　 | 　".join(breakdown_parts)
    return (
        '<div class="task-metrics"><div class="task-metrics-title">Task Metrics</div>'
        f'<div class="task-metrics-grid">{card_html}</div>'
        f'<div class="metrics-breakdown">{html.escape(breakdown)}</div></div>'
    )


def decode_data_url(data: str, name: str) -> Image.Image:
    if not data:
        raise gr.Error(f"Please provide {name} first.")
    try:
        encoded = data.split(",", 1)[1] if "," in data else data
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            return ImageOps.exif_transpose(image).copy()
    except Exception as exc:
        raise gr.Error(f"Cannot read {name}: {exc}") from exc


def _display(image: Image.Image | None, *, mask: bool = False):
    if image is None:
        return None
    return ui_preview.display_preview(image, mask=mask)


def _display_click_to_full(index, image: Image.Image) -> list[float]:
    size = ui_preview.preview_size(image.size)
    return ui_preview.display_point_to_full(index, image.size, size)


def reset_reference(image):
    if image is None:
        return None, [], [], None, None, None, None, "Waiting for reference image."
    image = ImageOps.exif_transpose(image).convert("RGB")
    return (
        image,
        [],
        [],
        None,
        None,
        _display(image),
        None,
        "✅ Reference loaded. Add one or more green foreground points to the center image.",
    )


def _predict_reference(
    image,
    points,
    labels,
    backend,
    sam2_path,
    sam1_path,
    ref_padding,
    ref_feather,
):
    if _generation_active():
        raise RuntimeError("FLUX is generating. Wait for the current task before loading SAM.")
    point_preview = sam_segmenter.draw_points(image, points, labels)
    if not any(label == 1 for label in labels):
        return None, None, _display(point_preview), None, "Background point recorded. Add at least one green foreground point."
    flux_inpaint.release_pipeline()
    mask, score, kind = sam_segmenter.segment_reference(
        image,
        points,
        labels,
        backend=backend,
        sam2_path=sam2_path.strip(),
        sam1_path=sam1_path.strip(),
    )
    cutout = sam_segmenter.make_reference_cutout(
        image,
        mask,
        padding_ratio=float(ref_padding),
        feather_radius=float(ref_feather),
    )
    preview = sam_segmenter.segmentation_preview(point_preview, mask)
    fg_count = sum(int(label) == 1 for label in labels)
    bg_count = len(labels) - fg_count
    status = (
        f"✅ {kind.upper()} Subject extracted | score={score:.4f} | "
        f"Foreground points: {fg_count}, background points: {bg_count}. The white-background cutout is the model reference."
    )
    return cutout, mask, _display(preview), _display(cutout), status


def select_reference_point(
    image,
    points,
    labels,
    point_mode,
    backend,
    sam2_path,
    sam1_path,
    ref_padding,
    ref_feather,
    evt: gr.SelectData,
):
    if image is None:
        raise gr.Error("Upload a reference image first.")
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("No click coordinates received. Click the image again.")
    full_point = _display_click_to_full(index, image)
    new_points = [list(item) for item in (points or [])] + [full_point]
    new_labels = [int(item) for item in (labels or [])] + [1 if point_mode == "Foreground (green)" else 0]
    try:
        ready, mask, preview, cutout, status = _predict_reference(
            image,
            new_points,
            new_labels,
            backend,
            sam2_path,
            sam1_path,
            ref_padding,
            ref_feather,
        )
        return new_points, new_labels, ready, mask, preview, cutout, status
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, None, preview, None, f"❌ SAM segmentation failed: {exc}"


def undo_reference_point(
    image,
    points,
    labels,
    backend,
    sam2_path,
    sam1_path,
    ref_padding,
    ref_feather,
):
    if image is None:
        return [], [], None, None, None, None, "Upload a reference image first."
    new_points = [list(item) for item in (points or [])][:-1]
    new_labels = [int(item) for item in (labels or [])][:-1]
    if not new_points:
        return [], [], None, None, _display(image), None, "All points removed. Select the subject again."
    try:
        ready, mask, preview, cutout, status = _predict_reference(
            image,
            new_points,
            new_labels,
            backend,
            sam2_path,
            sam1_path,
            ref_padding,
            ref_feather,
        )
        return new_points, new_labels, ready, mask, preview, cutout, "↶ Last point removed. " + status
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, None, preview, None, f"Add a valid foreground point after undo: {exc}"


def clear_reference_points(image):
    if image is None:
        return [], [], None, None, None, None, "Upload a reference image first."
    return [], [], None, None, _display(image), None, "Points cleared. Add green foreground points again."


def use_whole_reference(image):
    if image is None:
        raise gr.Error("Upload a reference image first.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    mask = Image.new("L", image.size, 255)
    return image, mask, _display(image), _display(image), "✅ SAM skipped. The whole image will be used as the reference."


def reset_target_sam(image):
    if image is None:
        return None, [], [], None, None, None, "Upload a target image to begin."
    image = ImageOps.exif_transpose(image).convert("RGB")
    return (
        image,
        [],
        [],
        None,
        _display(image),
        None,
        "✅ Target loaded. Add green foreground points to select the subject.",
    )


def _predict_target_mask(
    image,
    points,
    labels,
    backend,
    sam2_path,
    sam1_path,
):
    if _generation_active():
        raise RuntimeError("FLUX is generating. Wait for the current task before loading SAM.")
    point_preview = sam_segmenter.draw_points(image, points, labels)
    if not any(label == 1 for label in labels):
        return None, _display(point_preview), None, "Background point recorded. Add at least one green foreground point."
    flux_inpaint.release_pipeline()
    mask, score, kind = sam_segmenter.segment_reference(
        image,
        points,
        labels,
        backend=backend,
        sam2_path=sam2_path.strip(),
        sam1_path=sam1_path.strip(),
    )
    preview = sam_segmenter.segmentation_preview(point_preview, mask)
    fg_count = sum(int(label) == 1 for label in labels)
    bg_count = len(labels) - fg_count
    status = (
        f"✅ Target mask ready | {kind.upper()} score={score:.4f} | "
        f"Foreground points: {fg_count}, background points: {bg_count}. White indicates the editing region."
    )
    return mask, _display(preview), _display(mask, mask=True), status


def select_target_point(
    image,
    points,
    labels,
    point_mode,
    backend,
    sam2_path,
    sam1_path,
    evt: gr.SelectData,
):
    if image is None:
        raise gr.Error("Upload a target image first.")
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("No click coordinates received. Click the image again.")
    full_point = _display_click_to_full(index, image)
    new_points = [list(item) for item in (points or [])] + [full_point]
    new_labels = [int(item) for item in (labels or [])] + [1 if point_mode == "Foreground (green)" else 0]
    try:
        mask, preview, mask_preview, status = _predict_target_mask(
            image, new_points, new_labels, backend, sam2_path, sam1_path
        )
        return new_points, new_labels, mask, preview, mask_preview, status
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, preview, None, f"❌ Target SAM segmentation failed: {exc}"


def undo_target_point(image, points, labels, backend, sam2_path, sam1_path):
    if image is None:
        return [], [], None, None, None, "Upload a target image first."
    new_points = [list(item) for item in (points or [])][:-1]
    new_labels = [int(item) for item in (labels or [])][:-1]
    if not new_points:
        return [], [], None, _display(image), None, "All target points removed."
    try:
        mask, preview, mask_preview, status = _predict_target_mask(
            image, new_points, new_labels, backend, sam2_path, sam1_path
        )
        return new_points, new_labels, mask, preview, mask_preview, "↶ Last point removed. " + status
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, preview, None, f"Add a valid foreground point after undo: {exc}"


def clear_target_points(image):
    if image is None:
        return [], [], None, None, None, "Upload a target image first."
    return [], [], None, _display(image), None, "Target points and mask cleared. Select the target again."


def _completed_job_session(job_id: str) -> dict:
    if not job_id:
        raise gr.Error("Run FLUX generation first.")
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None or job.get("state") != "completed" or job.get("session") is None:
            raise gr.Error("Wait for FLUX generation to finish before refining the subject.")
        return job["session"]


def _predict_generated_subject(
    job_id,
    points,
    labels,
    backend,
    sam2_path,
    sam1_path,
    object_feather,
    object_edge_expand,
    seam_radius,
    seam_strength,
):
    session = _completed_job_session(job_id)
    output = session["output"]
    generated_roi = output["raw_roi"]
    point_preview = sam_segmenter.draw_points(generated_roi, points, labels)
    if not any(int(label) == 1 for label in labels):
        return (
            None,
            _display(point_preview),
            None,
            session["post_final_display"],
            session["download_path"],
            "Background point recorded. Add at least one green foreground point.",
            _metrics_html(output["metrics"]),
        )

    flux_inpaint.release_pipeline()
    started = time.perf_counter()
    mask, score, kind = sam_segmenter.segment_reference(
        generated_roi,
        points,
        labels,
        backend=backend,
        sam2_path=sam2_path.strip(),
        sam1_path=sam1_path.strip(),
    )
    # Use one adjusted binary mask as the authoritative subject ownership.
    # It is returned to the UI and later reused by the consistency pass, so a
    # removed SAM fringe cannot remain protected as raw first-pass FLUX pixels.
    effective_mask = object_segmenter.adjust_object_mask(
        mask, generated_roi.size, int(object_edge_expand)
    )
    result = object_segmenter.composite_object_layers(
        output["source"],
        generated_roi,
        effective_mask,
        # The original edit Mask only guides FLUX. Once the user has selected
        # the generated subject with SAM, paste that complete SAM result back;
        # do not clip wings, brims or other details at the old rectangular edge.
        Image.new("L", output["source"].size, 255),
        output["roi_box"],
        object_feather=float(object_feather),
        object_edge_expand=0,
        seam_gaussian_radius=float(seam_radius),
        seam_gaussian_strength=float(seam_strength),
        shadow_enabled=False,
    )
    segmentation_seconds = time.perf_counter() - started
    session["post_final"] = result.final
    session["generated_object_mask"] = effective_mask
    session["generated_object_alpha"] = result.object_alpha
    session["post_final_revision"] = int(session.get("post_final_revision", 0)) + 1
    _materialize_session_artifacts(session)

    metrics = dict(output["metrics"])
    metrics["segmentation_enabled"] = True
    metrics["segmentation_seconds"] = segmentation_seconds
    output["metrics"].update(metrics)
    fg_count = sum(int(label) == 1 for label in labels)
    bg_count = len(labels) - fg_count
    status = (
        f"✅ Generated subject segmented and composited with {kind.upper()} | score={score:.4f} | "
        f"Foreground points: {fg_count}, background points: {bg_count} | "
        f"Feathering={float(object_feather):.1f}px, mask adjustment={int(object_edge_expand):+d}px."
        "The refined binary subject has been composited. The removed edge ring is included in background harmonization."
    )
    _update_job(
        job_id,
        post_final=session["post_final_display"],
        final_file=session["download_path"],
        metrics_html=_metrics_html(metrics),
    )
    preview = sam_segmenter.segmentation_preview(point_preview, effective_mask)
    return (
        effective_mask,
        _display(preview),
        _display(mask, mask=True),
        session["post_final_display"],
        session["download_path"],
        status,
        _metrics_html(metrics),
    )


def select_generated_subject_point(
    job_id,
    points,
    labels,
    point_mode,
    backend,
    sam2_path,
    sam1_path,
    object_feather,
    object_edge_expand,
    seam_radius,
    seam_strength,
    evt: gr.SelectData,
):
    session = _completed_job_session(job_id)
    image = session["output"]["raw_roi"]
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("No click coordinates received. Click the generated image again.")
    full_point = _display_click_to_full(index, image)
    new_points = [list(item) for item in (points or [])] + [full_point]
    new_labels = [int(item) for item in (labels or [])] + [
        1 if point_mode == "Foreground (green)" else 0
    ]
    try:
        mask, preview, mask_preview, final, download, status, metrics = _predict_generated_subject(
            job_id,
            new_points,
            new_labels,
            backend,
            sam2_path,
            sam1_path,
            object_feather,
            object_edge_expand,
            seam_radius,
            seam_strength,
        )
        return new_points, new_labels, mask, preview, mask_preview, final, download, status, metrics
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, preview, None, gr.update(), gr.update(), f"❌ Generated subject segmentation failed: {exc}", gr.update()


def undo_generated_subject_point(
    job_id,
    points,
    labels,
    backend,
    sam2_path,
    sam1_path,
    object_feather,
    object_edge_expand,
    seam_radius,
    seam_strength,
):
    session = _completed_job_session(job_id)
    image = session["output"]["raw_roi"]
    new_points = [list(item) for item in (points or [])][:-1]
    new_labels = [int(item) for item in (labels or [])][:-1]
    if not new_points:
        return clear_generated_subject_points(job_id)
    mask, preview, mask_preview, final, download, status, metrics = _predict_generated_subject(
        job_id,
        new_points,
        new_labels,
        backend,
        sam2_path,
        sam1_path,
        object_feather,
        object_edge_expand,
        seam_radius,
        seam_strength,
    )
    return new_points, new_labels, mask, preview, mask_preview, final, download, "↶ Last point removed. " + status, metrics


def clear_generated_subject_points(job_id):
    session = _completed_job_session(job_id)
    output = session["output"]
    session["post_final"] = output["final"]
    session.pop("generated_object_mask", None)
    session.pop("generated_object_alpha", None)
    session["post_final_revision"] = int(session.get("post_final_revision", 0)) + 1
    _materialize_session_artifacts(session)
    metrics = dict(output["metrics"])
    metrics["segmentation_enabled"] = False
    metrics["segmentation_seconds"] = 0.0
    output["metrics"].update(metrics)
    _update_job(
        job_id,
        post_final=session["post_final_display"],
        final_file=session["download_path"],
        metrics_html=_metrics_html(metrics),
    )
    return (
        [],
        [],
        None,
        _display(output["raw_roi"]),
        None,
        session["post_final_display"],
        session["download_path"],
        "Generated subject points and mask cleared. Initial FLUX composite restored.",
        _metrics_html(metrics),
    )


def _predict_original_subject(job_id, points, labels, backend, sam2_path, sam1_path):
    session = _completed_job_session(job_id)
    image = session["output"]["source_roi"]
    point_preview = sam_segmenter.draw_points(image, points, labels)
    if not any(int(label) == 1 for label in labels):
        return None, _display(point_preview), None, "Background point recorded. Add at least one green foreground point."
    flux_inpaint.release_pipeline()
    mask, score, kind = sam_segmenter.segment_reference(
        image,
        points,
        labels,
        backend=backend,
        sam2_path=sam2_path.strip(),
        sam1_path=sam1_path.strip(),
    )
    session["original_object_mask"] = mask
    preview = sam_segmenter.segmentation_preview(point_preview, mask)
    fg_count = sum(int(label) == 1 for label in labels)
    bg_count = len(labels) - fg_count
    status = (
        f"✅ Original subject segmented with {kind.upper()} | score={score:.4f} | "
        f"Foreground points: {fg_count}, background points: {bg_count}."
    )
    return mask, _display(preview), _display(mask, mask=True), status


def select_original_subject_point(
    job_id, points, labels, point_mode, backend, sam2_path, sam1_path, evt: gr.SelectData
):
    session = _completed_job_session(job_id)
    image = session["output"]["source_roi"]
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("No click coordinates received. Click the original target region again.")
    new_points = [list(item) for item in (points or [])] + [_display_click_to_full(index, image)]
    new_labels = [int(item) for item in (labels or [])] + [
        1 if point_mode == "Foreground (green)" else 0
    ]
    try:
        mask, preview, mask_preview, status = _predict_original_subject(
            job_id, new_points, new_labels, backend, sam2_path, sam1_path
        )
        return new_points, new_labels, mask, preview, mask_preview, status
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, preview, None, f"❌ Original subject segmentation failed: {exc}"


def undo_original_subject_point(job_id, points, labels, backend, sam2_path, sam1_path):
    new_points = [list(item) for item in (points or [])][:-1]
    new_labels = [int(item) for item in (labels or [])][:-1]
    if not new_points:
        return clear_original_subject_points(job_id)
    mask, preview, mask_preview, status = _predict_original_subject(
        job_id, new_points, new_labels, backend, sam2_path, sam1_path
    )
    return new_points, new_labels, mask, preview, mask_preview, "↶ Last point removed. " + status


def clear_original_subject_points(job_id):
    session = _completed_job_session(job_id)
    session.pop("original_object_mask", None)
    return (
        [], [], None, _display(session["output"]["source_roi"]), None,
        "Original subject points and mask cleared.",
    )


def run_background_consistency(
    job_id,
    generated_subject_mask,
    original_subject_mask,
    original_cleanup_expand,
    generated_fill_expand,
    fill_radius,
    color_lock_strength,
    color_lock_radius,
    color_lock_max_delta,
    background_edge_feather,
    object_feather,
    object_edge_expand,
    seam_radius,
    seam_strength,
):
    if generated_subject_mask is None:
        raise gr.Error("Select the generated subject with SAM first.")
    if original_subject_mask is None:
        raise gr.Error("Select the original subject with SAM first.")
    session = _completed_job_session(job_id)
    output = session["output"]
    sam_segmenter.release_segmenter()
    flux_inpaint.release_pipeline()
    try:
        with _GENERATION_LOCK:
            consistency = flux_inpaint.edit_background_consistency(
                output["source_roi"],
                output["raw_roi"],
                original_subject_mask,
                generated_subject_mask,
                output["core_roi"],
                original_cleanup_expand=int(original_cleanup_expand),
                generated_fill_expand=int(generated_fill_expand),
                fill_radius=float(fill_radius),
                color_lock_strength=float(color_lock_strength),
                color_lock_radius=float(color_lock_radius),
                color_lock_max_delta=float(color_lock_max_delta),
                background_transition_radius=float(background_edge_feather),
            )
    except Exception as exc:
        raise gr.Error(f"Background harmonization failed: {exc}") from exc
    finally:
        flux_inpaint.release_pipeline()

    # Reuse the established outer compositor, not the uncorrected direct_final product.
    # First color-correct the single FLUX background candidate, then perform
    # exactly one source/ROI merge with the original 48/32/8 ownership rules.
    # Keeping direct_final itself would preserve its visible rectangular color
    # block; pasting a second local repair on top would create another contour.
    background_full = composite_roi_with_outward_feather(
        output["source"],
        consistency["consistent_background_roi"],
        output["core_mask"],
        output["roi_box"],
        expand_pixels=int(output["blend_expand"]),
        hard_protect_pixels=min(8, max(0, int(output["blend_expand"]))),
        feather_pixels=float(output["direct_edge_feather"]),
        seam_color_match=bool(output["direct_seam_color_match"]),
        seam_color_strength=float(output["direct_seam_color_strength"]),
        green_cleanup=bool(output.get("green_despill_enabled", True)),
        green_cleanup_width=int(output.get("green_despill_width", 24)),
    )
    boundary_info = {
        "applied": False,
        "sample_strategy": "single_corrected_background_outer_compositor",
    }
    result = object_segmenter.composite_object_layers(
        background_full,
        output["raw_roi"],
        generated_subject_mask,
        Image.new("L", output["source"].size, 255),
        output["roi_box"],
        object_feather=float(object_feather),
        # generated_subject_mask is already the adjusted authoritative mask
        # returned by _predict_generated_subject. Applying the UI adjustment a
        # second time would create another ownership mismatch.
        object_edge_expand=0,
        seam_gaussian_radius=float(seam_radius),
        seam_gaussian_strength=float(seam_strength),
        shadow_enabled=False,
    )
    session["post_final"] = result.final
    session["generated_object_mask"] = generated_subject_mask
    session["original_object_mask"] = original_subject_mask
    session["consistency"] = consistency
    session["consistency"]["outer_boundary_info"] = boundary_info
    session["post_final_revision"] = int(session.get("post_final_revision", 0)) + 1
    _materialize_session_artifacts(session)
    metrics = dict(output["metrics"])
    metrics["consistency_enabled"] = False
    metrics["second_flux_enabled"] = False
    metrics["consistency_lora_enabled"] = False
    metrics["first_pass_background_color_lock_enabled"] = True
    metrics["consistency_seconds"] = consistency["seconds"]
    output["metrics"].update(metrics)
    status = (
        f"✅ Background harmonized | Single background composite | No second FLUX pass | "
        f"First-pass background corrected | ROI={consistency['model_size'][0]}×{consistency['model_size'][1]} | "
        f"Elapsed={consistency['seconds']:.2f}s | Low-frequency background error: "
        f"{consistency['color_lock_info']['error_before']:.2f}→"
        f"{consistency['color_lock_info']['error_after']:.2f} | Background transition="
        f"{float(background_edge_feather):.1f}px | Subject edge adjustment="
        f"{int(object_edge_expand):+d}px. Background uses the "
        f"harmonized first-pass FLUX output | Original cleanup expansion={int(original_cleanup_expand)}px | "
        f"Generated subject exclusion margin={int(generated_fill_expand)}px. The full generated subject is pasted last; estimation images are not composited."
    )
    _update_job(
        job_id,
        post_final=session["post_final_display"],
        final_file=session["download_path"],
        metrics_html=_metrics_html(metrics),
    )
    return (
        _display(consistency["background_layer_roi"]),
        session["post_final_display"],
        session["download_path"],
        status,
        _metrics_html(metrics),
    )


def _build_v4_gallery(output):
    expansion = output.get("bbox_expansion", {})
    mask_box_preview = draw_user_mask_bbox(
        output["object_roi"],
        output["core_roi"],
        expand_left=expansion.get("left", 0),
        expand_right=expansion.get("right", 0),
        expand_top=expansion.get("top", 0),
        expand_bottom=expansion.get("bottom", 0),
    )
    return [
        (mask_box_preview, "① Generated image and mask bounding box"),
        (output["core_roi"], "② Compositing mask (white = generated region)"),
        (output["full_background"], "③ Original target"),
        (output["model_input"], "④ Model input (green = LoRA region)"),
    ]


def _get_current_flux_editor():
    """Repair a stale imported module after an app.py-only hot reload."""
    required = {
        "bbox_expand_left",
        "bbox_expand_right",
        "bbox_expand_top",
        "bbox_expand_bottom",
        "background_similarity_threshold",
        "background_restore_strength",
        "direct_overlay_enabled",
        "direct_edge_feather",
        "direct_seam_color_match",
        "direct_seam_color_strength",
    }
    editor = flux_inpaint.edit_with_reference
    if required.issubset(inspect.signature(editor).parameters):
        return editor

    # Some launchers reload app.py but keep imported local modules cached.
    # Release old CUDA objects before refreshing flux_inpaint.py.
    flux_inpaint.release_pipeline()
    importlib.invalidate_caches()
    importlib.reload(flux_inpaint)
    editor = flux_inpaint.edit_with_reference
    missing = required.difference(inspect.signature(editor).parameters)
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(
            f"flux_inpaint.py and app.py are out of sync. Missing parameters: {names}. "
            "Use matching files from the same project and restart Gradio."
        )
    return editor


def _artifact_dir(job_id: str) -> Path:
    if not job_id or not job_id.isalnum():
        raise ValueError("Invalid job id for output artifacts.")
    directory = _OUTPUT_ROOT / job_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _materialize_session_artifacts(
    session: dict,
    *,
    job_id: str | None = None,
    refresh_static: bool = False,
) -> None:
    """Encode compact previews once instead of Base64-encoding full PIL images."""
    if job_id is not None:
        directory = _artifact_dir(job_id)
        session["artifact_dir"] = str(directory)
    else:
        directory = Path(session["artifact_dir"])

    if refresh_static or not session.get("gallery_display"):
        gallery_display = []
        for index, (image, caption) in enumerate(session["gallery"], start=1):
            path = ui_preview.save_preview(
                image,
                directory / f"diagnostic_{index:02d}",
                max_side=1100,
                mask=image.mode in {"1", "L"},
            )
            gallery_display.append((path, caption))
        session["gallery_display"] = gallery_display

    # The user-facing final composite must keep the Target's exact pixel size.
    # Save it as a same-resolution JPEG for a smaller Gradio 3 Base64 payload;
    # the separate download remains a lossless full-resolution PNG.
    revision = max(0, int(session.get("post_final_revision", 0)))
    preview_stem = "final_preview" if revision == 0 else f"final_preview_sam_{revision:03d}"
    download_stem = (
        "final_full_resolution"
        if revision == 0
        else f"final_full_resolution_sam_{revision:03d}"
    )
    session["post_final_display"] = ui_preview.save_preview(
        session["post_final"],
        directory / preview_stem,
        max_side=max(session["post_final"].size),
    )
    session["download_path"] = ui_preview.save_full_png(
        session["post_final"], directory / download_stem
    )


def run_edit(
    source_data,
    mask_data,
    target_mask_mode,
    target_sam_source,
    target_sam_mask,
    reference_ready,
    model_path,
    prompt,
    steps,
    guidance,
    strength,
    seed,
    max_roi_side,
    roi_context,
    bbox_expand_left,
    bbox_expand_right,
    bbox_expand_top,
    bbox_expand_bottom,
    use_roi,
    model_mask_expand,
    model_mask_blur,
    blend_expand,
    direct_edge_feather,
    direct_seam_color_match,
    direct_seam_color_strength,
    cpu_offload,
    local_files_only,
    lora_enabled,
    lora_path,
    lora_weight_name,
    lora_scale,
    green_screen_input,
    direct_overlay_enabled,
    halo_fix_enabled,
    halo_luminance_only,
    halo_ring_width,
    halo_strength,
    halo_fade_radius,
    halo_interior_strength,
    green_despill_enabled,
    green_despill_width,
    texture_blend_enabled,
    texture_blend_levels,
    background_similarity_threshold,
    background_restore_strength,
    progress_callback=None,
):
    request_started = time.perf_counter()

    def report(message: str, value: float) -> None:
        if progress_callback is not None:
            progress_callback(message, value)

    if reference_ready is None:
        raise gr.Error("Reference is not ready. Select a subject with SAM or use the whole image as reference.")
    if target_mask_mode == TARGET_SAM_MODE:
        if target_sam_source is None:
            raise gr.Error("Upload the target in the Segment Target with SAM tab.")
        if target_sam_mask is None:
            raise gr.Error("The target mask is not ready. Add green foreground points to select the subject.")
        source = ImageOps.exif_transpose(target_sam_source).convert("RGB")
        mask = ImageOps.exif_transpose(target_sam_mask).convert("L")
        print("[Task] Target mask source: SAM", flush=True)
    else:
        source = decode_data_url(source_data, "Target image").convert("RGB")
        mask = decode_data_url(mask_data, "Target mask").convert("L")
        print("[Task] Target mask source: drawing editor", flush=True)
    if mask.resize(source.size, Image.Resampling.NEAREST).getbbox() is None:
        raise gr.Error("The editing region is empty. Draw a rectangle or use the brush on the target image.")
    report("Inputs validated", 0.02)

    try:
        sam_segmenter.release_segmenter()
        output = _get_current_flux_editor()(
            source,
            mask,
            reference_ready,
            model_path=model_path.strip(),
            prompt=prompt,
            seed=int(seed),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            max_roi_side=int(max_roi_side),
            roi_context_ratio=float(roi_context),
            bbox_expand_left=int(bbox_expand_left),
            bbox_expand_right=int(bbox_expand_right),
            bbox_expand_top=int(bbox_expand_top),
            bbox_expand_bottom=int(bbox_expand_bottom),
            use_roi=bool(use_roi),
            model_mask_expand=int(model_mask_expand),
            model_mask_blur=float(model_mask_blur),
            blend_expand=int(blend_expand),
            direct_edge_feather=float(direct_edge_feather),
            direct_seam_color_match=bool(direct_seam_color_match),
            direct_seam_color_strength=float(direct_seam_color_strength),
            cpu_offload=bool(cpu_offload),
            local_files_only=bool(local_files_only),
            lora_enabled=bool(lora_enabled),
            lora_path=lora_path.strip(),
            lora_weight_name=lora_weight_name.strip(),
            lora_scale=float(lora_scale),
            green_screen_input=bool(green_screen_input),
            direct_overlay_enabled=bool(direct_overlay_enabled),
            halo_fix_enabled=bool(halo_fix_enabled),
            halo_luminance_only=bool(halo_luminance_only),
            halo_ring_width=int(halo_ring_width),
            halo_strength=float(halo_strength),
            halo_fade_radius=float(halo_fade_radius),
            halo_interior_strength=float(halo_interior_strength),
            green_despill_enabled=bool(green_despill_enabled),
            green_despill_width=int(green_despill_width),
            texture_blend_enabled=bool(texture_blend_enabled),
            texture_blend_levels=int(texture_blend_levels),
            background_similarity_threshold=float(background_similarity_threshold),
            background_restore_strength=float(background_restore_strength),
            # Keep a small tail for model release, direct Mask compositing and
            # preparation of the Gradio display files.
            progress_callback=lambda message, value: report(
                message, min(0.94, max(0.0, float(value)) * 0.94)
            ),
        )
    except Exception as exc:
        raise gr.Error(f"FLUX generation failed: {exc}") from exc

    report("FLUX generation finished. Compositing the masked region...", 0.96)
    flux_inpaint.release_pipeline()
    final = output["final"]
    gallery = _build_v4_gallery(output)
    width, height = output["model_size"]
    metrics = dict(output["metrics"])
    metrics["end_to_end_seconds"] = time.perf_counter() - request_started
    metrics["segmentation_seconds"] = 0.0
    metrics["segmentation_enabled"] = False
    output["metrics"].update(metrics)
    lora_status = (
        f"ON · scale={output['lora_scale']:.2f} · green={'ON' if output['green_screen_input'] else 'OFF'}"
        if output["lora_enabled"]
        else "OFF"
    )
    halo_info = output["halo_info"]
    halo_status = (
        f"ON · Boundary error: {halo_info['edge_error_before']:.1f}→{halo_info['edge_error_after']:.1f} · "
        f"Texture levels: {halo_info['texture_blend_levels']} · "
        f"Despill={'ON' if halo_info['despill']['applied'] else 'No spill / not applied'}"
        if halo_info["applied"]
        else "OFF/Not applied"
    )
    status = (
        f"✅ Reference-guided editing complete | FLUX {output['flux_pass_count']} pass(es) | "
        f"seed={output['seed']} | ROI={output['roi_box_text']} | "
        f"Bounding box={output['location_box_text']} | "
        f"Composite expansion={output['blend_expand']}px | "
        f"Outer feathering={output['direct_edge_feather']:.0f}px | "
        f"Seam color matching={'ON' if output['direct_seam_color_match'] else 'OFF'}"
        f"({output['direct_seam_color_strength']:.2f}) | "
        f"Inference size={width}×{height} | Total elapsed time={metrics['end_to_end_seconds']:.2f}s | "
        f"Peak GPU memory={metrics['peak_reserved_gib']:.2f}GiB | LoRA={lora_status} | Halo={halo_status}"
        f"<br>Output={'Original generated subject with outer background blending' if output['direct_overlay_enabled'] else 'Boundary color correction and background restoration'}"
        f" | Subject SAM: waiting for selection | Shadow compositing: off"
        f"<br>Prompt: {output['prompt']}"
    )
    print(
        f"[Metrics] Gradio total elapsed time={metrics['end_to_end_seconds']:.2f}s",
        flush=True,
    )
    session = {
        "output": output,
        "gallery": gallery,
        "post_final": final,
    }
    return gallery, status, _metrics_html(metrics), session


def _run_background_edit(job_id: str, args: tuple) -> None:
    _update_job(job_id, state="running", message="Task started. Validating inputs...", progress=2)
    print(f"[Task {job_id[:8]}] Background generation started", flush=True)
    try:
        with _GENERATION_LOCK:
            gallery, status, metrics_html, session = run_edit(
                *args,
                progress_callback=lambda message, value: _update_job(
                    job_id, message=message, progress=float(value) * 100
                ),
            )
        _materialize_session_artifacts(
            session,
            job_id=job_id,
            refresh_static=True,
        )
        gallery = session["gallery_display"]
        _update_job(
            job_id,
            state="completed",
            message="Generation complete.",
            progress=100,
            gallery=gallery,
            status=status,
            metrics_html=metrics_html,
            session=session,
            post_final=session["post_final_display"],
            final_file=session["download_path"],
            artifact_dir=session["artifact_dir"],
        )
        _prune_finished_jobs(10)
        print(f"[Task {job_id[:8]}] Complete", flush=True)
    except Exception as exc:
        traceback.print_exc()
        message = f"Generation failed: {exc}"
        _update_job(job_id, state="failed", message=message, status=message)
        _prune_finished_jobs(10)
        print(f"[Task {job_id[:8]}] {message}", flush=True)


def start_background_edit(*args):
    if _generation_active():
        raise gr.Error("A FLUX task is already running. Wait for it to finish before submitting again.")
    job_id = uuid.uuid4().hex
    message = "Task submitted. Starting generation..."
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "state": "queued",
            "message": message,
            "progress": 1.0,
            "gallery": None,
            "status": message,
            "metrics_html": _metrics_html(),
            "session": None,
        }
    _JOB_EXECUTOR.submit(_run_background_edit, job_id, tuple(args))
    print(f"[Task {job_id[:8]}] Submitted", flush=True)
    return (
        job_id,
        message,
        _progress_html(1, message),
        _metrics_html(),
        None,
        None,
        gr.update(interactive=True),
        gr.update(interactive=False),
        None,
        [],
        [],
        None,
        None,
        "After generation, click the subject you want to keep.",
        None,
        [],
        [],
        None,
        None,
        "After generation, select the original subject in the target region.",
        None,
        "Select both the generated and original subject masks first.",
    )


def poll_background_edit(job_id: str):
    if not job_id:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(interactive=False), gr.update(interactive=True),
            gr.update(), gr.update(), gr.update(), gr.update(),
        )
    with _JOBS_LOCK:
        stored = _JOBS.get(job_id)
        job = dict(stored) if stored is not None else None
    if job is None:
        message = "Task state was lost. Submit the task again."
        return (
            gr.update(),
            message,
            _progress_html(0, message, "failed"),
            _metrics_html(),
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(),
            message,
            gr.update(),
            message,
        )
    state = job.get("state", "running")
    progress = float(job.get("progress", 0))
    message = job.get("message", "Generation is running...")
    if state == "completed":
        session = job.get("session") or {}
        output = session.get("output") or {}
        return (
            job.get("gallery"),
            job.get("status", "✅ Generation complete."),
            _progress_html(100, "Generation complete.", "completed"),
            job.get("metrics_html", _metrics_html()),
            job.get("post_final"),
            job.get("final_file"),
            gr.update(interactive=False),
            gr.update(interactive=True),
            _display(output.get("raw_roi")),
            "✅ FLUX finished. Add green points on the generated image to select the subject to keep.",
            _display(output.get("source_roi")),
            "✅ Add green points in the original target region to select the original subject.",
        )
    if state == "failed":
        return (
            gr.update(),
            job.get("status", message),
            _progress_html(progress, message, "failed"),
            job.get("metrics_html", _metrics_html()),
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(),
            job.get("status", message),
            gr.update(),
            job.get("status", message),
        )
    return (
        gr.update(),
        message,
        _progress_html(progress, message),
        job.get("metrics_html", _metrics_html()),
        gr.update(),
        gr.update(),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(),
        "FLUX is generating. Subject refinement will be available when it finishes.",
        gr.update(),
        "FLUX is generating. Original subject selection will be available when it finishes.",
    )


def step_title(number: int, title: str, note: str) -> str:
    return (
        '<div class="step-title">'
        f'<span class="step-badge">{number}</span>'
        f'<div><strong>{title}</strong><br><small>{note}</small></div></div>'
    )


def build_demo():
    with gr.Blocks(title="FLUX Reference Compositor", css=EDITOR_CSS) as demo:
        source_data = gr.Textbox(elem_id="green-source-data")
        mask_data = gr.Textbox(elem_id="green-mask-data")
        reference_base = gr.State(None)
        reference_points = gr.State([])
        reference_labels = gr.State([])
        reference_ready = gr.State(None)
        reference_mask = gr.State(None)
        target_mask_mode = gr.State(TARGET_DRAW_MODE)
        target_sam_source = gr.State(None)
        target_sam_points = gr.State([])
        target_sam_labels = gr.State([])
        target_sam_mask = gr.State(None)
        active_job_id = gr.State("")
        generated_sam_points = gr.State([])
        generated_sam_labels = gr.State([])
        generated_sam_mask = gr.State(None)
        original_sam_points = gr.State([])
        original_sam_labels = gr.State([])
        original_sam_mask = gr.State(None)

        gr.HTML(
            '<div class="app-hero"><h1>FLUX Reference Compositor · Reference-Guided Editing</h1>'
            '<p>Generate once with Outpaint LoRA, then harmonize the background and composite the subject without another diffusion pass.</p>'
            '<div class="input-map"><span>Reference: what to use</span><b>＋</b>'
            '<span>Mask: where to edit</span><b>＋</b><span>Prompt: optional guidance</span>'
            '<b>→</b><span>Single FLUX pass</span><b>→</b><span>Mask compositing</span></div></div>'
        )

        gr.HTML(step_title(1, "Prepare the Reference", "Use green points to select the subject and red points to exclude the background. The white-background preview is sent to the model."))
        with gr.Row(equal_height=True, elem_classes="reference-row"):
            reference_upload = gr.Image(
                label="A. Upload reference image",
                source="upload",
                type="pil",
                height=315,
                scale=2,
            )
            with gr.Column(scale=2, min_width=320, elem_classes="reference-panel"):
                reference_preview = gr.Image(
                    label="B. Click to add prompt points",
                    type="pil",
                    interactive=True,
                    height=255,
                )
                point_mode = gr.Radio(
                    ["Foreground (green)", "Background (red)"],
                    value="Foreground (green)",
                    label="Point type",
                    elem_classes="sam-mode",
                )
                with gr.Row():
                    undo_point = gr.Button("Undo point")
                    clear_points = gr.Button("Clear points")
            reference_cutout = gr.Image(
                label="C. Reference cutout",
                type="pil",
                interactive=False,
                height=315,
                scale=2,
                elem_classes="reference-ready",
            )
        with gr.Row():
            whole_reference = gr.Button("Use the whole image as reference")
            sam_status = gr.Markdown("Upload a reference image to begin.", elem_classes="status-panel")

        gr.HTML('<div class="workflow-section">' + step_title(2, "Select the Target Region", "Draw an editing region or use SAM points to select the target subject.") + '</div>')
        with gr.Tabs(selected="target-draw", elem_classes="target-mode-tabs"):
            with gr.TabItem("Draw Mask", id="target-draw") as target_draw_tab:
                gr.HTML(EDITOR_HTML)
            with gr.TabItem("Segment Target with SAM", id="target-sam") as target_sam_tab:
                gr.Markdown(
                    "Upload the target and add green points inside the subject. Add red background points to refine the selection.",
                    elem_classes="hint-panel",
                )
                with gr.Row(equal_height=True, elem_classes="reference-row"):
                    target_sam_upload = gr.Image(
                        label="A. Upload target image",
                        source="upload",
                        type="pil",
                        height=330,
                        scale=2,
                    )
                    with gr.Column(scale=2, min_width=320, elem_classes="reference-panel"):
                        target_sam_preview = gr.Image(
                            label="B. Click to select the target",
                            type="pil",
                            interactive=True,
                            height=265,
                        )
                        target_point_mode = gr.Radio(
                            ["Foreground (green)", "Background (red)"],
                            value="Foreground (green)",
                            label="Point type",
                            elem_classes="sam-mode",
                        )
                        with gr.Row():
                            undo_target = gr.Button("Undo point")
                            clear_target = gr.Button("Clear target points")
                    target_mask_preview = gr.Image(
                        label="C. Target mask (white = edit)",
                        type="pil",
                        interactive=False,
                        height=330,
                        scale=2,
                    )
                target_sam_status = gr.Markdown(
                    "Upload a target image to begin.",
                    elem_classes="status-panel",
                )

        gr.HTML('<div class="workflow-section">' + step_title(3, "Generate", "Optionally describe the desired edit, then generate with the reference image.") + '</div>')
        with gr.Row(elem_classes="generate-row"):
            prompt = gr.Textbox(
                label="Optional prompt",
                placeholder="For example: face left, place on the table, preserve the product label. You can leave this blank.",
                lines=3,
                scale=5,
            )
            with gr.Column(scale=2, min_width=290, elem_classes="generate-panel"):
                gr.Markdown("Check the target mask and reference cutout, then generate.", elem_classes="hint-panel")
                generate = gr.Button("Generate", variant="primary", elem_classes="generate-action")

        status = gr.Markdown("Prepare the reference and target first.", elem_classes="status-panel")
        generation_progress = gr.HTML(_progress_html(0, "Ready to generate."))
        generation_metrics = gr.HTML(_metrics_html())
        generation_poll = gr.Button(
            "Refresh generation status",
            elem_id="green-generation-poll",
            elem_classes="poll-trigger",
            interactive=False,
        )
        output_gallery = gr.Gallery(
            label="Diagnostic previews (download the full-resolution result below)",
            columns=3,
            height=440,
            elem_classes="result-gallery",
        )
        gr.HTML(
            '<div class="workflow-section">' + step_title(
                4,
                "Refine the Generated Subject",
                "Add green points to select the generated subject and red points to exclude the background.",
            ) + '</div>'
        )
        with gr.Row(equal_height=True, elem_classes="reference-row"):
            with gr.Column(scale=2):
                generated_sam_preview = gr.Image(
                    label="A. Select the subject in the generated image",
                    type="pil",
                    interactive=True,
                    height=360,
                )
                generated_sam_point_mode = gr.Radio(
                    ["Foreground (green)", "Background (red)"],
                    value="Foreground (green)",
                    label="Point type",
                    elem_classes="sam-mode",
                )
                with gr.Row():
                    undo_generated_sam = gr.Button("Undo point")
                    clear_generated_sam = gr.Button("Clear points and restore initial result")
            generated_sam_mask_preview = gr.Image(
                label="B. Generated subject mask (white = keep)",
                type="pil",
                interactive=False,
                height=360,
                scale=2,
            )
        generated_object_feather = gr.State(0.0)
        generated_seam_radius = gr.State(0.0)
        generated_seam_strength = gr.State(0.0)
        with gr.Row():
            generated_object_expand = gr.Slider(
                -8, 8, value=0, step=1,
                label="Subject mask adjustment (px, default: 0)"
            )
            gr.Markdown("The subject uses a binary mask without feathering or Gaussian seam blending.")
        generated_sam_status = gr.Markdown(
            "After generation, click the subject you want to keep.",
            elem_classes="status-panel",
        )
        with gr.Row(equal_height=True, elem_classes="reference-row"):
            with gr.Column(scale=2):
                original_sam_preview = gr.Image(
                    label="C. Select the original subject to remove",
                    type="pil",
                    interactive=True,
                    height=360,
                )
                original_sam_point_mode = gr.Radio(
                    ["Foreground (green)", "Background (red)"],
                    value="Foreground (green)",
                    label="Point type",
                    elem_classes="sam-mode",
                )
                with gr.Row():
                    undo_original_sam = gr.Button("Undo point")
                    clear_original_sam = gr.Button("Clear original subject points")
            original_sam_mask_preview = gr.Image(
                label="D. Original subject mask (white = exclude from color estimation)",
                type="pil",
                interactive=False,
                height=360,
                scale=2,
            )
        original_sam_status = gr.Markdown(
            "After generation, select the original subject in the target region.",
            elem_classes="status-panel",
        )

        gr.HTML(
            '<div class="workflow-section">' + step_title(
                5,
                "Harmonize & Composite",
                "Use both subject masks to harmonize the background and composite the final subject.",
            ) + '</div>'
        )
        with gr.Accordion("Background Harmonization", open=True):
            with gr.Row():
                consistency_original_cleanup_expand = gr.Slider(
                    0, 48, value=12, step=1,
                    label="Original subject mask expansion (px)",
                )
                consistency_generated_fill_expand = gr.Slider(
                    0, 48, value=16, step=1,
                    label="Generated subject exclusion margin (px)",
                )
                consistency_fill_radius = gr.Slider(8, 160, value=48, step=4, label="Background estimation radius (px)")
            with gr.Row():
                consistency_color_lock = gr.Slider(
                    0.0, 1.0, value=0.9, step=0.05, label="Target color matching strength"
                )
                consistency_color_radius = gr.Slider(
                    8, 256, value=64, step=8, label="Color field smoothing radius (px)"
                )
                consistency_color_max_delta = gr.Slider(
                    8, 128, value=72, step=4, label="Maximum RGB correction"
                )
                consistency_background_feather = gr.Slider(
                    0.0, 32.0, value=12.0, step=0.5,
                    label="Original subject repair transition (px)"
                )
            gr.Markdown(
                "Background harmonization runs on the CPU without another FLUX pass. "
                "Subject masks exclude subject colors from background estimation. The corrected background is blended once, "
                "then the complete generated subject is composited on top. "
                "Temporary estimation images are not used as final pixels."
            )
        run_consistency = gr.Button("Harmonize & Composite", variant="primary")
        consistency_status = gr.Markdown(
            "Select both the generated and original subject masks first.",
            elem_classes="status-panel",
        )
        consistency_background_preview = gr.Image(
            label="Harmonized background preview",
            type="pil",
            interactive=False,
            height=360,
        )
        post_final_preview = gr.Image(
            label="Final composite",
            type="pil",
            interactive=False,
            height=560,
        )
        final_download = gr.File(
            label="Download full-resolution PNG",
            interactive=False,
        )

        with gr.Accordion("FLUX Settings", open=False):
            gr.Markdown("Defaults are for the distilled 4B model. For Base-4B, try 28–50 steps and guidance 4–8.")
            model_path = gr.Textbox(label="Model path or Hugging Face ID", value=flux_inpaint.DEFAULT_MODEL_PATH)
            with gr.Row():
                steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                guidance = gr.Slider(0, 10, value=1.0, step=0.1, label="Guidance")
                strength = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Strength")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            with gr.Row():
                max_roi_side = gr.Slider(512, 2048, value=1024, step=16, label="Maximum ROI side")
                roi_context = gr.Slider(0.05, 1.5, value=0.45, step=0.05, label="ROI context ratio")
                use_roi = gr.Checkbox(value=True, label="Process only the region around the mask")
            gr.Markdown(
                "The yellow bounding box follows the mask. Expand each side below in original-image pixels. "
                "The context ratio adds scene context beyond this box."
            )
            with gr.Row():
                bbox_expand_left = gr.Slider(0, 1024, value=0, step=4, label="Left bounding-box expansion (px)")
                bbox_expand_right = gr.Slider(0, 1024, value=0, step=4, label="Right bounding-box expansion (px)")
            with gr.Row():
                bbox_expand_top = gr.Slider(0, 1024, value=0, step=4, label="Top bounding-box expansion (px)")
                bbox_expand_bottom = gr.Slider(0, 1024, value=0, step=4, label="Bottom bounding-box expansion (px)")
            with gr.Row():
                model_mask_expand = gr.Slider(0, 64, value=8, step=1, label="Model mask expansion")
                model_mask_blur = gr.Slider(0, 32, value=8, step=1, label="Model mask blur")
                blend_expand = gr.Slider(
                    0,
                    128,
                    value=48,
                    step=1,
                    label="Composite expansion (px)",
                )
                direct_edge_feather = gr.Slider(
                    0,
                    64,
                    value=32,
                    step=1,
                    label="Outer blend feathering (px)",
                )
            with gr.Row():
                direct_seam_color_match = gr.Checkbox(
                    value=True,
                    label="Match low-frequency seam colors",
                )
                direct_seam_color_strength = gr.Slider(
                    0.0,
                    1.0,
                    value=0.9,
                    step=0.05,
                    label="Seam color matching strength",
                )
            with gr.Row():
                cpu_offload = gr.Checkbox(value=True, label="CPU offload")
                local_files_only = gr.Checkbox(value=True, label="Local models only")

        with gr.Accordion("Outpaint LoRA", open=True):
            gr.Markdown(
                "Source: `fal/flux-2-klein-4B-outpaint-lora`. When enabled, the masked model input is filled with green (`#00FF00`). "
                "This affects the model input, not just the preview overlay. Start with a scale of 1.1."
            )
            with gr.Row():
                lora_enabled = gr.Checkbox(value=True, label="Enable Outpaint LoRA")
                green_screen_input = gr.Checkbox(value=True, label="Use green mask in model input")
                lora_scale = gr.Slider(0.0, 2.0, value=1.1, step=0.05, label="LoRA Scale")
            direct_overlay_enabled = gr.Checkbox(
                value=True,
                label="Preserve generated subject; blend outer background (recommended)",
            )
            lora_path = gr.Textbox(
                label="LoRA directory or Hugging Face ID",
                value=flux_inpaint.DEFAULT_LORA_PATH,
            )
            lora_weight_name = gr.Textbox(
                label="LoRA weight filename",
                value=flux_inpaint.DEFAULT_LORA_WEIGHT_NAME,
            )
            gr.Markdown(
                f"To download weights, enter `{flux_inpaint.OUTPAINT_LORA_REPO}` and disable Local models only."
            )

        with gr.Accordion("Seam & Halo Correction", open=True):
            gr.Markdown(
                "Match local background colors around the mask using a spatially varying low-frequency field. "
                "Sky, clouds, walls, and shadows receive local corrections within the editing mask."
            )
            with gr.Row():
                halo_fix_enabled = gr.Checkbox(value=True, label="Enable boundary color correction")
                halo_luminance_only = gr.Checkbox(value=False, label="Correct luminance only")
                halo_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Correction strength")
            with gr.Row():
                halo_ring_width = gr.Slider(4, 96, value=32, step=2, label="Outer sampling ring width (px)")
                halo_fade_radius = gr.Slider(16, 512, value=192, step=16, label="Color propagation radius (px)")
                halo_interior_strength = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Minimum interior correction")
            with gr.Row():
                texture_blend_enabled = gr.Checkbox(value=True, label="Enable multiband texture blending")
                texture_blend_levels = gr.Slider(1, 6, value=4, step=1, label="Texture pyramid levels")
                green_despill_enabled = gr.Checkbox(value=True, label="Suppress green mask spill")
                green_despill_width = gr.Slider(2, 48, value=24, step=2, label="Green/magenta spill detection width (px)")
            with gr.Row():
                background_similarity_threshold = gr.Slider(
                    1,
                    128,
                    value=32,
                    step=1,
                    label="Background color similarity threshold",
                )
                background_restore_strength = gr.Slider(
                    0.0,
                    1.0,
                    value=0.85,
                    step=0.05,
                    label="Similar-background restoration strength",
                )
            gr.Markdown(
                "For large sky/building masks, try strength 1.0, ring width 32px, propagation 192–256px, and interior correction 0.35. "
                "Try 4–5 texture levels for cloud seams. A higher similarity threshold classifies more generated pixels as background. "
                "Higher restoration strength keeps similar background regions closer to the original."
            )

        with gr.Accordion("SAM Settings", open=False):
            backend = gr.Radio(["sam2", "auto", "sam1"], value="sam2", label="SAM backend")
            sam2_path = gr.Textbox(label="SAM2 directory or Hugging Face ID", value=sam_segmenter.DEFAULT_SAM2_PATH)
            sam1_path = gr.Textbox(label="SAM1 vit_h checkpoint", value=sam_segmenter.DEFAULT_SAM1_PATH)
            ref_feather = gr.State(0.0)
            with gr.Row():
                ref_padding = gr.Slider(0, 0.5, value=0.10, step=0.01, label="Reference crop padding")
                gr.Markdown("Reference cutouts use a binary mask without feathering.")

        target_draw_tab.select(
            lambda: TARGET_DRAW_MODE,
            inputs=None,
            outputs=[target_mask_mode],
            queue=False,
        )
        target_sam_tab.select(
            lambda: TARGET_SAM_MODE,
            inputs=None,
            outputs=[target_mask_mode],
            queue=False,
        )
        target_sam_upload.change(
            reset_target_sam,
            [target_sam_upload],
            [
                target_sam_source,
                target_sam_points,
                target_sam_labels,
                target_sam_mask,
                target_sam_preview,
                target_mask_preview,
                target_sam_status,
            ],
        )
        target_sam_preview.select(
            select_target_point,
            [
                target_sam_source,
                target_sam_points,
                target_sam_labels,
                target_point_mode,
                backend,
                sam2_path,
                sam1_path,
            ],
            [
                target_sam_points,
                target_sam_labels,
                target_sam_mask,
                target_sam_preview,
                target_mask_preview,
                target_sam_status,
            ],
        )
        undo_target.click(
            undo_target_point,
            [target_sam_source, target_sam_points, target_sam_labels, backend, sam2_path, sam1_path],
            [target_sam_points, target_sam_labels, target_sam_mask, target_sam_preview, target_mask_preview, target_sam_status],
        )
        clear_target.click(
            clear_target_points,
            [target_sam_source],
            [target_sam_points, target_sam_labels, target_sam_mask, target_sam_preview, target_mask_preview, target_sam_status],
        )

        reference_upload.change(
            reset_reference,
            [reference_upload],
            [reference_base, reference_points, reference_labels, reference_ready, reference_mask, reference_preview, reference_cutout, sam_status],
        )
        reference_preview.select(
            select_reference_point,
            [reference_base, reference_points, reference_labels, point_mode, backend, sam2_path, sam1_path, ref_padding, ref_feather],
            [reference_points, reference_labels, reference_ready, reference_mask, reference_preview, reference_cutout, sam_status],
        )
        undo_point.click(
            undo_reference_point,
            [reference_base, reference_points, reference_labels, backend, sam2_path, sam1_path, ref_padding, ref_feather],
            [reference_points, reference_labels, reference_ready, reference_mask, reference_preview, reference_cutout, sam_status],
        )
        clear_points.click(
            clear_reference_points,
            [reference_base],
            [reference_points, reference_labels, reference_ready, reference_mask, reference_preview, reference_cutout, sam_status],
        )
        whole_reference.click(
            use_whole_reference,
            [reference_base],
            [reference_ready, reference_mask, reference_preview, reference_cutout, sam_status],
        )
        generate.click(
            start_background_edit,
            [
                source_data, mask_data, target_mask_mode, target_sam_source, target_sam_mask,
                reference_ready, model_path, prompt, steps, guidance, strength, seed,
                max_roi_side, roi_context,
                bbox_expand_left, bbox_expand_right, bbox_expand_top, bbox_expand_bottom,
                use_roi, model_mask_expand, model_mask_blur, blend_expand,
                direct_edge_feather,
                direct_seam_color_match, direct_seam_color_strength,
                cpu_offload, local_files_only, lora_enabled, lora_path,
                lora_weight_name, lora_scale, green_screen_input, direct_overlay_enabled, halo_fix_enabled,
                halo_luminance_only, halo_ring_width, halo_strength, halo_fade_radius,
                halo_interior_strength,
                green_despill_enabled, green_despill_width, texture_blend_enabled,
                texture_blend_levels, background_similarity_threshold, background_restore_strength,
            ],
            [
                active_job_id, status, generation_progress, generation_metrics,
                post_final_preview, final_download,
                generation_poll, generate,
                generated_sam_preview,
                generated_sam_points,
                generated_sam_labels,
                generated_sam_mask,
                generated_sam_mask_preview,
                generated_sam_status,
                original_sam_preview,
                original_sam_points,
                original_sam_labels,
                original_sam_mask,
                original_sam_mask_preview,
                original_sam_status,
                consistency_background_preview,
                consistency_status,
            ],
        )
        generation_poll.click(
            poll_background_edit,
            [active_job_id],
            [
                output_gallery, status, generation_progress, generation_metrics,
                post_final_preview, final_download,
                generation_poll, generate,
                generated_sam_preview,
                generated_sam_status,
                original_sam_preview,
                original_sam_status,
            ],
            queue=False,
            show_progress="hidden",
        )
        generated_sam_preview.select(
            select_generated_subject_point,
            [
                active_job_id,
                generated_sam_points,
                generated_sam_labels,
                generated_sam_point_mode,
                backend,
                sam2_path,
                sam1_path,
                generated_object_feather,
                generated_object_expand,
                generated_seam_radius,
                generated_seam_strength,
            ],
            [
                generated_sam_points,
                generated_sam_labels,
                generated_sam_mask,
                generated_sam_preview,
                generated_sam_mask_preview,
                post_final_preview,
                final_download,
                generated_sam_status,
                generation_metrics,
            ],
        )
        generated_object_expand.release(
            _predict_generated_subject,
            [
                active_job_id,
                generated_sam_points,
                generated_sam_labels,
                backend,
                sam2_path,
                sam1_path,
                generated_object_feather,
                generated_object_expand,
                generated_seam_radius,
                generated_seam_strength,
            ],
            [
                generated_sam_mask,
                generated_sam_preview,
                generated_sam_mask_preview,
                post_final_preview,
                final_download,
                generated_sam_status,
                generation_metrics,
            ],
        )
        undo_generated_sam.click(
            undo_generated_subject_point,
            [
                active_job_id,
                generated_sam_points,
                generated_sam_labels,
                backend,
                sam2_path,
                sam1_path,
                generated_object_feather,
                generated_object_expand,
                generated_seam_radius,
                generated_seam_strength,
            ],
            [
                generated_sam_points,
                generated_sam_labels,
                generated_sam_mask,
                generated_sam_preview,
                generated_sam_mask_preview,
                post_final_preview,
                final_download,
                generated_sam_status,
                generation_metrics,
            ],
        )
        clear_generated_sam.click(
            clear_generated_subject_points,
            [active_job_id],
            [
                generated_sam_points,
                generated_sam_labels,
                generated_sam_mask,
                generated_sam_preview,
                generated_sam_mask_preview,
                post_final_preview,
                final_download,
                generated_sam_status,
                generation_metrics,
            ],
        )
        original_sam_preview.select(
            select_original_subject_point,
            [
                active_job_id,
                original_sam_points,
                original_sam_labels,
                original_sam_point_mode,
                backend,
                sam2_path,
                sam1_path,
            ],
            [
                original_sam_points,
                original_sam_labels,
                original_sam_mask,
                original_sam_preview,
                original_sam_mask_preview,
                original_sam_status,
            ],
        )
        undo_original_sam.click(
            undo_original_subject_point,
            [
                active_job_id,
                original_sam_points,
                original_sam_labels,
                backend,
                sam2_path,
                sam1_path,
            ],
            [
                original_sam_points,
                original_sam_labels,
                original_sam_mask,
                original_sam_preview,
                original_sam_mask_preview,
                original_sam_status,
            ],
        )
        clear_original_sam.click(
            clear_original_subject_points,
            [active_job_id],
            [
                original_sam_points,
                original_sam_labels,
                original_sam_mask,
                original_sam_preview,
                original_sam_mask_preview,
                original_sam_status,
            ],
        )
        run_consistency.click(
            run_background_consistency,
            [
                active_job_id,
                generated_sam_mask,
                original_sam_mask,
                consistency_original_cleanup_expand,
                consistency_generated_fill_expand,
                consistency_fill_radius,
                consistency_color_lock,
                consistency_color_radius,
                consistency_color_max_delta,
                consistency_background_feather,
                generated_object_feather,
                generated_object_expand,
                generated_seam_radius,
                generated_seam_strength,
            ],
            [
                consistency_background_preview,
                post_final_preview,
                final_download,
                consistency_status,
                generation_metrics,
            ],
        )
        demo.load(fn=None, inputs=None, outputs=None, _js=MASK_EDITOR_JS)
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    if not gr.__version__.startswith("3."):
        raise RuntimeError(f"This project requires Gradio 3. Installed: {gr.__version__}. Install gradio==3.39.0.")
    args = parse_args()
    build_demo().queue(concurrency_count=1).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
