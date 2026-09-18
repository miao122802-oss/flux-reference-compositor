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

TARGET_DRAW_MODE = "手动画绿色区域"
TARGET_SAM_MODE = "SAM 自动分割 Target"


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
    safe_message = html.escape(message or "等待任务状态…")
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
            '<div class="task-metrics empty"><div class="task-metrics-title">任务统计</div>'
            '<div class="task-metrics-empty">生成完成后显示输入/输出尺寸、倍率、耗时和峰值显存。</div></div>'
        )

    def size(name: str) -> str:
        width, height = metrics[name]
        return f"{int(width)} × {int(height)}"

    scale = f"{metrics['output_scale_x']:.3f}× / {metrics['output_scale_y']:.3f}×"
    cards = [
        ("目标原图", size("source_input_size"), "用户输入尺寸"),
        ("Reference", size("reference_input_size"), f"送模 {size('reference_model_size') }"),
        ("FLUX ROI", size("model_input_size"), f"原始 ROI {size('roi_source_size')}"),
        ("最终输出", size("output_size"), f"宽/高倍率 {scale}"),
        ("端到端耗时", f"{metrics['end_to_end_seconds']:.2f} s", f"FLUX 内部 {metrics['flux_total_seconds']:.2f} s"),
        ("任务峰值显存", f"{metrics['peak_reserved_gib']:.2f} GiB", f"allocated {metrics['peak_allocated_gib']:.2f} GiB"),
    ]
    card_html = "".join(
        f'<div class="metric-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>'
        f'<small>{html.escape(note)}</small></div>'
        for label, value, note in cards
    )
    breakdown_parts = [
        f"预处理 {metrics['preprocess_seconds']:.2f}s",
        f"模型加载/准备 {metrics['pipeline_prepare_seconds']:.2f}s",
    ]
    if metrics.get("background_diffusion_seconds", 0.0) > 0:
        breakdown_parts.append(
            f"FLUX背景 {metrics['background_diffusion_seconds']:.2f}s"
        )
    breakdown_parts.extend(
        [
            f"FLUX物体 {metrics.get('object_diffusion_seconds', metrics['diffusion_seconds']):.2f}s",
            f"恢复尺寸与合成 {metrics['postprocess_seconds']:.2f}s",
        ]
    )
    if metrics.get("segmentation_enabled", False):
        breakdown_parts.append(
            f"物体分割 {metrics.get('segmentation_seconds', 0.0):.2f}s"
        )
    else:
        breakdown_parts.append("生成后 SAM 等待用户点选")
    breakdown = "　｜　".join(breakdown_parts)
    return (
        '<div class="task-metrics"><div class="task-metrics-title">任务统计</div>'
        f'<div class="task-metrics-grid">{card_html}</div>'
        f'<div class="metrics-breakdown">{html.escape(breakdown)}</div></div>'
    )


def decode_data_url(data: str, name: str) -> Image.Image:
    if not data:
        raise gr.Error(f"请先提供{name}。")
    try:
        encoded = data.split(",", 1)[1] if "," in data else data
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            return ImageOps.exif_transpose(image).copy()
    except Exception as exc:
        raise gr.Error(f"无法读取{name}：{exc}") from exc


def _display(image: Image.Image | None, *, mask: bool = False):
    if image is None:
        return None
    return ui_preview.display_preview(image, mask=mask)


def _display_click_to_full(index, image: Image.Image) -> list[float]:
    size = ui_preview.preview_size(image.size)
    return ui_preview.display_point_to_full(index, image.size, size)


def reset_reference(image):
    if image is None:
        return None, [], [], None, None, None, None, "等待 Reference。"
    image = ImageOps.exif_transpose(image).convert("RGB")
    return (
        image,
        [],
        [],
        None,
        None,
        _display(image),
        None,
        "✅ Reference 原图已加载。请在中间图片点击一个或多个绿色前景点。",
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
        raise RuntimeError("FLUX 正在生成，暂时不能重新加载 SAM；请等待当前任务完成。")
    point_preview = sam_segmenter.draw_points(image, points, labels)
    if not any(label == 1 for label in labels):
        return None, None, _display(point_preview), None, "已记录背景排除点，还需要至少一个绿色前景点。"
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
        f"✅ {kind.upper()} 已提取主体｜score={score:.4f}｜"
        f"前景点 {fg_count} 个，背景点 {bg_count} 个。右侧白底图就是实际 Reference。"
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
        raise gr.Error("请先上传 Reference 原图。")
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("未获取到点击坐标，请重新点击图片。")
    full_point = _display_click_to_full(index, image)
    new_points = [list(item) for item in (points or [])] + [full_point]
    new_labels = [int(item) for item in (labels or [])] + [1 if point_mode == "前景点（绿色）" else 0]
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
        return new_points, new_labels, None, None, preview, None, f"❌ SAM 分割失败：{exc}"


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
        return [], [], None, None, None, None, "请先上传 Reference。"
    new_points = [list(item) for item in (points or [])][:-1]
    new_labels = [int(item) for item in (labels or [])][:-1]
    if not new_points:
        return [], [], None, None, _display(image), None, "已撤销全部提示点，请重新点选主体。"
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
        return new_points, new_labels, ready, mask, preview, cutout, "↶ 已撤销最后一个点。" + status
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, None, preview, None, f"撤销后等待有效前景点：{exc}"


def clear_reference_points(image):
    if image is None:
        return [], [], None, None, None, None, "请先上传 Reference。"
    return [], [], None, None, _display(image), None, "提示点已清空，请重新点击绿色前景点。"


def use_whole_reference(image):
    if image is None:
        raise gr.Error("请先上传 Reference。")
    image = ImageOps.exif_transpose(image).convert("RGB")
    mask = Image.new("L", image.size, 255)
    return image, mask, _display(image), _display(image), "✅ 已跳过 SAM，整张图片将作为独立 Reference。"


def reset_target_sam(image):
    if image is None:
        return None, [], [], None, None, None, "等待上传 Target 原图。"
    image = ImageOps.exif_transpose(image).convert("RGB")
    return (
        image,
        [],
        [],
        None,
        _display(image),
        None,
        "✅ Target 原图已加载。请在中间图片点击绿色前景点选择要修改的对象。",
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
        raise RuntimeError("FLUX 正在生成，暂时不能重新加载 SAM；请等待当前任务完成。")
    point_preview = sam_segmenter.draw_points(image, points, labels)
    if not any(label == 1 for label in labels):
        return None, _display(point_preview), None, "已记录背景排除点，还需要至少一个绿色前景点。"
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
        f"✅ Target Mask 已生成｜{kind.upper()} score={score:.4f}｜"
        f"前景点 {fg_count} 个，背景点 {bg_count} 个。白色区域就是最终修改范围。"
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
        raise gr.Error("请先上传 Target 原图。")
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("未获取到点击坐标，请重新点击图片。")
    full_point = _display_click_to_full(index, image)
    new_points = [list(item) for item in (points or [])] + [full_point]
    new_labels = [int(item) for item in (labels or [])] + [1 if point_mode == "前景点（绿色）" else 0]
    try:
        mask, preview, mask_preview, status = _predict_target_mask(
            image, new_points, new_labels, backend, sam2_path, sam1_path
        )
        return new_points, new_labels, mask, preview, mask_preview, status
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, preview, None, f"❌ Target SAM 分割失败：{exc}"


def undo_target_point(image, points, labels, backend, sam2_path, sam1_path):
    if image is None:
        return [], [], None, None, None, "请先上传 Target 原图。"
    new_points = [list(item) for item in (points or [])][:-1]
    new_labels = [int(item) for item in (labels or [])][:-1]
    if not new_points:
        return [], [], None, _display(image), None, "已撤销全部 Target 提示点。"
    try:
        mask, preview, mask_preview, status = _predict_target_mask(
            image, new_points, new_labels, backend, sam2_path, sam1_path
        )
        return new_points, new_labels, mask, preview, mask_preview, "↶ 已撤销最后一个点。" + status
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, preview, None, f"撤销后等待有效前景点：{exc}"


def clear_target_points(image):
    if image is None:
        return [], [], None, None, None, "请先上传 Target 原图。"
    return [], [], None, _display(image), None, "Target 提示点和 Mask 已清空，请重新点选。"


def _completed_job_session(job_id: str) -> dict:
    if not job_id:
        raise gr.Error("请先完成一次 FLUX 生成。")
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None or job.get("state") != "completed" or job.get("session") is None:
            raise gr.Error("当前 FLUX 任务尚未完成，暂时不能进行生成后 SAM 分割。")
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
            "已记录背景排除点，还需要至少一个绿色前景点。",
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
        f"✅ 生成后 {kind.upper()} 主体分割并合成完成｜score={score:.4f}｜"
        f"前景点 {fg_count} 个，背景点 {bg_count} 个｜"
        f"羽化={float(object_feather):.1f}px，Mask 调整={int(object_edge_expand):+d}px。"
        "清理后的二值 SAM 主体已贴回；剔除的外圈会归入第一次 FLUX 背景校色区域。"
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
        raise gr.Error("未获取到点击坐标，请重新点击生成图。")
    full_point = _display_click_to_full(index, image)
    new_points = [list(item) for item in (points or [])] + [full_point]
    new_labels = [int(item) for item in (labels or [])] + [
        1 if point_mode == "前景点（绿色）" else 0
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
        return new_points, new_labels, None, preview, None, gr.update(), gr.update(), f"❌ 生成后 SAM 分割失败：{exc}", gr.update()


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
    return new_points, new_labels, mask, preview, mask_preview, final, download, "↶ 已撤销最后一点。" + status, metrics


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
        "生成后 SAM 点与 Mask 已清空，已恢复 FLUX 默认合成结果。",
        _metrics_html(metrics),
    )


def _predict_original_subject(job_id, points, labels, backend, sam2_path, sam1_path):
    session = _completed_job_session(job_id)
    image = session["output"]["source_roi"]
    point_preview = sam_segmenter.draw_points(image, points, labels)
    if not any(int(label) == 1 for label in labels):
        return None, _display(point_preview), None, "已记录背景排除点，还需要至少一个绿色前景点。"
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
        f"✅ 原图 {kind.upper()} 主体分割完成｜score={score:.4f}｜"
        f"前景点 {fg_count} 个，背景点 {bg_count} 个。"
    )
    return mask, _display(preview), _display(mask, mask=True), status


def select_original_subject_point(
    job_id, points, labels, point_mode, backend, sam2_path, sam1_path, evt: gr.SelectData
):
    session = _completed_job_session(job_id)
    image = session["output"]["source_roi"]
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("未获取到点击坐标，请重新点击原图 ROI。")
    new_points = [list(item) for item in (points or [])] + [_display_click_to_full(index, image)]
    new_labels = [int(item) for item in (labels or [])] + [
        1 if point_mode == "前景点（绿色）" else 0
    ]
    try:
        mask, preview, mask_preview, status = _predict_original_subject(
            job_id, new_points, new_labels, backend, sam2_path, sam1_path
        )
        return new_points, new_labels, mask, preview, mask_preview, status
    except Exception as exc:
        preview = _display(sam_segmenter.draw_points(image, new_points, new_labels))
        return new_points, new_labels, None, preview, None, f"❌ 原图 SAM 分割失败：{exc}"


def undo_original_subject_point(job_id, points, labels, backend, sam2_path, sam1_path):
    new_points = [list(item) for item in (points or [])][:-1]
    new_labels = [int(item) for item in (labels or [])][:-1]
    if not new_points:
        return clear_original_subject_points(job_id)
    mask, preview, mask_preview, status = _predict_original_subject(
        job_id, new_points, new_labels, backend, sam2_path, sam1_path
    )
    return new_points, new_labels, mask, preview, mask_preview, "↶ 已撤销最后一点。" + status


def clear_original_subject_points(job_id):
    session = _completed_job_session(job_id)
    session.pop("original_object_mask", None)
    return (
        [], [], None, _display(session["output"]["source_roi"]), None,
        "原图主体 SAM 点与 Mask 已清空。",
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
        raise gr.Error("请先在生成图中用 SAM 选出新主体。")
    if original_subject_mask is None:
        raise gr.Error("请先在原图 ROI 中用 SAM 选出旧主体。")
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
        raise gr.Error(f"第一次 FLUX 背景校色失败：{exc}") from exc
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
        f"✅ 背景处理完成｜校色背景只合成一次｜48/32/8 外圈合成｜无第二次 FLUX｜"
        f"直接校色第一次 raw_roi｜ROI={consistency['model_size'][0]}×{consistency['model_size'][1]}｜"
        f"耗时={consistency['seconds']:.2f}s｜背景低频色差 "
        f"{consistency['color_lock_info']['error_before']:.2f}→"
        f"{consistency['color_lock_info']['error_after']:.2f}｜背景外侧过渡="
        f"{float(background_edge_feather):.1f}px｜主体边缘清理="
        f"{int(object_edge_expand):+d}px。编辑区内的新主体外使用"
        f"第一次 FLUX 校色背景｜旧主体清理外扩={int(original_cleanup_expand)}px｜"
        f"新主体填底外扩={int(generated_fill_expand)}px；最后贴回完整新主体，临时底板不参与合成。"
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
        (mask_box_preview, "① FLUX 生成图 + 用户 Mask 黄色外接框"),
        (output["core_roi"], "② 用户最终合成 Mask（白色=生成区域）"),
        (output["full_background"], "③ 原始 Target（不再生成背景底板）"),
        (output["model_input"], "④ 模型实际输入（纯绿色=LoRA区）"),
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
            f"flux_inpaint.py 与 app.py 版本不一致，仍缺少参数：{names}。"
            "请确认两个文件来自同一项目目录并重新启动 Gradio。"
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
    edit_mode,
    progress_callback=None,
):
    request_started = time.perf_counter()

    def report(message: str, value: float) -> None:
        if progress_callback is not None:
            progress_callback(message, value)

    if reference_ready is None:
        raise gr.Error("Reference 尚未就绪：请完成 SAM 主体提取，或点击“整张直接作为 Reference”。")
    if target_mask_mode == TARGET_SAM_MODE:
        if target_sam_source is None:
            raise gr.Error("请在“Target SAM 自动分割”标签中上传 Target 原图。")
        if target_sam_mask is None:
            raise gr.Error("Target SAM Mask 尚未生成，请先用绿色前景点选择要修改的对象。")
        source = ImageOps.exif_transpose(target_sam_source).convert("RGB")
        mask = ImageOps.exif_transpose(target_sam_mask).convert("L")
        print("[任务] Target Mask 来源：SAM 自动分割", flush=True)
    else:
        source = decode_data_url(source_data, "目标原图").convert("RGB")
        mask = decode_data_url(mask_data, "目标 Mask").convert("L")
        print("[任务] Target Mask 来源：页面手绘", flush=True)
    if mask.resize(source.size, Image.Resampling.NEAREST).getbbox() is None:
        raise gr.Error("绿色目标区域为空，请先在目标图上画框或使用画笔。")
    report("输入检查完成", 0.02)

    try:
        sam_segmenter.release_segmenter()
        output = _get_current_flux_editor()(
            source,
            mask,
            reference_ready,
            model_path=model_path.strip(),
            prompt=prompt,
            edit_mode=edit_mode,
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
        raise gr.Error(f"FLUX生成失败：{exc}") from exc

    report("单次 FLUX 替换已完成，正在按用户 Mask 直接覆盖…", 0.96)
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
        f"ON · 边界误差 {halo_info['edge_error_before']:.1f}→{halo_info['edge_error_after']:.1f} · "
        f"纹理L{halo_info['texture_blend_levels']} · "
        f"去绿={'ON' if halo_info['despill']['applied'] else '无残边/未应用'}"
        if halo_info["applied"]
        else "OFF/未应用"
    )
    status = (
        f"✅ FLUX 生成完成｜模式={output['edit_mode']}｜FLUX {output['flux_pass_count']}次｜"
        f"seed={output['seed']}｜ROI={output['roi_box_text']}｜"
        f"黄色框={output['location_box_text']}｜"
        f"粘贴外扩={output['blend_expand']}px｜"
        f"外圈柔化={output['direct_edge_feather']:.0f}px｜"
        f"接缝色匹配={'ON' if output['direct_seam_color_match'] else 'OFF'}"
        f"({output['direct_seam_color_strength']:.2f})｜"
        f"推理尺寸={width}×{height}｜端到端耗时={metrics['end_to_end_seconds']:.2f}s｜"
        f"峰值显存={metrics['peak_reserved_gib']:.2f}GiB｜LoRA={lora_status}｜Halo={halo_status}"
        f"<br>输出={'主体保留原始FLUX＋外扩背景接缝融合' if output['direct_overlay_enabled'] else '边界校色＋相似背景还原'}"
        f"｜生成后SAM=等待手动点选｜生成后阴影合成=关闭"
        f"<br>实际 Prompt：{output['prompt']}"
    )
    print(
        f"[统计] Gradio端到端任务耗时={metrics['end_to_end_seconds']:.2f}s",
        flush=True,
    )
    session = {
        "output": output,
        "gallery": gallery,
        "post_final": final,
    }
    return gallery, status, _metrics_html(metrics), session


def _run_background_edit(job_id: str, args: tuple) -> None:
    _update_job(job_id, state="running", message="后台任务已启动，正在检查输入…", progress=2)
    print(f"[任务 {job_id[:8]}] 后台生成任务开始", flush=True)
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
            message="全部生成完成。",
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
        print(f"[任务 {job_id[:8]}] 全部完成", flush=True)
    except Exception as exc:
        traceback.print_exc()
        message = f"生成失败：{exc}"
        _update_job(job_id, state="failed", message=message, status=message)
        _prune_finished_jobs(10)
        print(f"[任务 {job_id[:8]}] {message}", flush=True)


def start_background_edit(*args):
    if _generation_active():
        raise gr.Error("已经有一个 FLUX 任务正在运行，请等待它完成，不要重复提交。")
    job_id = uuid.uuid4().hex
    message = "任务已提交，正在启动后台线程…"
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
    print(f"[任务 {job_id[:8]}] 已提交", flush=True)
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
        "等待 FLUX 完成后，在生成图上点击需要保留的主体。",
        None,
        [],
        [],
        None,
        None,
        "等待 FLUX 完成后，在原图 ROI 上点击需要移除的旧主体。",
        None,
        "请先完成上面的新主体和旧主体两个 SAM Mask。",
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
        message = "任务状态已丢失，请重新提交。"
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
    message = job.get("message", "后台任务正在运行…")
    if state == "completed":
        session = job.get("session") or {}
        output = session.get("output") or {}
        return (
            job.get("gallery"),
            job.get("status", "✅ 生成完成。"),
            _progress_html(100, "全部生成完成。", "completed"),
            job.get("metrics_html", _metrics_html()),
            job.get("post_final"),
            job.get("final_file"),
            gr.update(interactive=False),
            gr.update(interactive=True),
            _display(output.get("raw_roi")),
            "✅ FLUX 已完成。请在左侧生成图点击绿色前景点，让 SAM 提取真正需要粘贴的主体。",
            _display(output.get("source_roi")),
            "✅ 请在原图 ROI 点击绿色前景点，让 SAM 提取并遮掉旧主体。",
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
        "FLUX 正在生成，完成后才能使用生成后 SAM。",
        gr.update(),
        "FLUX 正在生成，完成后才能选择原图旧主体。",
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
            '<div class="app-hero"><h1>FLUX Reference Compositor · 参考图引导的局部替换</h1>'
            '<p>第一次生成仍使用 Outpaint LoRA；之后不再运行任何扩散模型，只校正第一次背景的色差并重新合成。</p>'
            '<div class="input-map"><span>Reference：放什么</span><b>＋</b>'
            '<span>绿色 Mask：放哪里</span><b>＋</b><span>文字：可选要求</span>'
            '<b>→</b><span>单次 FLUX 直接替换</span><b>→</b><span>Mask 合成</span></div></div>'
        )

        gr.HTML(step_title(1, "准备 Reference 物体", "绿色点选主体，红色点排除背景；白底预览是实际送入模型的 Reference"))
        with gr.Row(equal_height=True, elem_classes="reference-row"):
            reference_upload = gr.Image(
                label="A. 上传 Reference 原图",
                source="upload",
                type="pil",
                height=315,
                scale=2,
            )
            with gr.Column(scale=2, min_width=320, elem_classes="reference-panel"):
                reference_preview = gr.Image(
                    label="B. 点击这里添加提示点",
                    type="pil",
                    interactive=True,
                    height=255,
                )
                point_mode = gr.Radio(
                    ["前景点（绿色）", "背景排除点（红色）"],
                    value="前景点（绿色）",
                    label="当前点击类型",
                    elem_classes="sam-mode",
                )
                with gr.Row():
                    undo_point = gr.Button("↶ 撤销一点")
                    clear_points = gr.Button("清空提示点")
            reference_cutout = gr.Image(
                label="C. 实际 Reference（白底主体）",
                type="pil",
                interactive=False,
                height=315,
                scale=2,
                elem_classes="reference-ready",
            )
        with gr.Row():
            whole_reference = gr.Button("Reference 已经抠好？整张直接作为 Reference")
            sam_status = gr.Markdown("等待上传 Reference 原图。", elem_classes="status-panel")

        gr.HTML('<div class="workflow-section">' + step_title(2, "确定 Target 修改区域", "可手动画绿色区域，也可以用 SAM 前景点自动分割要修改的对象") + '</div>')
        with gr.Tabs(selected="target-draw", elem_classes="target-mode-tabs"):
            with gr.TabItem("手动画绿色区域", id="target-draw") as target_draw_tab:
                gr.HTML(EDITOR_HTML)
            with gr.TabItem("SAM 自动分割 Target", id="target-sam") as target_sam_tab:
                gr.Markdown(
                    "上传 Target 原图，在对象内部点击绿色前景点；分割范围过大时添加红色背景排除点。",
                    elem_classes="hint-panel",
                )
                with gr.Row(equal_height=True, elem_classes="reference-row"):
                    target_sam_upload = gr.Image(
                        label="A. 上传 Target 原图",
                        source="upload",
                        type="pil",
                        height=330,
                        scale=2,
                    )
                    with gr.Column(scale=2, min_width=320, elem_classes="reference-panel"):
                        target_sam_preview = gr.Image(
                            label="B. 点击选择要修改的对象",
                            type="pil",
                            interactive=True,
                            height=265,
                        )
                        target_point_mode = gr.Radio(
                            ["前景点（绿色）", "背景排除点（红色）"],
                            value="前景点（绿色）",
                            label="当前点击类型",
                            elem_classes="sam-mode",
                        )
                        with gr.Row():
                            undo_target = gr.Button("↶ 撤销一点")
                            clear_target = gr.Button("清空 Target 点")
                    target_mask_preview = gr.Image(
                        label="C. Target 二值 Mask（白色=修改）",
                        type="pil",
                        interactive=False,
                        height=330,
                        scale=2,
                    )
                target_sam_status = gr.Markdown(
                    "等待上传 Target 原图。",
                    elem_classes="status-panel",
                )

        gr.HTML('<div class="workflow-section">' + step_title(3, "添加可选要求并生成", "文字可留空；系统仍会自动执行 Reference 物体迁移") + '</div>')
        edit_mode = gr.Radio(
            [flux_inpaint.EDIT_MODE_INSERT, flux_inpaint.EDIT_MODE_REPLACE],
            value=flux_inpaint.EDIT_MODE_REPLACE,
            label="合成模式（两种模式均为单次 FLUX；替换模式不再生成背景底板）",
        )
        with gr.Row(elem_classes="generate-row"):
            prompt = gr.Textbox(
                label="可选文字要求",
                placeholder="例如：朝向左侧、放在桌面上、保持产品标签；不输入也可以",
                lines=3,
                scale=5,
            )
            with gr.Column(scale=2, min_width=290, elem_classes="generate-panel"):
                gr.Markdown("绿色目标区和右侧白底 Reference 都确认后即可生成。", elem_classes="hint-panel")
                generate = gr.Button("开始 Reference 局部重绘", variant="primary", elem_classes="generate-action")

        status = gr.Markdown("等待完成前两步。", elem_classes="status-panel")
        generation_progress = gr.HTML(_progress_html(0, "等待 FLUX 生成任务。"))
        generation_metrics = gr.HTML(_metrics_html())
        generation_poll = gr.Button(
            "刷新生成状态",
            elem_id="green-generation-poll",
            elem_classes="poll-trigger",
            interactive=False,
        )
        output_gallery = gr.Gallery(
            label="轻量诊断图（原尺寸最终图请在下方下载）",
            columns=3,
            height=440,
            elem_classes="result-gallery",
        )
        gr.HTML(
            '<div class="workflow-section">' + step_title(
                4,
                "用 SAM 提取生成主体并重新合成",
                "在生成图点击绿色前景点选择新主体；分割过大时添加红色背景排除点",
            ) + '</div>'
        )
        with gr.Row(equal_height=True, elem_classes="reference-row"):
            with gr.Column(scale=2):
                generated_sam_preview = gr.Image(
                    label="A. 点击 FLUX 生成图选择要保留的主体",
                    type="pil",
                    interactive=True,
                    height=360,
                )
                generated_sam_point_mode = gr.Radio(
                    ["前景点（绿色）", "背景排除点（红色）"],
                    value="前景点（绿色）",
                    label="当前点击类型",
                    elem_classes="sam-mode",
                )
                with gr.Row():
                    undo_generated_sam = gr.Button("↶ 撤销一点")
                    clear_generated_sam = gr.Button("清空生成后 SAM 点并恢复默认结果")
            generated_sam_mask_preview = gr.Image(
                label="B. 新主体 SAM Mask（白色=最终粘贴）",
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
                label="主体边缘调整 px（默认 0；仅修正 SAM 轮廓）"
            )
            gr.Markdown("主体 Mask 内部固定为二值合成：羽化=0、接缝高斯=0。")
        generated_sam_status = gr.Markdown(
            "等待 FLUX 完成后，在生成图上点击需要保留的主体。",
            elem_classes="status-panel",
        )
        with gr.Row(equal_height=True, elem_classes="reference-row"):
            with gr.Column(scale=2):
                original_sam_preview = gr.Image(
                    label="C. 点击原图 ROI 选择需要移除的旧主体",
                    type="pil",
                    interactive=True,
                    height=360,
                )
                original_sam_point_mode = gr.Radio(
                    ["前景点（绿色）", "背景排除点（红色）"],
                    value="前景点（绿色）",
                    label="当前点击类型",
                    elem_classes="sam-mode",
                )
                with gr.Row():
                    undo_original_sam = gr.Button("↶ 撤销一点")
                    clear_original_sam = gr.Button("清空原图主体 SAM 点")
            original_sam_mask_preview = gr.Image(
                label="D. 原图旧主体 SAM Mask（白色=一致性处理前遮掉）",
                type="pil",
                interactive=False,
                height=360,
                scale=2,
            )
        original_sam_status = gr.Markdown(
            "等待 FLUX 完成后，在原图 ROI 上点击需要移除的旧主体。",
            elem_classes="status-panel",
        )

        gr.HTML(
            '<div class="workflow-section">' + step_title(
                5,
                "直接校色第一次 FLUX 背景（测试版）",
                "第二次基础 FLUX 和一致性 LoRA 均已关闭；保留双 SAM、色差锁定与最终主体贴回",
            ) + '</div>'
        )
        with gr.Accordion("第一次 FLUX 背景校色参数", open=True):
            with gr.Row():
                consistency_original_cleanup_expand = gr.Slider(
                    0, 48, value=12, step=1,
                    label="旧主体 SAM 向外膨胀 px",
                )
                consistency_generated_fill_expand = gr.Slider(
                    0, 48, value=16, step=1,
                    label="新主体条件底板外扩 px",
                )
                consistency_fill_radius = gr.Slider(8, 160, value=48, step=4, label="遮挡区背景估算半径 px")
            with gr.Row():
                consistency_color_lock = gr.Slider(
                    0.0, 1.0, value=0.9, step=0.05, label="Target 背景色差锁定强度"
                )
                consistency_color_radius = gr.Slider(
                    8, 256, value=64, step=8, label="色差场平滑半径 px"
                )
                consistency_color_max_delta = gr.Slider(
                    8, 128, value=72, step=4, label="最大 RGB 校正量"
                )
                consistency_background_feather = gr.Slider(
                    0.0, 32.0, value=12.0, step=0.5,
                    label="旧主体清除区外侧余弦过渡 px（内部保持实心）"
                )
            gr.Markdown(
                "这一阶段只做 CPU 背景校色，不会运行第二次 FLUX。"
                "主体遮挡用于排除色差估算中的主体颜色；完整背景校色后仅调用一次 48/32/8 外圈合成器；"
                "不保留未校色的 direct_final，也不叠加第二个局部背景层，最后贴第一次 FLUX 输出的完整新主体 SAM；"
                "临时去主体底板不会进入最终图。"
            )
        run_consistency = gr.Button("校色第一次背景并贴回完整主体", variant="primary")
        consistency_status = gr.Markdown(
            "请先完成上面的新主体和旧主体两个 SAM Mask。",
            elem_classes="status-panel",
        )
        consistency_background_preview = gr.Image(
            label="第一次 FLUX 校色背景 ROI（无第二次模型生成）",
            type="pil",
            interactive=False,
            height=360,
        )
        post_final_preview = gr.Image(
            label="最终合成（第一次背景校色后贴回完整新主体）",
            type="pil",
            interactive=False,
            height=560,
        )
        final_download = gr.File(
            label="下载原尺寸无损 PNG",
            interactive=False,
        )

        with gr.Accordion("FLUX 高级参数", open=False):
            gr.Markdown("默认适合蒸馏 4B；Base-4B 建议 Steps 28–50、Guidance 4–8。")
            model_path = gr.Textbox(label="模型路径或 Hugging Face ID", value=flux_inpaint.DEFAULT_MODEL_PATH)
            with gr.Row():
                steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                guidance = gr.Slider(0, 10, value=1.0, step=0.1, label="Guidance")
                strength = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Strength")
                seed = gr.Number(value=-1, precision=0, label="Seed（-1 随机）")
            with gr.Row():
                max_roi_side = gr.Slider(512, 2048, value=1024, step=16, label="ROI 最长边")
                roi_context = gr.Slider(0.05, 1.5, value=0.45, step=0.05, label="ROI 上下文比例")
                use_roi = gr.Checkbox(value=True, label="只推理 Mask 周边 ROI")
            gr.Markdown(
                "黄色框默认紧贴绿色 Mask。下面四项按原图像素向外扩展黄色框，并参与 FLUX ROI 定位；"
                "ROI 上下文比例还会在黄色框之外补充模型观察范围。"
            )
            with gr.Row():
                bbox_expand_left = gr.Slider(0, 1024, value=0, step=4, label="黄色框向左外扩 px")
                bbox_expand_right = gr.Slider(0, 1024, value=0, step=4, label="黄色框向右外扩 px")
            with gr.Row():
                bbox_expand_top = gr.Slider(0, 1024, value=0, step=4, label="黄色框向上外扩 px")
                bbox_expand_bottom = gr.Slider(0, 1024, value=0, step=4, label="黄色框向下外扩 px")
            with gr.Row():
                model_mask_expand = gr.Slider(0, 64, value=8, step=1, label="模型 Mask 外扩")
                model_mask_blur = gr.Slider(0, 32, value=8, step=1, label="模型 Mask 模糊")
                blend_expand = gr.Slider(
                    0,
                    128,
                    value=48,
                    step=1,
                    label="生成图粘贴向外扩展 px（四周）",
                )
                direct_edge_feather = gr.Slider(
                    0,
                    64,
                    value=32,
                    step=1,
                    label="拼接外圈柔化 px（不影响内部主体）",
                )
            with gr.Row():
                direct_seam_color_match = gr.Checkbox(
                    value=True,
                    label="接缝背景低频颜色匹配",
                )
                direct_seam_color_strength = gr.Slider(
                    0.0,
                    1.0,
                    value=0.9,
                    step=0.05,
                    label="接缝颜色匹配强度",
                )
            with gr.Row():
                cpu_offload = gr.Checkbox(value=True, label="CPU offload")
                local_files_only = gr.Checkbox(value=True, label="只读本地模型")

        with gr.Accordion("Outpaint LoRA（纯绿色训练版）", open=True):
            gr.Markdown(
                "官方来源：`fal/flux-2-klein-4B-outpaint-lora`。启用时会把模型 Mask 区域真正改成纯绿色 `#00FF00`，"
                "不是只改变网页覆盖层。权重建议从 1.1 开始。"
            )
            with gr.Row():
                lora_enabled = gr.Checkbox(value=True, label="加载 Outpaint LoRA")
                green_screen_input = gr.Checkbox(value=True, label="模型输入使用纯绿色 Mask")
                lora_scale = gr.Slider(0.0, 2.0, value=1.1, step=0.05, label="LoRA Scale")
            direct_overlay_enabled = gr.Checkbox(
                value=True,
                label="主体保留原始 FLUX，仅融合外扩背景（推荐）",
            )
            lora_path = gr.Textbox(
                label="LoRA 本地目录或 Hugging Face ID",
                value=flux_inpaint.DEFAULT_LORA_PATH,
            )
            lora_weight_name = gr.Textbox(
                label="LoRA 权重文件名",
                value=flux_inpaint.DEFAULT_LORA_WEIGHT_NAME,
            )
            gr.Markdown(
                f"若不使用本地目录，可填写 `{flux_inpaint.OUTPAINT_LORA_REPO}`，并取消“只读本地模型”以允许首次下载。"
            )

        with gr.Accordion("接缝色差 / Halo 修复", open=True):
            gr.Markdown(
                "使用空间变化的低频色场：分别从 Mask 外侧读取相邻原图背景，再与内侧 FLUX 结果比较，"
                "天空、云、墙面和阴影各自校正，不再共用一组 RGB 参数。校正仍严格限制在用户 Mask 内。"
            )
            with gr.Row():
                halo_fix_enabled = gr.Checkbox(value=True, label="启用边界颜色校正")
                halo_luminance_only = gr.Checkbox(value=False, label="仅校正亮度（不校正色偏）")
                halo_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="校正强度")
            with gr.Row():
                halo_ring_width = gr.Slider(4, 96, value=32, step=2, label="局部外环采样宽度 px")
                halo_fade_radius = gr.Slider(16, 512, value=192, step=16, label="空间传播半径 px")
                halo_interior_strength = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="内部最低校正比例")
            with gr.Row():
                texture_blend_enabled = gr.Checkbox(value=True, label="启用云层/纹理多频段融合")
                texture_blend_levels = gr.Slider(1, 6, value=4, step=1, label="纹理融合层级")
                green_despill_enabled = gr.Checkbox(value=True, label="抑制绿色 Mask 残边")
                green_despill_width = gr.Slider(2, 48, value=24, step=2, label="绿/紫残边检测宽度 px")
            with gr.Row():
                background_similarity_threshold = gr.Slider(
                    1,
                    128,
                    value=32,
                    step=1,
                    label="背景相似色差阈值",
                )
                background_restore_strength = gr.Slider(
                    0.0,
                    1.0,
                    value=0.85,
                    step=0.05,
                    label="相似背景还原强度",
                )
            gr.Markdown(
                "这类天空+建筑的大 Mask 推荐：强度 1.0、外环 32px、传播 192–256px、内部 0.35；"
                "云层接缝使用纹理融合 4–5 层；最终融合外扩保持 0。色差阈值越大，越多生成区域会被认作背景；"
                "还原强度越大，相似背景越接近原图。"
            )

        with gr.Accordion("SAM 高级参数", open=False):
            backend = gr.Radio(["sam2", "auto", "sam1"], value="sam2", label="SAM 后端（v4默认SAM2）")
            sam2_path = gr.Textbox(label="SAM2 模型目录或 HF ID", value=sam_segmenter.DEFAULT_SAM2_PATH)
            sam1_path = gr.Textbox(label="SAM1 vit_h checkpoint", value=sam_segmenter.DEFAULT_SAM1_PATH)
            ref_feather = gr.State(0.0)
            with gr.Row():
                ref_padding = gr.Slider(0, 0.5, value=0.10, step=0.01, label="Reference 裁剪留白")
                gr.Markdown("Reference 抠图固定使用二值 Mask，羽化=0。")

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
                edit_mode,
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
        raise RuntimeError(f"本项目要求 Gradio 3，当前为 {gr.__version__}；请安装 gradio==3.39.0。")
    args = parse_args()
    build_demo().queue(concurrency_count=1).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
