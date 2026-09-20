"""Native FLUX.2 Klein inpainting with an independent reference image."""

from __future__ import annotations

import gc
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image, ImageChops

from image_processing import (
    bbox_text,
    composite_roi,
    composite_roi_with_outward_feather,
    despill_green_boundary,
    expand_mask,
    feather_mask,
    fill_mask_from_surroundings,
    final_background_alpha,
    harmonize_roi_boundary_colors,
    limit_reference,
    match_reference_background_colors,
    mask_bbox_with_context,
    normalize_mask,
    paint_mask_green,
    resize_pair_for_model,
    rgb,
)


DEFAULT_MODEL_PATH = os.getenv("FLUX_MODEL_PATH", "models/FLUX.2-klein-4B")
OUTPAINT_LORA_REPO = "fal/flux-2-klein-4B-outpaint-lora"
DEFAULT_LORA_PATH = os.getenv(
    "FLUX_OUTPAINT_LORA_PATH", "models/flux-2-klein-4B-outpaint-lora"
)
DEFAULT_LORA_WEIGHT_NAME = "flux-outpaint-lora.safetensors"
OUTPAINT_LORA_ADAPTER_NAME = "green_outpaint"
DEFAULT_PROMPT = (
    "Replace the original object in the masked region with the main subject from the reference image. "
    "The reference subject must take the place of the original target object, preserving the reference "
    "subject's identity, shape, material, colors and distinctive details. "
    "Match the replaced object's apparent size, occupied area, placement, pose, body orientation and action "
    "as closely as possible. Preserve the target scene outside the replacement and keep the surrounding "
    "background structure, colors, brightness, texture, perspective and lighting unchanged. "
    "Integrate the replacement naturally with correct scale, perspective, lighting, contact and shadows. "
    "Do not retain the original object and do not add unrelated objects."
)


@dataclass(frozen=True)
class PipelineKey:
    model_path: str
    cpu_offload: bool
    local_files_only: bool
    lora_enabled: bool
    lora_path: str
    lora_weight_name: str
    lora_adapter_name: str


_PIPE: Any | None = None
_PIPE_KEY: PipelineKey | None = None
_PIPE_LOCK = threading.Lock()


def build_prompt(user_prompt: str | None, use_outpaint_lora: bool = False) -> str:
    text = (user_prompt or "").strip()
    return DEFAULT_PROMPT if not text else f"{DEFAULT_PROMPT} Additional replacement instruction: {text}"


def run_layered_diffusion(
    pipe: Any,
    torch_module: Any,
    *,
    model_input: Image.Image,
    model_mask: Image.Image,
    reference_model: Image.Image,
    object_prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    callback_factory: Callable[[int, str], Callable | None] | None = None,
    before_pass: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Run a single reference-guided FLUX editing pass."""
    pass_count = 1
    pass_seconds: dict[str, float] = {}

    def invoke(index: int, label: str, prompt: str, image_reference=None):
        if before_pass is not None:
            before_pass(index, label)
        kwargs = {
            "prompt": prompt,
            "image": model_input,
            "mask_image": model_mask,
            "strength": float(strength),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "height": model_input.height,
            "width": model_input.width,
            # Use a fresh generator initialized with the requested seed.
            "generator": torch_module.Generator(device="cuda").manual_seed(int(seed)),
        }
        callback = callback_factory(index, label) if callback_factory is not None else None
        if callback is not None:
            kwargs["callback_on_step_end"] = callback
        if image_reference is not None:
            kwargs["image_reference"] = image_reference
        started = time.perf_counter()
        image = pipe(**kwargs).images[0]
        pass_seconds[label] = time.perf_counter() - started
        return image

    object_layer = invoke(1, "Reference-guided diffusion", object_prompt, image_reference=reference_model)
    return {
        # Kept for callers that display diagnostics. It now always means that
        # no generated background layer exists; the original Target is used.
        "background": None,
        "object": object_layer,
        "pass_count": pass_count,
        "pass_seconds": pass_seconds,
    }


def _set_lora_scale(
    pipe: Any,
    scale: float,
    adapter_name: str = OUTPAINT_LORA_ADAPTER_NAME,
) -> None:
    """Activate the named adapter without fusing it, so its scale stays adjustable."""
    scale = max(0.0, float(scale))
    try:
        pipe.set_adapters([adapter_name], adapter_weights=[scale])
    except TypeError:
        # Compatibility with Diffusers/PEFT variants that still call this argument `weights`.
        pipe.set_adapters([adapter_name], weights=[scale])


def _empty_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_pipeline() -> None:
    global _PIPE, _PIPE_KEY
    with _PIPE_LOCK:
        _PIPE = None
        _PIPE_KEY = None
    _empty_cuda_cache()


def load_pipeline(
    model_path: str = DEFAULT_MODEL_PATH,
    cpu_offload: bool = True,
    local_files_only: bool = True,
    lora_enabled: bool = True,
    lora_path: str = DEFAULT_LORA_PATH,
    lora_weight_name: str = DEFAULT_LORA_WEIGHT_NAME,
    lora_scale: float = 1.1,
    lora_adapter_name: str = OUTPAINT_LORA_ADAPTER_NAME,
):
    global _PIPE, _PIPE_KEY
    lora_path = lora_path.strip() if lora_enabled else ""
    lora_weight_name = lora_weight_name.strip() if lora_enabled else ""
    key = PipelineKey(
        model_path.strip(),
        bool(cpu_offload),
        bool(local_files_only),
        bool(lora_enabled),
        lora_path,
        lora_weight_name,
        lora_adapter_name.strip() or OUTPAINT_LORA_ADAPTER_NAME,
    )
    with _PIPE_LOCK:
        if _PIPE is not None and _PIPE_KEY == key:
            print(f"[FLUX] Using cached model: {key.model_path}", flush=True)
            if key.lora_enabled:
                _set_lora_scale(_PIPE, lora_scale, key.lora_adapter_name)
                print(f"[LoRA] Using cached adapter, scale={float(lora_scale):.3f}", flush=True)
            return _PIPE
        _PIPE = None
        _PIPE_KEY = None
        _empty_cuda_cache()
        try:
            import torch
            from diffusers import Flux2KleinInpaintPipeline
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Flux2KleinInpaintPipeline is unavailable. Install Diffusers from its main branch as described in the README."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2 Klein requires a CUDA GPU. PyTorch did not detect CUDA.")
        print(f"[FLUX] Loading model: {key.model_path}", flush=True)
        print(f"[FLUX] dtype=bfloat16, cpu_offload={key.cpu_offload}, local_only={key.local_files_only}", flush=True)
        load_started = time.perf_counter()
        pipe = Flux2KleinInpaintPipeline.from_pretrained(
            key.model_path, torch_dtype=torch.bfloat16, local_files_only=key.local_files_only
        )
        print(f"[FLUX] from_pretrained completed in {time.perf_counter() - load_started:.1f}s", flush=True)
        if key.lora_enabled:
            if not key.lora_path:
                raise ValueError("LoRA is enabled but its path or Hugging Face ID is empty.")
            if not hasattr(pipe, "load_lora_weights") or not hasattr(pipe, "set_adapters"):
                raise RuntimeError("This pipeline does not support FLUX.2 LoRA. Update Diffusers and PEFT.")
            lora_started = time.perf_counter()
            lora_kwargs = {
                "adapter_name": key.lora_adapter_name,
                "local_files_only": key.local_files_only,
            }
            if key.lora_weight_name:
                lora_kwargs["weight_name"] = key.lora_weight_name
            print(
                f"[LoRA] Loading {key.lora_path} / {key.lora_weight_name or 'auto-selected weights'}",
                flush=True,
            )
            pipe.load_lora_weights(key.lora_path, **lora_kwargs)
            _set_lora_scale(pipe, lora_scale, key.lora_adapter_name)
            print(
                f"[LoRA] Loaded, scale={float(lora_scale):.3f}, elapsed {time.perf_counter() - lora_started:.1f}s",
                flush=True,
            )
        if key.cpu_offload:
            print("[FLUX] Enabling model CPU offload...", flush=True)
            pipe.enable_model_cpu_offload()
        else:
            print("[FLUX] Moving model to CUDA...", flush=True)
            pipe.to("cuda")
        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
        _PIPE, _PIPE_KEY = pipe, key
        print(f"[FLUX] Model ready in {time.perf_counter() - load_started:.1f}s", flush=True)
        return pipe


def prepare_consistency_backgrounds(
    source_roi: Image.Image,
    generated_roi: Image.Image,
    original_subject_mask: Image.Image,
    generated_subject_mask: Image.Image,
    core_roi: Image.Image,
    *,
    original_cleanup_expand: int = 12,
    generated_fill_expand: int = 16,
    fill_radius: float = 48.0,
) -> dict[str, Image.Image]:
    """Build subject-free plates and a solid subject-removal repair region."""
    source_roi = rgb(source_roi)
    generated_roi = rgb(generated_roi).resize(source_roi.size, Image.Resampling.LANCZOS)
    original_mask = normalize_mask(original_subject_mask, source_roi.size).point(
        lambda value: 255 if value >= 128 else 0
    )
    generated_mask = normalize_mask(generated_subject_mask, source_roi.size).point(
        lambda value: 255 if value >= 128 else 0
    )
    core = normalize_mask(core_roi, source_roi.size).point(
        lambda value: 255 if value >= 128 else 0
    )
    # Old objects commonly leave a wider shadow, brim, outline or reflection
    # than SAM's semantic silhouette. Clean them with a deliberately larger
    # margin. The generated subject only needs enough padding to make a neutral
    # subject-free conditioning plate.
    original_hidden = expand_mask(
        original_mask, max(0, int(original_cleanup_expand))
    )
    generated_hidden = expand_mask(
        generated_mask, max(0, int(generated_fill_expand))
    )
    hidden_union = ImageChops.lighter(original_hidden, generated_hidden)
    # SAM can miss a few antialiased pixels on the old Target subject. Its
    # small dilation must therefore become part of the real edit region rather
    # than being clipped back to the user's original mask later.
    effective_edit_mask = ImageChops.lighter(core, original_hidden).point(
        lambda value: 255 if value >= 128 else 0
    )
    # This mask is used only for final pixel ownership after diffusion: outside
    # the exact new-subject contour comes from the generated background.
    background_mask = ImageChops.subtract(
        effective_edit_mask, generated_mask
    ).point(
        lambda value: 255 if value >= 128 else 0
    )
    # SAM silhouettes are conditioning masks only. If their union is also used
    # as the output ownership mask, any small color difference between the two
    # FLUX passes draws the complete old-object silhouette into the final image.
    # Reconstruct one continuous edit region instead, whose only visible seam
    # is the user edit/Target boundary handled later in app.py.
    model_background_mask = effective_edit_mask.copy()
    # The model also reconstructs the background under the new subject, but
    # those pixels are hidden when the exact binary first-pass subject is pasted
    # last. Everywhere else in the continuous edit region uses one background.
    visible_repair_mask = background_mask.copy()
    # Target pixels covered by the old subject are synthetic/unknown and must
    # never be used as color-reference samples.
    valid_target_background = ImageChops.invert(original_hidden).point(
        lambda value: 255 if value >= 128 else 0
    )
    if background_mask.getbbox() is None:
        raise ValueError("The generated subject covers the entire editing mask. No background remains for harmonization.")
    return {
        # Target reference removes its old subject. The generated conditioning
        # plate removes both masks because first-pass pixels in the old-only
        # region can still carry that subject's low-frequency silhouette.
        "source_plate": fill_mask_from_surroundings(source_roi, original_hidden, fill_radius),
        # The first FLUX pass can retain a low-frequency impression of the old
        # subject even when it no longer contains the original pixels. Feeding
        # those pixels into the consistency pass lets the model reproduce the
        # same silhouette. Neutralize both subject regions in its conditioning
        # image; this plate remains conditioning-only and is never composited.
        "generated_plate": fill_mask_from_surroundings(
            generated_roi, hidden_union, fill_radius
        ),
        "original_hidden": original_hidden,
        "generated_hidden": generated_hidden,
        "hidden_union": hidden_union,
        "generated_subject_mask": generated_mask,
        "effective_edit_mask": effective_edit_mask,
        "background_mask": background_mask,
        "model_background_mask": model_background_mask,
        "visible_repair_mask": visible_repair_mask,
        "valid_target_background": valid_target_background,
    }


def compose_consistency_roi(
    color_matched_background: Image.Image,
    generated_roi: Image.Image,
    background_mask: Image.Image,
) -> Image.Image:
    """Merge a corrected FLUX background without ever using fill plates."""
    generated_roi = rgb(generated_roi)
    corrected = rgb(color_matched_background).resize(
        generated_roi.size, Image.Resampling.LANCZOS
    )
    # Internal subject/background ownership stays strictly binary. The only
    # allowed compositing feather is applied later at the outer edit/Target
    # boundary in app.py.
    alpha = normalize_mask(background_mask, generated_roi.size).point(
        lambda value: 255 if value >= 128 else 0
    )
    return Image.composite(corrected, generated_roi, alpha)


def edit_background_consistency(
    source_roi: Image.Image,
    generated_roi: Image.Image,
    original_subject_mask: Image.Image,
    generated_subject_mask: Image.Image,
    core_roi: Image.Image,
    *,
    original_cleanup_expand: int = 12,
    generated_fill_expand: int = 16,
    fill_radius: float = 48.0,
    color_lock_strength: float = 0.9,
    color_lock_radius: float = 64.0,
    color_lock_max_delta: float = 72.0,
    background_transition_radius: float = 12.0,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Color-correct the first-pass background without a second model call."""

    def report(message: str, value: float) -> None:
        if progress_callback is not None:
            progress_callback(message, value)

    started = time.perf_counter()
    prepared = prepare_consistency_backgrounds(
        source_roi,
        generated_roi,
        original_subject_mask,
        generated_subject_mask,
        core_roi,
        original_cleanup_expand=original_cleanup_expand,
        generated_fill_expand=generated_fill_expand,
        fill_radius=fill_radius,
    )
    # No second model call. The first-pass FLUX ROI is the only generated image
    # used by this stage. Temporary subject-free plates below are estimation
    # inputs only and never become visible pixels.
    raw_roi = rgb(generated_roi).resize(source_roi.size, Image.Resampling.LANCZOS)
    # The user/core mask is a *generation instruction*, not final pixel
    # ownership.  Reusing it here pastes the complete rectangular FLUX ROI
    # over Target: skies show a faint rectangle and roof/railing lines no
    # longer meet because diffusion has redrawn them.  After generation the
    # only background pixels that actually need replacing are those occupied
    # by the old Target subject (plus its cleanup margin).  Keep every other
    # Target pixel byte-for-byte and transition only around that local repair.
    empty_core = Image.new("L", source_roi.size, 0)
    transition_alpha = final_background_alpha(
        empty_core,
        prepared["original_hidden"],
        background_transition_radius,
    )
    # Estimate/correct the complete ROI background before the one and only
    # final source/ROI merge. Limiting correction to the old-subject mask leaves
    # the rest of the FLUX rectangle at a different exposure, which the final
    # feather can only soften, not remove. Protect the new subject because its
    # original raw pixels are pasted back last by its binary SAM mask.
    color_application_alpha = ImageChops.subtract(
        Image.new("L", source_roi.size, 255),
        prepared["generated_subject_mask"],
    )
    report("Harmonizing the first-pass background without another FLUX pass...", 0.25)
    color_matched_roi, color_lock_info = match_reference_background_colors(
        source_roi,
        raw_roi,
        color_application_alpha,
        reference_valid_mask=prepared["valid_target_background"],
        generated_estimation_image=prepared["generated_plate"],
        radius=color_lock_radius,
        strength=color_lock_strength,
        max_delta=color_lock_max_delta,
    )
    # There is no model-output/first-pass internal seam to harmonize. Final
    # outer edit/Target boundary matching is performed once in app.py.
    seam_matched_roi = color_matched_roi
    repair_seam_info = {
        "applied": False,
        "sample_strategy": "skipped_no_second_diffusion",
    }
    # color_matched_roi is raw_roi plus a correction that already fades through
    # the old-subject repair alpha.  The complete ROI is returned only as a
    # carrier image; app.py pastes from it exclusively through transition_alpha.
    consistent_roi = seam_matched_roi
    report("Background harmonized. Preparing to composite the complete SAM subject...", 0.95)
    return {
        **prepared,
        "raw_consistency_roi": raw_roi,
        "color_matched_roi": color_matched_roi,
        "background_layer_roi": seam_matched_roi,
        "color_lock_info": color_lock_info,
        "repair_seam_info": repair_seam_info,
        "consistent_background_roi": consistent_roi,
        "transition_alpha": transition_alpha,
        "color_application_alpha": color_application_alpha,
        # Retain diagnostic keys expected by older callers. They now describe
        # the CPU-only estimation inputs, not data sent to a diffusion model.
        "model_input": prepared["generated_plate"],
        "model_mask": prepared["model_background_mask"],
        "reference_model": prepared["source_plate"],
        "seed": None,
        "seconds": time.perf_counter() - started,
        "prompt": "",
        "model_size": source_roi.size,
        "second_flux_enabled": False,
        "consistency_lora_enabled": False,
        "consistency_lora_scale": 0.0,
    }


def edit_with_reference(
    source: Image.Image,
    mask: Image.Image,
    reference: Image.Image,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    prompt: str = "",
    seed: int = -1,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    strength: float = 1.0,
    max_roi_side: int = 1024,
    roi_context_ratio: float = 0.45,
    bbox_expand_left: int = 0,
    bbox_expand_right: int = 0,
    bbox_expand_top: int = 0,
    bbox_expand_bottom: int = 0,
    use_roi: bool = True,
    model_mask_expand: int = 8,
    model_mask_blur: float = 8.0,
    blend_expand: int = 48,
    direct_edge_feather: float = 32.0,
    direct_seam_color_match: bool = True,
    direct_seam_color_strength: float = 0.9,
    cpu_offload: bool = True,
    local_files_only: bool = True,
    lora_enabled: bool = True,
    lora_path: str = DEFAULT_LORA_PATH,
    lora_weight_name: str = DEFAULT_LORA_WEIGHT_NAME,
    lora_scale: float = 1.1,
    green_screen_input: bool = True,
    direct_overlay_enabled: bool = True,
    halo_fix_enabled: bool = True,
    halo_luminance_only: bool = False,
    halo_ring_width: int = 32,
    halo_strength: float = 1.0,
    halo_fade_radius: float = 192.0,
    halo_interior_strength: float = 0.35,
    green_despill_enabled: bool = True,
    green_despill_width: int = 24,
    texture_blend_enabled: bool = True,
    texture_blend_levels: int = 4,
    background_similarity_threshold: float = 32.0,
    background_restore_strength: float = 0.85,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    def report(message: str, value: float) -> None:
        if progress_callback is not None:
            progress_callback(message, value)

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    task_started = time.perf_counter()
    source_input_size = source.size
    reference_input_size = reference.size
    preprocess_started = time.perf_counter()
    report("Preparing target, mask, and reference...", 0.05)
    print("[Task] Preparing target, mask, reference, and ROI", flush=True)
    source = rgb(source)
    core_mask = normalize_mask(mask, source.size)
    if core_mask.getbbox() is None:
        raise ValueError("The mask is empty. Draw an editing region on the target image first.")
    reference = rgb(reference)
    location_box = mask_bbox_with_context(
        core_mask,
        context_ratio=0.0,
        min_context=0,
        expand_left=bbox_expand_left,
        expand_right=bbox_expand_right,
        expand_top=bbox_expand_top,
        expand_bottom=bbox_expand_bottom,
    )
    roi_box = (
        mask_bbox_with_context(
            core_mask,
            roi_context_ratio,
            expand_left=bbox_expand_left,
            expand_right=bbox_expand_right,
            expand_top=bbox_expand_top,
            expand_bottom=bbox_expand_bottom,
        )
        if use_roi
        else (0, 0, source.width, source.height)
    )
    source_roi = source.crop(roi_box)
    core_roi = core_mask.crop(roi_box)
    green_fill_roi = expand_mask(core_roi, model_mask_expand)
    # Keep the pure-green plate safely inside the region FLUX is allowed to
    # repaint. Otherwise the blurred mask can leave a partially preserved
    # green contour at exactly the final compositing seam.
    green_guard = (
        max(2, int(round(float(model_mask_blur))))
        if lora_enabled and green_screen_input
        else 0
    )
    model_mask_roi = feather_mask(
        expand_mask(green_fill_roi, green_guard), model_mask_blur
    )
    source_model, model_mask = resize_pair_for_model(source_roi, model_mask_roi, max_roi_side, 16)
    green_fill_model = normalize_mask(green_fill_roi, source_roi.size).resize(
        source_model.size, Image.Resampling.NEAREST
    )
    model_input = (
        paint_mask_green(source_model, green_fill_model)
        if lora_enabled and green_screen_input
        else source_model.copy()
    )
    reference_model = limit_reference(reference, max_roi_side, 16)
    actual_seed = random.SystemRandom().randint(0, 2**31 - 1) if int(seed) < 0 else int(seed)
    preprocess_seconds = time.perf_counter() - preprocess_started

    report("Loading FLUX.2 Klein and Outpaint LoRA..." if lora_enabled else "Loading FLUX.2 Klein...", 0.15)
    pipeline_started = time.perf_counter()
    pipe = load_pipeline(
        model_path,
        cpu_offload=cpu_offload,
        local_files_only=local_files_only,
        lora_enabled=lora_enabled,
        lora_path=lora_path,
        lora_weight_name=lora_weight_name,
        lora_scale=lora_scale,
    )
    pipeline_prepare_seconds = time.perf_counter() - pipeline_started

    pass_count = 1
    report("FLUX is ready. Starting generation...", 0.25)
    print(
        f"[Task] seed={actual_seed}, ROI={bbox_text(roi_box)}, "
        f"model_size={source_model.width}x{source_model.height}, steps={int(num_inference_steps)}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    diffusion_started = time.perf_counter()

    def callback_factory(pass_index: int, label: str):
        pass_started = time.perf_counter()

        def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
            completed = int(step_index) + 1
            total = max(1, int(num_inference_steps))
            elapsed = time.perf_counter() - pass_started
            seconds_per_step = elapsed / completed
            eta = seconds_per_step * max(0, total - completed)
            memory_text = ""
            if torch.cuda.is_available():
                gib = 1024**3
                allocated = torch.cuda.memory_allocated() / gib
                reserved = torch.cuda.memory_reserved() / gib
                peak = torch.cuda.max_memory_reserved() / gib
                memory_text = f" | GPU memory alloc={allocated:.1f}G reserved={reserved:.1f}G peak={peak:.1f}G"
            message = (
                f"{label} {completed}/{total} | {seconds_per_step:.1f}s/step | "
                f"ETA: {eta:.0f}s{memory_text}"
            )
            fraction = ((pass_index - 1) + completed / total) / pass_count
            report(message, 0.25 + 0.60 * fraction)
            print(f"[FLUX] {message}", flush=True)
            return callback_kwargs
        return on_step_end

    def before_pass(index: int, label: str) -> None:
        report("Generating the selected region with the reference...", 0.27)

    layers = run_layered_diffusion(
        pipe,
        torch,
        model_input=model_input,
        model_mask=model_mask,
        reference_model=reference_model,
        object_prompt=build_prompt(prompt, use_outpaint_lora=lora_enabled),
        seed=actual_seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        callback_factory=callback_factory,
        before_pass=before_pass,
    )
    raw_model = layers["object"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    diffusion_seconds = time.perf_counter() - diffusion_started
    print(f"[FLUX] Generation completed in {diffusion_seconds:.1f}s", flush=True)
    report("Restoring original resolution and compositing...", 0.87)
    postprocess_started = time.perf_counter()
    raw_roi = rgb(raw_model).resize(source_roi.size, Image.Resampling.LANCZOS)
    corrected_roi, halo_info = harmonize_roi_boundary_colors(
        source_roi,
        raw_roi,
        core_roi,
        enabled=halo_fix_enabled,
        luminance_only=halo_luminance_only,
        ring_width=halo_ring_width,
        strength=halo_strength,
        fade_radius=halo_fade_radius,
        interior_strength=halo_interior_strength,
    )
    prepared_roi, despill_info = despill_green_boundary(
        source_roi,
        corrected_roi,
        core_roi,
        enabled=green_despill_enabled,
        width=green_despill_width,
    )
    background_raw_roi = source_roi.copy()
    background_prepared_roi = source_roi.copy()
    background_halo_info: dict[str, Any] = {"applied": False}
    # V5.1 never asks FLUX to erase the old object. The only background/base
    # image is the untouched Target, retained here for diagnostic compatibility.
    full_background = source.copy()
    background_roi = full_background.crop(roi_box)

    # Processed composite retained for users who prefer boundary harmonization.
    fallback_final = composite_roi(
        source,
        prepared_roi,
        core_mask,
        roi_box,
        blend_expand,
        texture_blend_enabled=texture_blend_enabled,
        texture_blend_levels=texture_blend_levels,
        background_similarity_threshold=background_similarity_threshold,
        background_restore_strength=background_restore_strength,
    )
    # The clarity-first path uses the unmodified FLUX pixels and a binary user
    # mask. No halo correction, despill, multiband blending or soft edge is
    # allowed to alter the generated subject.
    direct_mask = core_mask.point(lambda value: 255 if value >= 128 else 0)
    direct_final = composite_roi_with_outward_feather(
        source,
        raw_roi,
        direct_mask,
        roi_box,
        expand_pixels=max(0, int(blend_expand)),
        hard_protect_pixels=min(8, max(0, int(blend_expand))),
        feather_pixels=max(0.0, float(direct_edge_feather)),
        seam_color_match=bool(direct_seam_color_match),
        seam_color_strength=float(direct_seam_color_strength),
        green_cleanup=bool(green_despill_enabled),
        green_cleanup_width=max(2, int(green_despill_width)),
    )
    final = direct_final if direct_overlay_enabled else fallback_final
    halo_info["despill"] = despill_info
    halo_info["texture_blend_enabled"] = bool(texture_blend_enabled)
    halo_info["texture_blend_levels"] = int(texture_blend_levels)
    halo_info["background_similarity_threshold"] = float(background_similarity_threshold)
    halo_info["background_restore_strength"] = float(background_restore_strength)
    halo_info["background"] = background_halo_info
    postprocess_seconds = time.perf_counter() - postprocess_started
    total_seconds = time.perf_counter() - task_started
    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0
    if torch.cuda.is_available():
        gib = 1024**3
        peak_allocated_gib = torch.cuda.max_memory_allocated() / gib
        peak_reserved_gib = torch.cuda.max_memory_reserved() / gib
    output_scale_x = final.width / max(1, source_input_size[0])
    output_scale_y = final.height / max(1, source_input_size[1])
    roi_scale_x = source_model.width / max(1, source_roi.width)
    roi_scale_y = source_model.height / max(1, source_roi.height)
    final_message = (
        f"Complete | Total time: {total_seconds:.1f}s | Output: {final.width}x{final.height} "
        f"({output_scale_x:.3f}x, {output_scale_y:.3f}x) | Peak GPU memory: {peak_reserved_gib:.2f} GiB"
    )
    report(final_message, 1.0)
    print(
        f"[Metrics] Target input={source_input_size[0]}x{source_input_size[1]} | "
        f"Reference={reference_input_size[0]}x{reference_input_size[1]} | "
        f"Original ROI={source_roi.width}x{source_roi.height} | "
        f"FLUX input={source_model.width}x{source_model.height} | "
        f"Final output={final.width}x{final.height} | "
        f"Output scale={output_scale_x:.3f}x/{output_scale_y:.3f}x",
        flush=True,
    )
    print(
        f"[Metrics] Preprocessing={preprocess_seconds:.2f}s | Model preparation={pipeline_prepare_seconds:.2f}s | "
        f"Generation={diffusion_seconds:.2f}s | Postprocessing={postprocess_seconds:.2f}s | "
        f"FLUX total time={total_seconds:.2f}s",
        flush=True,
    )
    print(
        f"[Metrics] Peak CUDA allocated={peak_allocated_gib:.2f} GiB | "
        f"reserved={peak_reserved_gib:.2f} GiB",
        flush=True,
    )
    if halo_info["applied"]:
        bias_text = "/".join(f"{value:+.1f}" for value in halo_info["bias"])
        print(
            f"[Halo correction] strategy={halo_info['sample_strategy']} | samples={halo_info['sample_pixels']} | "
            f"median_bias={bias_text} | field_max={halo_info['field_abs_max']:.1f} | "
            f"Boundary error: {halo_info['edge_error_before']:.2f} -> {halo_info['edge_error_after']:.2f}",
            flush=True,
        )
    if despill_info["applied"]:
        print(
            f"[Green spill] pixels={despill_info['pixels']} | "
            f"mean_excess={despill_info['mean_excess']:.1f}",
            flush=True,
        )
    print(
        f"[Texture blending] enabled={bool(texture_blend_enabled)} | levels={int(texture_blend_levels)}",
        flush=True,
    )
    return {
        "final": final,
        "direct_final": direct_final,
        "fallback_final": fallback_final,
        "raw_roi": raw_roi,
        "object_roi": prepared_roi,
        "corrected_roi": prepared_roi,
        "full_background": full_background,
        "background_roi": background_roi,
        "background_raw_roi": background_raw_roi,
        "background_prepared_roi": background_prepared_roi,
        "halo_info": halo_info,
        "source_roi": source_roi,
        "source": source,
        "core_roi": core_roi,
        "core_mask": core_mask,
        "model_mask": model_mask,
        "reference": reference_model,
        "model_input": model_input,
        "prompt": build_prompt(prompt, use_outpaint_lora=lora_enabled),
        "flux_pass_count": pass_count,
        "lora_enabled": bool(lora_enabled),
        "lora_path": lora_path.strip() if lora_enabled else "",
        "lora_weight_name": lora_weight_name.strip() if lora_enabled else "",
        "lora_scale": float(lora_scale) if lora_enabled else 0.0,
        "green_screen_input": bool(lora_enabled and green_screen_input),
        "green_guard": green_guard,
        "direct_overlay_enabled": bool(direct_overlay_enabled),
        "blend_expand": max(0, int(blend_expand)),
        "direct_edge_feather": max(0.0, float(direct_edge_feather)),
        "direct_seam_color_match": bool(direct_seam_color_match),
        "direct_seam_color_strength": max(0.0, min(1.0, float(direct_seam_color_strength))),
        "green_despill_enabled": bool(green_despill_enabled),
        "green_despill_width": max(2, int(green_despill_width)),
        "seed": actual_seed,
        "roi_box": roi_box,
        "roi_box_text": bbox_text(roi_box),
        "location_box": location_box,
        "location_box_text": bbox_text(location_box),
        "bbox_expansion": {
            "left": max(0, int(bbox_expand_left)),
            "right": max(0, int(bbox_expand_right)),
            "top": max(0, int(bbox_expand_top)),
            "bottom": max(0, int(bbox_expand_bottom)),
        },
        "model_size": source_model.size,
        "metrics": {
            "source_input_size": source_input_size,
            "reference_input_size": reference_input_size,
            "reference_model_size": reference_model.size,
            "roi_source_size": source_roi.size,
            "model_input_size": source_model.size,
            "output_size": final.size,
            "output_scale_x": output_scale_x,
            "output_scale_y": output_scale_y,
            "roi_scale_x": roi_scale_x,
            "roi_scale_y": roi_scale_y,
            "preprocess_seconds": preprocess_seconds,
            "pipeline_prepare_seconds": pipeline_prepare_seconds,
            "diffusion_seconds": diffusion_seconds,
            "background_diffusion_seconds": 0.0,
            "object_diffusion_seconds": layers["pass_seconds"].get("Reference-guided diffusion", 0.0),
            "postprocess_seconds": postprocess_seconds,
            "flux_total_seconds": total_seconds,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
        },
    }
