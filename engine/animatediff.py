"""Motion (AnimateDiff): the whole clip through an SD1.5 checkpoint + a motion module, in overlapping
16-frame windows. Picked as the third way frames connect (Warp · Boil · Motion), for a video or a picture
(Image mode: the still, animated for the clip's length). No optical flow: the motion module keeps frames consistent inside a window, and
windows crossfade where they overlap (4 frames). ControlNets, DepthDiff, prompt travel, Live and
colour match work as in the frame-by-frame mode.

Motion modules and motion LoRAs load from the original AnimateDiff / ComfyUI files.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from engine.warp import (_CACHE, Annotators, RenderJob, camera_move, diff_mask, encode_prompts, extract_frames, load_lora, load_rgb,
                         match_color, style_embeds, to_image, to_tensor, travel)

# AnimateDiff's trained context, and the frames neighbouring windows share. Each window paints its own take on the
# motion, slightly ahead of or behind the last: a 4-frame overlap made a steady zoom stall, then jump, every 12
# frames (measured on run 39). 8 frames with an eased crossfade spreads that over the handover (~40% more windows).
WINDOW, OVERLAP = 16, 8


def is_lightning(path: str) -> bool:
    return "lightning" in Path(path).name.lower()


def is_lcm(path: str) -> bool:
    """LCM-distilled checkpoints only work with the LCM sampler (few steps, low CFG)."""
    return "lcm" in Path(path).name.lower()


def is_animatelcm(path: str) -> bool:
    """AnimateLCM motion module: LCM sampler, few steps, low CFG, plus its spatial LoRA on the UNet."""
    return "lcm" in Path(path).name.lower()


def animatelcm_lora(motion: str) -> str | None:
    """AnimateLCM's spatial LoRA (AnimateLCM_sd15_t2v_lora), looked for in the models root's loras."""
    for root in (Path(motion).parents[1] / "loras", Path(__file__).resolve().parents[1] / "models"):
        hit = next((p for p in sorted(root.rglob("*.safetensors")) if re.search(r"animatelcm.*lora", p.name, re.I)), None)
        if hit:
            return str(hit)
    return None


def lightning_steps(path: str) -> int:
    m = re.search(r"(\d+)\s*step", Path(path).name.lower())
    return int(m.group(1)) if m else 8


def motion_lora_to_diffusers(sd: dict) -> dict:
    """Original AnimateDiff / ComfyUI motion-LoRA keys -> diffusers PEFT keys on the motion UNet:
    `...attention_blocks.0.processor.to_q_lora.down.weight` -> `...attn1.to_q.lora_A.weight`,
    `to_out.processor.0_lora` -> `to_out.0`, `ff.net.0.processor.proj_lora` -> `ff.net.0.proj`."""
    out = {}
    for k, v in sd.items():
        k = k.replace("temporal_transformer.", "")
        k = re.sub(r"attention_blocks\.(\d+)", lambda m: f"attn{int(m.group(1)) + 1}", k)
        k = re.sub(r"\.processor\.(\w+?)_lora\.", r".\1.", k)
        k = k.replace(".down.weight", ".lora_A.weight").replace(".up.weight", ".lora_B.weight")
        k = k.replace(".to_out.lora_", ".to_out.0.lora_")   # v2 camera LoRAs say to_out_lora: the Linear in to_out
        out["unet." + k] = v
    return out


def load_ad_pipeline(job: RenderJob, device, dtype=torch.float16):
    import safetensors.torch
    from diffusers import (AnimateDiffVideoToVideoControlNetPipeline, AnimateDiffVideoToVideoPipeline, ControlNetModel,
                           DPMSolverMultistepScheduler, EulerDiscreteScheduler, MotionAdapter, StableDiffusionPipeline)
    base = StableDiffusionPipeline.from_single_file(job.checkpoint, torch_dtype=dtype, safety_checker=None,
                                                    requires_safety_checker=False, disable_mmap=True)
    # AnimateLCM stores no positional-encoding buffer, which diffusers reads the layout from: it is v2's
    extra = dict(config="guoyww/animatediff-motion-adapter-v1-5-2") if is_animatelcm(job.motion) else {}
    adapter = MotionAdapter.from_single_file(job.motion, torch_dtype=dtype, **extra)
    paths = list(dict.fromkeys(c.path for c in job.controlnets))
    nets = [ControlNetModel.from_single_file(p, torch_dtype=dtype, disable_mmap=True) for p in paths]
    parts = dict(vae=base.vae, text_encoder=base.text_encoder, tokenizer=base.tokenizer, unet=base.unet,
                 motion_adapter=adapter, scheduler=base.scheduler)
    pipe = (AnimateDiffVideoToVideoControlNetPipeline(**parts, controlnet=nets if len(nets) > 1 else nets[0]) if nets
            else AnimateDiffVideoToVideoPipeline(**parts))          # ControlNet off: the motion module alone
    if is_lightning(job.motion):             # distilled motion: its own sampler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing",
                                                            beta_schedule="linear")
    elif is_lcm(job.checkpoint) or is_animatelcm(job.motion):   # LCM: DPM++ turns it to noise
        from diffusers import LCMScheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    else:                                     # AnimateDiff's linear betas, DPM++ 2M Karras
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True,
                                                                 beta_schedule="linear")
    adapters = []
    if is_animatelcm(job.motion) and (spatial := animatelcm_lora(job.motion)):
        pipe.load_lora_weights(spatial, adapter_name="lcm")
        adapters.append(("lcm", job.lcm_lora_weight))
    if job.motion_lora:
        if job.motion_lora.lower().endswith(".safetensors"):
            raw = safetensors.torch.load_file(job.motion_lora)
        else:                                 # the official v2 camera LoRAs ship as .ckpt
            raw = torch.load(job.motion_lora, map_location="cpu", weights_only=True)
            raw = raw.get("state_dict", raw)
        sd = motion_lora_to_diffusers(raw)
        pipe.load_lora_weights(sd, adapter_name="motion")
        adapters.append(("motion", job.motion_lora_weight))
    if job.lora:                              # the Style card's LoRA, alongside the motion ones
        load_lora(pipe, job.lora)
        adapters.append(("style", job.lora_weight))
    if adapters:
        pipe.set_adapters([a for a, _ in adapters], [w for _, w in adapters])
    pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def cached_ad_pipeline(job: RenderJob, device, log):
    # shares the frame-by-frame mode's slot, so switching modes frees the other pipeline
    key = ("ad", job.checkpoint, job.motion, job.motion_lora, job.motion_lora_weight, job.lcm_lora_weight,
           job.lora, job.lora_weight, tuple(dict.fromkeys(c.path for c in job.controlnets)))
    if _CACHE.get("pipe_key") != key:
        _CACHE.pop("pipe", None)
        torch.cuda.empty_cache()
        t = time.time()
        _CACHE["pipe"] = load_ad_pipeline(job, device)
        _CACHE["pipe_key"] = key
        lora = f" + {Path(job.motion_lora).stem} x{job.motion_lora_weight}" if job.motion_lora else ""
        lora += f" + LoRA {Path(job.lora).stem} x{job.lora_weight}" if job.lora else ""
        log(f"loaded {Path(job.checkpoint).name} + {Path(job.motion).name}{lora} in {time.time() - t:.0f}s")
    else:
        log("reusing loaded AnimateDiff pipeline")
    return _CACHE["pipe"]


class VideoDiff:
    """DepthDiff for a window of frames: pixels whose turn hasn't come are held at the source,
    re-noised to the current step (the frame-by-frame mode's DiffDiffusion, for 5-D video latents
    and any scheduler). `amount` is F x 1 x H x W."""

    def __init__(self, pipe, amount: torch.Tensor):
        self.pipe, self.amount_px, self.orig = pipe, amount, None

    def __enter__(self):
        sch, real = self.pipe.scheduler, self.pipe.scheduler.add_noise
        self.prev = sch.__dict__.get("add_noise")

        def catch(orig, noise, t):
            if self.orig is None:              # the start latents, B,F,C,H,W
                self.orig, self.noise = orig, noise
            return real(orig, noise, t)
        sch.add_noise = catch
        return self

    def __exit__(self, *exc):
        if self.prev is not None:
            self.pipe.scheduler.add_noise = self.prev
        else:
            self.pipe.scheduler.__dict__.pop("add_noise", None)

    def __call__(self, pipe, i, t, kw):
        k, total = i + 1, pipe.num_timesteps
        if self.orig is None or k >= total:
            return kw
        a = F.interpolate(self.amount_px.float(), self.orig.shape[-2:], mode="area")       # F,1,h,w
        held = (a.permute(1, 0, 2, 3)[None] < 1 - k / total).to(self.orig.device)          # 1,1,F,h,w
        if held.any():
            ts = pipe.scheduler.timesteps       # the full schedule; this run uses its last `total` steps
            nxt = ts[len(ts) - total + k].reshape(1)
            o, nz = self.orig[0], self.noise[0]                                              # F,C,h,w
            ref = pipe.scheduler.add_noise(o, nz, nxt.repeat(o.shape[0]))[None].permute(0, 2, 1, 3, 4)
            kw["latents"] = torch.where(held, ref.to(kw["latents"].dtype), kw["latents"])
        return kw


class FrameNoise:
    """Start noise tied to each frame of the clip, not to the window: frame 37 always gets the same noise, every
    frame different. Neighbouring windows agree on the frames they share (calm seams), but the clip keeps evolving.
    With one seed per window instead, every window of a still (Image mode) started from the same noise and the
    same frames, so it painted the same 16 frames again: a short loop, repeating."""

    def __init__(self, pipe, seed: int, start: int, cache: dict):
        self.pipe, self.seed, self.start, self.cache = pipe, int(seed), start, cache

    def frames(self, like: torch.Tensor) -> torch.Tensor:
        five = like.dim() == 5                  # B,F,C,H,W (start latents) or F,C,H,W (DepthDiff's re-noise)
        n, shape = (like.shape[1], like.shape[2:]) if five else (like.shape[0], like.shape[1:])
        out = []
        for k in range(n):
            f = self.start + k
            if f not in self.cache:
                g = torch.Generator(device="cpu").manual_seed(self.seed * 1000003 + f)
                self.cache[f] = torch.randn(tuple(shape), generator=g)
            out.append(self.cache[f])
        x = torch.stack(out)
        return (x[None].expand(like.shape[0], *x.shape) if five else x).to(like.device, like.dtype)

    def __enter__(self):
        sch, real = self.pipe.scheduler, self.pipe.scheduler.add_noise
        sch.add_noise = lambda orig, noise, t: real(orig, self.frames(noise), t)
        return self

    def __exit__(self, *exc):
        self.pipe.scheduler.__dict__.pop("add_noise", None)


def windows(total: int) -> list[tuple[int, int]]:
    if total <= WINDOW:
        return [(0, total)]
    starts = list(range(0, total - WINDOW + 1, WINDOW - OVERLAP))
    if starts[-1] + WINDOW < total:
        starts.append(total - WINDOW)
    return [(s, s + WINDOW) for s in starts]


def render_ad(job: RenderJob, progress, cancelled: Callable[[], bool], log, live, on_keys) -> list[Path]:
    device = torch.device("cuda")
    out = Path(job.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    for c in job.controlnets:
        (out / "control" / c.kind).mkdir(parents=True, exist_ok=True)
    if job.diff_source != "off":
        (out / "control" / "diff").mkdir(parents=True, exist_ok=True)
    (out / "job.json").write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")

    t0 = time.time()
    progress("extracting", 0, 0, "extracting frames")
    i2v = bool(job.init_image)                # Image mode: the picture over the clip's length; the motion module moves it
    if i2v:
        import shutil
        from dataclasses import replace
        pic = Image.open(job.init_image).convert("RGB").resize((job.width, job.height), Image.LANCZOS)
        n = max(1, job.t2v_frames)
        src_paths = [out / "src" / f"{i:06d}.jpg" for i in range(n)]
        moving = abs(job.cam_zoom - 1) > 1e-4 or job.cam_rotate or job.cam_x or job.cam_y
        if moving:                             # the Camera card's move, baked into the source: frame k is the picture
            still = to_tensor(np.asarray(pic), torch.device("cuda"))   # moved k frames' worth, straight from it (no
            for k, pth in enumerate(src_paths):                         # re-sampling blur building up)
                to_image(camera_move(still, replace(job, nth=k * max(1, job.nth))) if k else still).save(pth, quality=95)
            log(f"image -> video: {Path(job.init_image).name} for {n} frames, the camera moving it (zoom {job.cam_zoom}, "
                f"rotate {job.cam_rotate}°, pan {job.cam_x}/{job.cam_y} per frame); the motion module paints the motion"
                + (" — 3D turn / tilt isn't used with Motion" if job.cam_3d and (job.cam_yaw or job.cam_pitch) else ""))
        else:
            pic.save(src_paths[0], quality=95)
            for pth in src_paths[1:]:
                shutil.copyfile(src_paths[0], pth)
            log(f"image -> video: animating {Path(job.init_image).name} for {n} frames (camera still: the motion module and prompt move it)")
    else:
        src_paths = extract_frames(job, out / "src")
    if job.shape:
        log("note: Shape isn't used with Motion (AnimateDiff) — Warp or Boil take it")
    total = len(src_paths)
    log(f"extracted {total} frames at {job.width}x{job.height} (every {job.nth}) in {time.time() - t0:.1f}s")

    progress("loading", 0, total, "loading models")
    pipe = cached_ad_pipeline(job, device, log)
    ann = _CACHE.setdefault("ann", Annotators(device))

    # hints and DepthDiff masks for every frame up front (the viewer shows them per frame too)
    progress("hints", 0, total, "ControlNet hints")
    src_np = [load_rgb(p) for p in src_paths]
    hints = {c.kind: [] for c in job.controlnets}
    amounts = []
    for i, img in enumerate(src_np):
        for c in job.controlnets:
            h = ann(c.kind, img)
            h.convert("RGB").save(out / "control" / c.kind / f"{i:06d}.jpg", quality=90)
            hints[c.kind].append(h.convert("RGB"))
        if job.diff_source != "off":
            base = to_tensor(np.asarray((hints.get("depth") or [None] * total)[i] or ann("depth", img)), device) \
                if job.diff_source == "depth" else to_tensor(img, device)
            dm = diff_mask(base, job)
            to_image(dm.expand(-1, 3, -1, -1)).save(out / "control" / "diff" / f"{i:06d}.jpg", quality=90)
            amounts.append(dm)
    log(f"hints for {total} frames in {time.time() - t0:.0f}s")

    # Motion modules are trained on 16-frame batches: fewer (a Single Frame, a short Preview) comes out as
    # colour confetti. Pad to a full window by bouncing through the clip's own frames; only the real ones
    # are kept.
    if total < WINDOW:
        cycle = list(range(total)) + list(range(total - 2, 0, -1)) or [0]
        pad = [cycle[i % len(cycle)] for i in range(WINDOW)]
        src_np = [src_np[i] for i in pad]
        hints = {k: [v[i] for i in pad] for k, v in hints.items()}
        amounts = [amounts[i] for i in pad] if amounts else amounts
        log(f"{total} frame{'s' * (total > 1)} padded to a {WINDOW}-frame window (the motion module needs one)")

    lightning = is_lightning(job.motion)
    steps = lightning_steps(job.motion) if lightning else job.steps
    cfg = 1.0 if lightning else job.cfg
    from dataclasses import replace
    ip_embeds = style_embeds(pipe, replace(job, cfg=cfg), device, log)   # Style ref (None without one)
    wins = windows(len(src_np))
    if job.motion_lora and not re.search(r"v2|temporaldiff|lcm", Path(job.motion).name, re.I):
        log(f"note: motion LoRAs are trained for the v2 motion module (TemporalDiff and AnimateLCM take them too); on "
            f"{Path(job.motion).stem} {Path(job.motion_lora).stem} can break the image")
    sampler = ("Lightning (Euler)" if lightning else "LCM" if is_lcm(job.checkpoint) or is_animatelcm(job.motion)
               else "DPM++ 2M Karras")
    log(f"AnimateDiff: {Path(job.motion).stem}, {sampler}, {len(wins)} window{'s' * (len(wins) > 1)} of {WINDOW} "
        f"(overlap {OVERLAP}), denoise {job.style}, {steps} steps, cfg {cfg}; colour match {job.color_match}")

    keys = sorted([int(f), str(p)] for f, p in job.prompt_keys) or [[0, job.prompt]]
    blend = job.prompt_blend
    encoded: dict[str, dict] = {}

    def embeds_for(ks):
        new = list(dict.fromkeys(p for _, p in ks if p not in encoded))
        if new:
            encoded.update(zip(new, encode_prompts(pipe, job, device, new)))
        return [encoded[p] for _, p in ks]

    embeds = embeds_for(keys)
    edits = []
    on_keys(keys, blend)
    nets = len(dict.fromkeys(c.path for c in job.controlnets))
    kinds = [c.kind for c in job.controlnets]
    first = None
    frame_noise: dict[int, torch.Tensor] = {}  # each frame's start noise, shared by the windows that cover it
    done: dict[int, np.ndarray] = {}           # rendered frames waiting for the next window's crossfade
    written = []
    for w, (s, e) in enumerate(wins):
        if cancelled():
            log("cancelled")
            break
        upd = live()
        if upd:
            keys = sorted([int(f), str(p)] for f, p in upd.get("prompt_keys") or []) or keys
            blend = int(upd.get("prompt_blend", blend))
            if upd.get("now"):                  # Live: the next window morphs into it
                sf = s * max(1, job.nth) if i2v else (job.frame_start + s) * max(1, job.nth)
                keys = sorted([k for k in keys if k[0] != sf] + [[sf, str(upd["now"])]])
            embeds = embeds_for(keys)
            edits.append({"from_frame": s, "prompt_keys": keys, "prompt_blend": blend})
            (out / "job.json").write_text(json.dumps(asdict(job) | {"live_edits": edits}, indent=2), encoding="utf-8")
            log(f"live prompts from frame {s + 1}: {len(keys)} keyframe{'s' * (len(keys) > 1)} at source frames "
                f"{[f for f, _ in keys]}, blend {blend}")
            on_keys(keys, blend)
        tw = time.time()
        mid = (s + e - 1) // 2 * max(1, job.nth) if i2v else (job.frame_start + (s + e - 1) // 2) * max(1, job.nth)
        text, cond = travel(keys, embeds, blend, mid)
        if len(keys) > 1:
            log(f"window {w + 1}: {cond}")
        # the same ControlNet can serve two hints only as separate entries; the pipeline takes a list per net
        cn_kw = {}
        if not nets:
            pass
        elif nets > 1:
            cond_frames = [hints[k][s:e] for k in kinds]
            scale = [c.weight for c in job.controlnets]
            starts = [c.start for c in job.controlnets]
            ends = [c.end for c in job.controlnets]
        else:
            c = job.controlnets[0]
            cond_frames, scale, starts, ends = hints[c.kind][s:e], c.weight, c.start, c.end
        if nets:
            cn_kw = dict(conditioning_frames=cond_frames, controlnet_conditioning_scale=scale,
                         control_guidance_start=starts, control_guidance_end=ends)
        amount = torch.cat(amounts[s:e]) if amounts else None
        if amount is not None and amount.min() >= 1:
            amount = None
        vd = VideoDiff(pipe, amount) if amount is not None else None
        g = torch.Generator(device="cpu").manual_seed(int(job.seed) + s)   # step noise (LCM / ancestral): per window
        with torch.inference_mode(), FrameNoise(pipe, job.seed, s, frame_noise), (vd if vd else torch.no_grad()):
            res = pipe(video=[Image.fromarray(x) for x in src_np[s:e]], **cn_kw,
                       prompt_embeds=text["prompt_embeds"], negative_prompt_embeds=text["negative_prompt_embeds"],
                       strength=job.style, num_inference_steps=steps, guidance_scale=cfg,
                       generator=g, output_type="np",
                       **({"ip_adapter_image_embeds": ip_embeds} if ip_embeds is not None else {}),
                       callback_on_step_end=vd, callback_on_step_end_tensor_inputs=["latents"])
        frames = res.frames[0]                  # F,H,W,3 in 0..1
        del res
        torch.cuda.empty_cache()
        prev_end = max(done) + 1 if done else s   # frames s..prev_end-1 overlap the previous window
        for j, f in enumerate(frames):
            i = s + j
            if i in done:                       # crossfade from the previous window into this one, eased in and out
                a = (i - s + 1) / (prev_end - s + 1)
                a = a * a * (3 - 2 * a)
                f = done[i] * (1 - a) + f * a
            done[i] = f
        last = w == len(wins) - 1
        final_to = min(total, e if last else wins[w + 1][0])      # padding frames aren't written
        for i in sorted(k for k in done if k < final_to):
            res_t = to_tensor((np.clip(done.pop(i), 0, 1) * 255).astype(np.uint8), device)
            if first is None:
                first = res_t
            else:
                res_t = match_color(res_t, first, job.color_match)
            path = out / "frames" / f"{i:06d}.png"
            to_image(res_t).save(path)
            written.append(path)
        log(f"window {w + 1}/{len(wins)} (frames {s + 1}-{e}): {time.time() - tw:.1f}s")
        progress("rendering", len(written), total, f"frame {len(written)}/{total}")
    return written
