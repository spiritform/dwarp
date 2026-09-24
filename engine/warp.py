"""DWARP render engine: video -> stylized video, frame by frame, with optical-flow feedback.

The loop, per frame i:
  1. flow between source frames i-1 and i (RAFT, both directions)
  2. warp the previous *stylized* frame forward along that flow
  3. forward/backward consistency check -> mask of pixels the warp can be trusted for;
     everywhere else (occlusions, new content, off-screen) falls back to the raw source
  4. img2img from that blend, steered by ControlNets computed on the raw source frame
  5. save, and feed the result into frame i+1

Written from the idea of WarpFusion (Alex Spirin / Sxela) — no code taken from it.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ------------------------------------------------------------------ job description
@dataclass
class ControlSpec:
    kind: str            # depth | softedge | canny | lineart
    path: str            # ControlNet weights file
    weight: float = 1.0
    start: float = 0.0
    end: float = 1.0


@dataclass
class RenderJob:
    video: str
    out_dir: str
    checkpoint: str
    family: str = "sd15"                 # sd15 | sdxl
    prompt: str = ""
    negative: str = ""
    width: int = 512
    height: int = 512
    frame_start: int = 0                 # in extracted-frame units
    frame_end: int = -1                  # inclusive; -1 = to the end
    nth: int = 1                         # use every nth source frame
    style: float = 0.75                  # img2img strength on the first frame
    style_next: float = -1               # strength on later frames; -1 = 0.65 * style. Lower keeps
                                         # more of the warped previous frame (less boiling)
    steps: int = 20
    cfg: float = 6.0
    seed: int = 0
    controlnets: list[ControlSpec] = field(default_factory=list)
    # flow feedback
    flow_blend: float = 1.0              # 1 = trust the warped stylized frame fully where consistent
    mask_dilate: int = 5                 # grow the "don't trust" regions a little (px)
    mask_blur: int = 3
    color_match: float = 0.5             # 0..1 pull colour stats toward frame 0 (fights drift)

    @staticmethod
    def from_dict(d: dict) -> "RenderJob":
        d = dict(d)
        d["controlnets"] = [ControlSpec(**c) for c in d.get("controlnets", [])]
        return RenderJob(**d)


# ------------------------------------------------------------------ frames
def extract_frames(job: RenderJob, dest: Path) -> list[Path]:
    """Source frames at render size, numbered from 0 in extracted-frame units."""
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.jpg"):
        old.unlink()
    nth = max(1, job.nth)
    start = job.frame_start * nth
    end = "" if job.frame_end < 0 else f"*lte(n\\,{job.frame_end * nth})"
    vf = (f"select=gte(n\\,{start}){end}*not(mod(n\\,{nth})),"
          f"scale={job.width}:{job.height}:flags=lanczos")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", job.video, "-vf", vf, "-vsync", "vfr",
                    "-q:v", "2", "-start_number", "0", str(dest / "%06d.jpg")], check=True)
    frames = sorted(dest.glob("*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg extracted no frames — check the frame range")
    return frames


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def to_tensor(img: np.ndarray, device) -> torch.Tensor:
    """HWC uint8 -> 1x3xHxW float 0..1"""
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)


def to_image(t: torch.Tensor) -> Image.Image:
    arr = t.clamp(0, 1)[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(arr)


# ------------------------------------------------------------------ optical flow
class Flow:
    """RAFT-large from torchvision. flow(a, b)[p] = where pixel p of a moved to in b."""

    def __init__(self, device):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.device = device
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(device)

    @torch.inference_mode()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2:]
        # RAFT wants dims divisible by 8 and inputs in [-1, 1]
        H, W = math.ceil(h / 8) * 8, math.ceil(w / 8) * 8
        pa = F.interpolate(a, (H, W), mode="bilinear", align_corners=False) * 2 - 1
        pb = F.interpolate(b, (H, W), mode="bilinear", align_corners=False) * 2 - 1
        flow = self.model(pa, pb, num_flow_updates=20)[-1]
        if (H, W) != (h, w):
            flow = F.interpolate(flow, (h, w), mode="bilinear", align_corners=False)
            flow[:, 0] *= w / W
            flow[:, 1] *= h / H
        return flow


def _grid(flow: torch.Tensor) -> torch.Tensor:
    """Sampling grid for grid_sample: identity + flow, in [-1, 1]."""
    _, _, h, w = flow.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
    x = (xs + flow[0, 0]) / (w - 1) * 2 - 1
    y = (ys + flow[0, 1]) / (h - 1) * 2 - 1
    return torch.stack((x, y), dim=-1).unsqueeze(0)


def warp(img: torch.Tensor, flow_cur_to_prev: torch.Tensor) -> torch.Tensor:
    """Backward warp: pull each current-frame pixel from where it was in the previous frame."""
    return F.grid_sample(img, _grid(flow_cur_to_prev), mode="bilinear", padding_mode="border", align_corners=True)


def consistency(flow_bwd: torch.Tensor, flow_fwd: torch.Tensor) -> torch.Tensor:
    """1 where the warp is trustworthy. flow_bwd: cur->prev, flow_fwd: prev->cur.

    A pixel is consistent if going cur->prev->cur lands back where it started
    (Sundaram et al. 2010 threshold), and its source position is on screen."""
    _, _, h, w = flow_bwd.shape
    fwd_at = F.grid_sample(flow_fwd, _grid(flow_bwd), mode="bilinear", padding_mode="border", align_corners=True)
    round_trip = flow_bwd + fwd_at
    err = (round_trip ** 2).sum(1)
    mag = (flow_bwd ** 2).sum(1) + (fwd_at ** 2).sum(1)
    ok = err <= 0.01 * mag + 0.5
    g = _grid(flow_bwd)[0]
    on_screen = (g[..., 0].abs() <= 1) & (g[..., 1].abs() <= 1)
    return (ok & on_screen).float().unsqueeze(1)


def soften_mask(mask: torch.Tensor, dilate: int, blur: int) -> torch.Tensor:
    if dilate > 0:  # grow the untrusted (0) regions: erode the trusted ones
        k = dilate * 2 + 1
        mask = 1 - F.max_pool2d(1 - mask, k, stride=1, padding=dilate)
    if blur > 0:
        k = blur * 2 + 1
        mask = F.avg_pool2d(mask, k, stride=1, padding=blur, count_include_pad=False)
    return mask


def match_color(img: torch.Tensor, ref: torch.Tensor, amount: float) -> torch.Tensor:
    """Per-channel mean/std transfer toward ref — cheap guard against feedback colour drift."""
    if amount <= 0:
        return img
    m, s = img.mean((2, 3), keepdim=True), img.std((2, 3), keepdim=True) + 1e-5
    rm, rs = ref.mean((2, 3), keepdim=True), ref.std((2, 3), keepdim=True) + 1e-5
    return torch.lerp(img, ((img - m) / s * rs + rm).clamp(0, 1), amount)


# ------------------------------------------------------------------ ControlNet hints
class Annotators:
    """Condition images from the raw source frame. Detectors load lazily and stay cached."""

    def __init__(self, device):
        self.device = device
        self._cache: dict = {}

    def _get(self, kind):
        if kind not in self._cache:
            import controlnet_aux as ca
            if kind == "depth":
                det = ca.MidasDetector.from_pretrained("lllyasviel/Annotators")
            elif kind == "softedge":
                det = ca.PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            elif kind == "lineart":
                det = ca.LineartDetector.from_pretrained("lllyasviel/Annotators")
            elif kind == "canny":
                det = ca.CannyDetector()
            else:
                raise ValueError(f"unknown control kind {kind}")
            if hasattr(det, "to"):
                det = det.to(self.device)
            self._cache[kind] = det
        return self._cache[kind]

    def __call__(self, kind: str, img: np.ndarray) -> Image.Image:
        h, w = img.shape[:2]
        det = self._get(kind)
        res = min(h, w)
        if kind == "canny":
            out = det(Image.fromarray(img), low_threshold=100, high_threshold=200,
                      detect_resolution=res, image_resolution=res)
        elif kind == "softedge":
            out = det(Image.fromarray(img), detect_resolution=res, image_resolution=res, safe=True)
        else:
            out = det(Image.fromarray(img), detect_resolution=res, image_resolution=res)
        return out.resize((w, h), Image.BICUBIC)


# ------------------------------------------------------------------ diffusion
def load_pipeline(job: RenderJob, device, dtype=torch.float16):
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler

    # disable_mmap: memory-mapping multi-GB checkpoints crashes the process on Windows with an
    # access violation (seen with 7 GB SDXL files on a busy HDD); a plain read into RAM doesn't.
    nets = [ControlNetModel.from_single_file(c.path, torch_dtype=dtype, disable_mmap=True) for c in job.controlnets]
    import diffusers
    # No ControlNets = plain img2img: lighter and faster, structure comes from the flow warp alone.
    name = {("sdxl", True): "StableDiffusionXLControlNetImg2ImgPipeline",
            ("sdxl", False): "StableDiffusionXLImg2ImgPipeline",
            ("sd15", True): "StableDiffusionControlNetImg2ImgPipeline",
            ("sd15", False): "StableDiffusionImg2ImgPipeline"}[(job.family, bool(nets))]
    Pipe = getattr(diffusers, name)
    kwargs = dict(torch_dtype=dtype, disable_mmap=True)
    if nets:
        kwargs["controlnet"] = nets if len(nets) != 1 else nets[0]
    if job.family != "sdxl":
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    pipe = Pipe.from_single_file(job.checkpoint, **kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)
    if job.family != "sdxl":
        pipe.to(device)
    else:
        # 12 GB budget: UNet + ControlNets + VAE stay resident (~7.8 GB fp16); the two text
        # encoders (1.6 GB) live on the CPU and visit the GPU once per job (see sdxl_embeds).
        # Tiled VAE keeps the fp32-upcast decode from spiking. Spilling into shared memory
        # (the Windows driver does that silently) made frames 3-4x slower.
        for part in (pipe.unet, getattr(pipe, "controlnet", None), pipe.vae):   # text encoders stay on the CPU
            if part is not None:
                part.to(device)
        pipe.vae.enable_tiling()
        torch.cuda.empty_cache()
    return pipe


def sdxl_embeds(pipe, job: RenderJob, device) -> dict:
    """Encode the prompt once per job, then send the text encoders back to the CPU."""
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    with torch.inference_mode():
        pe, npe, ppe, nppe = pipe.encode_prompt(
            prompt=job.prompt, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=job.cfg > 1, negative_prompt=job.negative or None)
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    return dict(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=ppe,
                negative_pooled_prompt_embeds=nppe)


# Models survive between jobs: the pipeline for the last (checkpoint, family, ControlNets),
# plus RAFT and the annotators, which never change.
_CACHE: dict = {}


def cached_pipeline(job: RenderJob, device, log):
    key = (job.checkpoint, job.family, tuple(c.path for c in job.controlnets))
    if _CACHE.get("pipe_key") != key:
        _CACHE.pop("pipe", None)
        torch.cuda.empty_cache()
        t = time.time()
        _CACHE["pipe"] = load_pipeline(job, device)
        _CACHE["pipe_key"] = key
        log(f"loaded {Path(job.checkpoint).name} + {len(job.controlnets)} ControlNet(s) in {time.time() - t:.1f}s")
    else:
        log(f"reusing loaded {Path(job.checkpoint).name}")
    return _CACHE["pipe"]


def free_models():
    _CACHE.clear()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ the loop
Progress = Callable[[str, int, int, str], None]      # stage, frame, total, message


def render(job: RenderJob, progress: Progress = lambda *a: None, cancelled: Callable[[], bool] = lambda: False,
           log: Callable[[str], None] = print) -> list[Path]:
    device = torch.device("cuda")
    out = Path(job.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    (out / "debug").mkdir(exist_ok=True)
    (out / "job.json").write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")

    t0 = time.time()
    progress("extracting", 0, 0, "extracting frames")
    src_paths = extract_frames(job, out / "src")
    total = len(src_paths)
    log(f"extracted {total} frames at {job.width}x{job.height} (every {job.nth}) in {time.time() - t0:.1f}s")

    progress("loading", 0, total, "loading models")
    pipe = cached_pipeline(job, device, log)
    if total > 1 and "flow" not in _CACHE:
        _CACHE["flow"] = Flow(device)
    flow = _CACHE.get("flow")
    ann = _CACHE.setdefault("ann", Annotators(device))

    embeds = sdxl_embeds(pipe, job, device) if job.family == "sdxl" else None
    written, prev_src, prev_out, first_out = [], None, None, None
    style_next = job.style_next if job.style_next >= 0 else round(job.style * 0.65, 3)
    log(f"style {job.style} on frame 0, {style_next} after; colour match {job.color_match}")
    for i, sp in enumerate(src_paths):
        if cancelled():
            log("cancelled")
            break
        tf = time.time()
        src_np = load_rgb(sp)
        src = to_tensor(src_np, device)

        if prev_out is None:
            init = src
        else:
            fb = flow(src, prev_src)            # current -> previous
            ff = flow(prev_src, src)            # previous -> current
            warped = warp(prev_out, fb)
            mask = soften_mask(consistency(fb, ff), job.mask_dilate, job.mask_blur)
            init = torch.lerp(src, warped, mask * job.flow_blend)
            if i % 10 == 1:                     # a few debug snapshots, not every frame
                to_image(mask.expand(-1, 3, -1, -1)).save(out / "debug" / f"mask_{i:06d}.png")
                to_image(init).save(out / "debug" / f"init_{i:06d}.png")

        hints = [ann(c.kind, src_np) for c in job.controlnets]
        g = torch.Generator(device="cpu").manual_seed(int(job.seed) + i)
        text = embeds or dict(prompt=job.prompt, negative_prompt=job.negative or None)
        kw = dict(**text, image=to_image(init),
                  strength=job.style if i == 0 else style_next, num_inference_steps=job.steps, guidance_scale=job.cfg, generator=g)
        if job.controlnets:
            single = len(job.controlnets) == 1
            kw.update(control_image=hints[0] if single else hints,
                      controlnet_conditioning_scale=job.controlnets[0].weight if single else [c.weight for c in job.controlnets],
                      control_guidance_start=job.controlnets[0].start if single else [c.start for c in job.controlnets],
                      control_guidance_end=job.controlnets[0].end if single else [c.end for c in job.controlnets])
        result = pipe(**kw).images[0]

        res = to_tensor(np.asarray(result), device)
        if first_out is None:
            first_out = res
        else:
            res = match_color(res, first_out, job.color_match)
        path = out / "frames" / f"{i:06d}.png"
        to_image(res).save(path)
        written.append(path)
        prev_src, prev_out = src, res
        # Hand this frame's scratch memory back every frame. Without it the caching allocator's
        # footprint crept up until the Windows driver spilled into system RAM (30 s -> 235 s/frame).
        del result, init, hints, kw
        torch.cuda.empty_cache()
        log(f"frame {i + 1}/{total}: {time.time() - tf:.1f}s")
        progress("rendering", i + 1, total, f"frame {i + 1}/{total}")

    return written


def assemble(out_dir: str, fps: float) -> Path:
    out = Path(out_dir)
    target = out / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:g}", "-i", str(out / "frames" / "%06d.png"),
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                   check=True)
    return target
