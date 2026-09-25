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

import contextlib
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
    diff: bool = False                   # differential diffusion: per-pixel strength from the trust
                                         # mask — trusted pixels get style_next, the rest full style
    # DepthDiff: per-pixel strength from the source frame's luma or depth (0 = keep, 1 = full style)
    diff_source: str = "off"             # off | luma | depth
    diff_invert: bool = True             # dark / far areas repaint most
    diff_black: float = 0.0              # levels, 0..255
    diff_white: float = 255.0
    diff_gamma: float = 1.0
    diff_brightness: float = 0.0
    diff_contrast: float = 1.0
    diff_blur: int = 4                   # px
    diff_amount: float = 1.0             # scales the mask: <1 caps the repaint, >1 pushes it harder
    # prompt travel: [[source_frame, prompt], ...]; from each keyframe the render morphs into its
    # prompt over prompt_blend source frames (0 = hard cut). Empty = job.prompt throughout.
    prompt_keys: list = field(default_factory=list)
    prompt_blend: int = 12

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


# ------------------------------------------------------------------ DepthDiff mask
def gaussian_blur(m: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return m
    sigma = max(radius / 2.0, 0.1)
    x = torch.arange(radius * 2 + 1, dtype=m.dtype, device=m.device) - radius
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    m = F.pad(m, (radius, radius, radius, radius), mode="reflect")
    m = F.conv2d(m, k.view(1, 1, 1, -1))
    return F.conv2d(m, k.view(1, 1, -1, 1))


def diff_mask(img: torch.Tensor, job: "RenderJob") -> torch.Tensor:
    """1x3xHxW 0..1 image -> 1x1xHxW per-pixel strength share, shaped like the DepthDiff node."""
    m = (0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]).unsqueeze(1)
    if job.diff_invert:
        m = 1 - m
    b, w = job.diff_black / 255, job.diff_white / 255
    m = ((m - b) / max(w - b, 1e-4)).clamp(0, 1)
    if job.diff_gamma != 1:
        m = m.pow(1 / max(job.diff_gamma, 1e-4))
    m = ((m - 0.5) * job.diff_contrast + 0.5 + job.diff_brightness).clamp(0, 1)
    return (gaussian_blur(m, int(job.diff_blur)) * job.diff_amount).clamp(0, 1)


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
# xinsir's ControlNet Union (ProMax) for SDXL: one network, many condition types, selected per
# image by index. The files ship without a config and diffusers can't infer one, so it lives here.
UNION_CONFIG = dict(
    act_fn="silu", addition_embed_type="text_time", addition_embed_type_num_heads=64,
    addition_time_embed_dim=256, attention_head_dim=[5, 10, 20], block_out_channels=[320, 640, 1280],
    conditioning_channels=3, conditioning_embedding_out_channels=[16, 32, 96, 256],
    controlnet_conditioning_channel_order="rgb", cross_attention_dim=2048,
    down_block_types=["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    downsample_padding=1, flip_sin_to_cos=True, freq_shift=0, global_pool_conditions=False,
    in_channels=4, layers_per_block=2, mid_block_scale_factor=1, norm_eps=1e-5, norm_num_groups=32,
    projection_class_embeddings_input_dim=2816, resnet_time_scale_shift="default",
    transformer_layers_per_block=[1, 2, 10], use_linear_projection=True, num_control_type=8)
UNION_MODE = {"depth": 1, "softedge": 2, "canny": 3, "lineart": 3}


# SSD-1B (Segmind's distilled SDXL; SDXL Flash Mini is built on it): same channels and ControlNet
# hookup points as SDXL, but fewer transformer layers and a mid block without attention.
# diffusers' single-file loader assumes the full SDXL layout, so the UNet is built from this.
SSD1B_UNET_CONFIG = dict(
    addition_embed_type="text_time", addition_time_embed_dim=256, attention_head_dim=[5, 10, 20],
    block_out_channels=[320, 640, 1280], cross_attention_dim=2048,
    down_block_types=["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    mid_block_type="UNetMidBlock2D", projection_class_embeddings_input_dim=2816, sample_size=128,
    transformer_layers_per_block=[[1], [2, 2], [4, 4]],
    reverse_transformer_layers_per_block=[[4, 4, 10], [2, 1, 1], 1], use_linear_projection=True)


def _header(path: str) -> bytes:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return f.read(n)


def is_union(path: str) -> bool:
    """A Union ControlNet has a control_add_embedding."""
    return b'"control_add_embedding.' in _header(path)


def is_ssd1b(path: str) -> bool:
    """Full SDXL has middle_block.0/1/2 (resnet, attention, resnet); SSD-1B only the first."""
    h = _header(path)
    return b'"model.diffusion_model.middle_block.0.' in h and b'"model.diffusion_model.middle_block.1.' not in h


def load_ssd1b_unet(path: str, dtype):
    from diffusers import UNet2DConditionModel
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
    from safetensors.torch import load_file
    sd = {k: v for k, v in load_file(path).items() if k.startswith("model.diffusion_model.")}
    with torch.device("meta"):
        unet = UNet2DConditionModel(**SSD1B_UNET_CONFIG)
    sd = convert_ldm_unet_checkpoint(sd, unet.config)
    unet.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True, assign=True)
    return unet.eval()


def load_union(path: str, dtype):
    from diffusers import ControlNetUnionModel
    from safetensors.torch import load_file
    with torch.device("meta"):
        net = ControlNetUnionModel(**UNION_CONFIG)
    net.load_state_dict({k: v.to(dtype) for k, v in load_file(path).items()}, strict=True, assign=True)
    return net.eval()


def load_pipeline(job: RenderJob, device, dtype=torch.float16):
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler

    paths = list(dict.fromkeys(c.path for c in job.controlnets))
    union = len(paths) == 1 and is_union(paths[0])
    # disable_mmap: memory-mapping multi-GB checkpoints crashes the process on Windows with an
    # access violation (seen with 7 GB SDXL files on a busy HDD); a plain read into RAM doesn't.
    nets = [load_union(paths[0], dtype)] if union else \
        [ControlNetModel.from_single_file(c.path, torch_dtype=dtype, disable_mmap=True) for c in job.controlnets]
    import diffusers
    # No ControlNets = plain img2img: lighter and faster, structure comes from the flow warp alone.
    name = "StableDiffusionXLControlNetUnionImg2ImgPipeline" if union else \
           {("sdxl", True): "StableDiffusionXLControlNetImg2ImgPipeline",
            ("sdxl", False): "StableDiffusionXLImg2ImgPipeline",
            ("sd15", True): "StableDiffusionControlNetImg2ImgPipeline",
            ("sd15", False): "StableDiffusionImg2ImgPipeline"}[(job.family, bool(nets))]
    Pipe = getattr(diffusers, name)
    kwargs = dict(torch_dtype=dtype, disable_mmap=True)
    if nets:
        kwargs["controlnet"] = nets if len(nets) != 1 else nets[0]
    if job.family != "sdxl":
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    elif is_ssd1b(job.checkpoint):
        kwargs["unet"] = load_ssd1b_unet(job.checkpoint, dtype)
    pipe = Pipe.from_single_file(job.checkpoint, **kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)
    if job.family != "sdxl":
        pipe.to(device)
    else:
        # 12 GB budget: UNet + ControlNets + VAE stay resident (~7.8 GB fp16); the two text
        # encoders (1.6 GB) live on the CPU and visit the GPU once per job (see encode_prompts).
        # Tiled VAE keeps the fp32-upcast decode from spiking. Spilling into shared memory
        # (the Windows driver does that silently) made frames 3-4x slower.
        for part in (pipe.unet, getattr(pipe, "controlnet", None), pipe.vae):   # text encoders stay on the CPU
            if part is not None:
                part.to(device)
        pipe.vae.enable_tiling()
        torch.cuda.empty_cache()
    return pipe


def encode_prompts(pipe, job: RenderJob, device, prompts: list[str]) -> list[dict]:
    """Encode each prompt once per job. SDXL's text encoders visit the GPU just for this."""
    sdxl = job.family == "sdxl"
    if sdxl:
        pipe.text_encoder.to(device)
        pipe.text_encoder_2.to(device)
    out = []
    with torch.inference_mode():
        for p in prompts:
            if sdxl:
                pe, npe, ppe, nppe = pipe.encode_prompt(
                    prompt=p, device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=job.cfg > 1, negative_prompt=job.negative or None)
                out.append(dict(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=ppe,
                                negative_pooled_prompt_embeds=nppe))
            else:
                pe, npe = pipe.encode_prompt(p, device, 1, job.cfg > 1, job.negative or None)
                out.append(dict(prompt_embeds=pe, negative_prompt_embeds=npe))
    if sdxl:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
    return out


def travel(keys: list, embeds: list[dict], blend: int, f: int) -> tuple[dict, str]:
    """Conditioning at source frame f: the last keyframe at or before f, morphing in from the
    previous one over `blend` frames. Returns the embeds and a short description for the log."""
    j = max([k for k, (kf, _) in enumerate(keys) if kf <= f] or [0])
    if j == 0 or blend <= 0 or f >= keys[j][0] + blend:
        return embeds[j], f"prompt {j + 1}"
    w = (f - keys[j][0]) / blend
    a, b = embeds[j - 1], embeds[j]
    return ({k: None if a[k] is None else torch.lerp(a[k], b[k], w) for k in a},
            f"prompt {j} -> {j + 1} {w:.0%}")


# Models survive between jobs: the pipeline for the last (checkpoint, family, ControlNets),
# plus RAFT and the annotators, which never change.
_CACHE: dict = {}


def cached_pipeline(job: RenderJob, device, log):
    key = (job.checkpoint, job.family, tuple(dict.fromkeys(c.path for c in job.controlnets)))   # union: depth / depth+edge share one net
    if _CACHE.get("pipe_key") != key:
        _CACHE.pop("pipe", None)
        torch.cuda.empty_cache()
        t = time.time()
        _CACHE["pipe"] = load_pipeline(job, device)
        _CACHE["pipe_key"] = key
        log(f"loaded {Path(job.checkpoint).name} + {len(key[2])} ControlNet(s) in {time.time() - t:.1f}s")
    else:
        log(f"reusing loaded {Path(job.checkpoint).name}")
    return _CACHE["pipe"]


class DiffDiffusion:
    """Per-pixel img2img strength as a step callback (Differential Diffusion, Levin & Fried 2023).

    `amount` (1x1xHxW, 0..1) is the share of the run's steps each pixel may change in: 1 = all of
    them, 0.5 = only the second half, 0 = none. Until its turn a pixel is held at the init image,
    re-noised to the current step, so it joins the denoise at exactly the right noise level.
    Held pixels also get the init as the model's clean-image prediction (as ComfyUI's masked
    sampling does), so a multistep solver's history stays on the init's path: without that, a
    pixel's first free step reused a stale prediction and trusted areas flickered more.
    The init latents and noise are the pipeline's own (caught as it noises the start latents).
    Use as a context manager around the pipeline call."""

    def __init__(self, pipe, amount: torch.Tensor):
        self.pipe, self.amount_px = pipe, amount
        self.orig = self.noise = self.amount = self.held = None

    def _prepare_latents(self, *a, **k):
        sch = self.pipe.scheduler
        real = sch.add_noise

        def catch(orig, noise, t):
            self.orig, self.noise = orig, noise
            return real(orig, noise, t)
        sch.add_noise = catch
        try:
            return self._orig_prepare(*a, **k)
        finally:
            del sch.add_noise

    def _convert(self, *a, **k):
        x0 = self._orig_convert(*a, **k)
        if self.held is None:                   # first step: everything below amount 1 is held
            self.amount = F.interpolate(self.amount_px.float(), self.orig.shape[-2:], mode="area").to(self.orig)
            self.held = self.was_held = self.amount < 1
        hist = getattr(self.pipe.scheduler, "model_outputs", None)
        released = self.was_held & ~self.held
        if hist and hist[-1] is not None and released.any():
            # A plain run's first step is first order; give just-released pixels the same by making
            # their "previous prediction" equal this one (the multistep correction term vanishes).
            hist[-1] = torch.where(released, x0, hist[-1])
        self.was_held = self.held
        return torch.where(self.held, self.orig.to(x0.dtype), x0)

    def __enter__(self):
        sch = self.pipe.scheduler
        self._orig_prepare, self._orig_convert = self.pipe.prepare_latents, sch.convert_model_output
        self.pipe.prepare_latents, sch.convert_model_output = self._prepare_latents, self._convert
        return self

    def __exit__(self, *exc):
        del self.pipe.prepare_latents           # back to the class methods
        del self.pipe.scheduler.convert_model_output

    def __call__(self, pipe, i, t, kw):
        # After step i the scheduler sits at step i+1; add_noise uses that level (sigma 0 at the end).
        n, k = pipe.num_timesteps, i + 1
        self.held = self.amount < 1 - k / n
        if self.held.any():
            ref = pipe.scheduler.add_noise(self.orig, self.noise, t.reshape(1))
            kw["latents"] = torch.where(self.held, ref, kw["latents"])
        return kw


def depth_map(img: np.ndarray) -> Image.Image:
    """The depth hint for one RGB frame, with the cached annotator (DepthDiff preview)."""
    return _CACHE.setdefault("ann", Annotators(torch.device("cuda")))("depth", img)


def free_models():
    _CACHE.clear()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ the loop
Progress = Callable[[str, int, int, str], None]      # stage, frame, total, message


def render(job: RenderJob, progress: Progress = lambda *a: None, cancelled: Callable[[], bool] = lambda: False,
           log: Callable[[str], None] = print, live: Callable[[], dict | None] = lambda: None,
           on_keys: Callable[[list, int], None] = lambda keys, blend: None) -> list[Path]:
    """`live` is polled before every frame; it returns None, or edits sent while the job runs, for
    the frames still to come: new prompt travel ("prompt_keys": [[frame, prompt], ...], "prompt_blend")
    and/or "now": a prompt that becomes a keyframe at the frame about to render (Live mode).
    `on_keys` hears the keyframes in use at the start and after every edit."""
    device = torch.device("cuda")
    out = Path(job.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    (out / "debug").mkdir(exist_ok=True)
    for c in job.controlnets:                   # the hints each frame was steered by, for the viewer
        (out / "control" / c.kind).mkdir(parents=True, exist_ok=True)
    if job.diff_source != "off":
        (out / "control" / "diff").mkdir(parents=True, exist_ok=True)
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

    keys = sorted([int(f), str(p)] for f, p in job.prompt_keys) or [[0, job.prompt]]
    blend = job.prompt_blend
    # Encoded prompts by text, so a live edit only encodes what's new. SD1.5 without travel keeps
    # passing the text itself until a live edit arrives; everything else goes through embeds.
    encoded: dict[str, dict] = {}

    def embeds_for(ks):
        new = list(dict.fromkeys(p for _, p in ks if p not in encoded))
        if new:
            encoded.update(zip(new, encode_prompts(pipe, job, device, new)))
        return [encoded[p] for _, p in ks]

    embeds = embeds_for(keys) if job.family == "sdxl" or len(keys) > 1 else None
    if len(keys) > 1:
        log(f"prompt travel: {len(keys)} keyframes at source frames {[f for f, _ in keys]}, blend {blend}")
    edits = []                                  # live prompt edits, kept in job.json next to the start state
    on_keys(keys, blend)
    last_cond = None
    union = type(getattr(pipe, "controlnet", None)).__name__ == "ControlNetUnionModel"
    written, prev_src, prev_out, first_out = [], None, None, None
    style_next = job.style_next if job.style_next >= 0 else round(job.style * 0.65, 3)
    log(f"style {job.style} on frame 0, {style_next} after{' (untrusted pixels: full style)' if job.diff else ''}; "
        f"colour match {job.color_match}")
    for i, sp in enumerate(src_paths):
        if cancelled():
            log("cancelled")
            break
        upd = live()
        if upd:
            keys = sorted([int(f), str(p)] for f, p in upd.get("prompt_keys") or []) or keys
            blend = int(upd.get("prompt_blend", blend))
            if upd.get("now"):                  # Live: from this frame on, morph into the new prompt
                sf = (job.frame_start + i) * max(1, job.nth)
                keys = sorted([k for k in keys if k[0] != sf] + [[sf, str(upd["now"])]])
            te = time.time()
            embeds = embeds_for(keys)
            edits.append({"from_frame": i, "prompt_keys": keys, "prompt_blend": blend})
            (out / "job.json").write_text(json.dumps(asdict(job) | {"live_edits": edits}, indent=2), encoding="utf-8")
            log(f"live prompts from frame {i + 1}: {len(keys)} keyframe{'s' * (len(keys) > 1)} at source frames "
                f"{[f for f, _ in keys]}, blend {blend} ({time.time() - te:.1f}s)")
            on_keys(keys, blend)
            last_cond = None
        tf = time.time()
        src_np = load_rgb(sp)
        src = to_tensor(src_np, device)

        amount = None                           # per-pixel strength, as a share of `strength`
        trust = False                           # job.diff: untrusted pixels get full style
        if prev_out is None:
            init = src
        else:
            fb = flow(src, prev_src)            # current -> previous
            ff = flow(prev_src, src)            # previous -> current
            warped = warp(prev_out, fb)
            mask = soften_mask(consistency(fb, ff), job.mask_dilate, job.mask_blur)
            init = torch.lerp(src, warped, mask * job.flow_blend)
            if job.diff and style_next < job.style:
                trust = True
                # trusted -> style_next, untrusted (new content, occlusions) -> full style
                amount = torch.lerp(torch.ones_like(mask), torch.full_like(mask, style_next / job.style),
                                    mask * job.flow_blend)
            if i % 10 == 1:                     # a few debug snapshots, not every frame
                to_image(mask.expand(-1, 3, -1, -1)).save(out / "debug" / f"mask_{i:06d}.png")
                to_image(init).save(out / "debug" / f"init_{i:06d}.png")

        hints = [ann(c.kind, src_np) for c in job.controlnets]
        for c, h in zip(job.controlnets, hints):
            h.convert("RGB").save(out / "control" / c.kind / f"{i:06d}.jpg", quality=90)
        if job.diff_source != "off":
            if job.diff_source == "depth":      # reuse the depth ControlNet's map when there is one
                depth = next((h for c, h in zip(job.controlnets, hints) if c.kind == "depth"), None)
                base = to_tensor(np.asarray((depth or ann("depth", src_np)).convert("RGB")), device)
            else:
                base = src
            dm = diff_mask(base, job)
            to_image(dm.expand(-1, 3, -1, -1)).save(out / "control" / "diff" / f"{i:06d}.jpg", quality=90)
            amount = dm if amount is None else amount * dm
            if amount.min() >= 1:
                amount = None
        g = torch.Generator(device="cpu").manual_seed(int(job.seed) + i)
        if embeds is None:
            text = dict(prompt=job.prompt, negative_prompt=job.negative or None)
        else:
            text, cond = travel(keys, embeds, blend, (job.frame_start + i) * max(1, job.nth))
            if len(keys) > 1 and cond != last_cond and not cond.endswith("%"):
                log(f"frame {i + 1}: {cond}")
            last_cond = cond
        kw = dict(**text, image=to_image(init),
                  strength=job.style if i == 0 or trust else style_next,
                  num_inference_steps=job.steps, guidance_scale=job.cfg, generator=g)
        dd = DiffDiffusion(pipe, amount) if amount is not None else contextlib.nullcontext()
        if amount is not None:
            kw.update(callback_on_step_end=dd, callback_on_step_end_tensor_inputs=["latents"])
        # ControlNet start/end are fractions of the run's steps. A differential run is longer (full
        # style), so rescale them to switch at the same noise levels the trusted pixels saw before.
        share = style_next / job.style if trust else 1.0
        cn = job.controlnets
        starts = [c.start if c.start == 0 else 1 - share + c.start * share for c in cn]
        ends = [1 - share + c.end * share for c in cn]
        if union:
            # one network, every hint at once, each tagged with its condition type
            kw.update(control_image=hints, control_mode=[UNION_MODE[c.kind] for c in cn],
                      controlnet_conditioning_scale=[c.weight for c in cn],
                      control_guidance_start=starts, control_guidance_end=ends)
        elif cn:
            single = len(cn) == 1
            kw.update(control_image=hints[0] if single else hints,
                      controlnet_conditioning_scale=cn[0].weight if single else [c.weight for c in cn],
                      control_guidance_start=starts[0] if single else starts,
                      control_guidance_end=ends[0] if single else ends)
        with dd:
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
