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
import re
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
    kind: str            # depth | softedge | hed | canny | lineart | tile (the frame itself: refine / upscale)
    path: str            # ControlNet weights file
    weight: float = 1.0
    start: float = 0.0
    end: float = 1.0
    repo: str = ""       # Hugging Face repo to fetch `path` from on first use, when it isn't on disk yet

    def __post_init__(self):             # JSON sends 1.0 as 1; a single-net pipeline refuses an int weight
        self.weight, self.start, self.end = float(self.weight), float(self.start), float(self.end)


@dataclass
class RenderJob:
    video: str
    out_dir: str
    checkpoint: str
    family: str = "sd15"                 # sd15 | sd2 | sdxl
    prompt: str = ""
    negative: str = ""
    width: int = 512
    height: int = 512
    frame_start: int = 0                 # in extracted-frame units
    frame_end: int = -1                  # inclusive; -1 = to the end
    nth: int = 1                         # use every nth source frame
    lora: str = ""                       # a LoRA file for the checkpoint ("" = none), at lora_weight
    lora_weight: float = 0.8
    # Style ref (IP-Adapter): a picture whose look steers every frame. style_only: only the layers that carry
    # style (InstantStyle), so its colours / strokes / mood come through without its content
    style_image: str = ""
    style_scale: float = 0.6
    style_only: bool = True
    ip_adapter: str = ""                 # the mode's IP-Adapter file; missing = fetched from ip_adapter_repo
    ip_adapter_repo: str = ""            # "repo::path/in/repo"
    image_encoder: str = ""              # its CLIP ViT-H image encoder (HF layout, one safetensors file)
    image_encoder_repo: str = ""
    embeddings_dir: list = field(default_factory=list)   # folders of textual-inversion embeddings (SD 1.5):
                                         # a file's name typed in a prompt loads it, as in A1111 / ComfyUI
    # Text -> Video: no clip. Frame 1 comes from the prompt alone, every later one from the previous
    # result moved by a camera (per clip frame; on twos a drawing moves `nth` frames' worth) and
    # repainted at `style`. 0 = video -> video.
    t2v_frames: int = 0
    init_image: str = ""                 # Image -> Video: frame 1 is this picture repainted at `style`, not
                                         # the prompt alone; ControlNet / DepthDiff read each frame's start
    cam_zoom: float = 1.01               # scale per frame (>1 pushes in)
    cam_rotate: float = 0.0              # degrees per frame
    cam_x: float = 0.0                   # pan per frame, share of the width (+ = the image drifts right)
    cam_y: float = 0.0                   # share of the height (+ = down)
    cam_3d: bool = False                 # 3D (Disco / Deforum): the frame's depth lifts it into space, so the
                                         # camera moves through it with parallax. Zoom = dolly, Rotate = roll
    cam_yaw: float = 0.0                 # 3D: degrees per frame the camera turns (+ = right)
    cam_pitch: float = 0.0               # 3D: degrees per frame it tilts (+ = up)
    hold: int = 1                        # mp4: show each rendered frame this many times (on twos: nth 2, hold 2 —
                                         # 12 drawings a second, played at the clip's 24 fps)
    style: float = 0.75                  # img2img strength on the first frame
    style_next: float = -1               # strength on later frames; -1 = 0.65 * style. Lower keeps
                                         # more of the warped previous frame (less boiling)
    tile: int = 0                        # > 0: frames larger than this are denoised in overlapping tiles of
                                         # this size (refine / upscale: detail at the scale SD was trained at,
                                         # VRAM flat whatever the frame size). 0 = whole frame
    tile_overlap: int = 128
    steps: int = 20
    sampler: str = "dpmpp_2m"            # see SAMPLERS
    schedule: str = "karras"             # karras | normal | exponential | beta | trailing (what the sampler supports)
    cfg: float = 6.0
    seed: int = 0
    controlnets: list[ControlSpec] = field(default_factory=list)
    # flow feedback
    flow_blend: float = 1.0              # 1 = trust the warped stylized frame fully where consistent
    fresh: bool = False                  # "Boil": no warp, every frame repainted from its source at the
                                         # full style — strokes boil like hand-painted animation
    mask_dilate: int = 5                 # grow the "don't trust" regions a little (px)
    mask_blur: int = 3
    color_match: float = 0.5             # 0..1 pull colour stats toward frame 0 (fights drift)
    diff: bool = False                   # differential diffusion: per-pixel strength from the trust
                                         # mask — trusted pixels get style_next, uncovered ones full style
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
    # Shape: a mask picture / video (white = the subject) that steers the render on its own, apart from the
    # clip. Mask frame N drives render frame N; a shorter mask holds its last frame.
    shape: str = ""
    shape_invert: bool = False           # black is the subject (a dark logo on white)
    shape_start: int = 0                 # the mask frame (its own numbering) render frame 1 uses: Single Frame at a spot
    shape_blur: int = 0                  # px of softening on the mask's edges (0 = hard, as drawn)
    shape_cn: str = ""                   # the mode's depth ControlNet, for Pull
    shape_cn_repo: str = ""              # where it downloads from on first use, when missing
    shape_pull: float = 0.0              # the mask as a depth hint at this weight: something near where it's white
    shape_repaint: float = 0.0           # white repaints at the frame's strength (where it newly spreads: almost
                                         # fully), black holds at (1 - repaint) of it. Any shape, any animation
    shape_follow: float = 0.0            # Text / Image: the picture moves and scales with the mask (centre + size),
                                         # and the inside repaints enough to re-form there. One blob growing / moving
    shape_nudge: float = 0.0             # the outside of the mask darkened in each frame's start (Text -> Video frame 1 too)

    @staticmethod
    def from_dict(d: dict) -> "RenderJob":
        d = dict(d)
        d["controlnets"] = [ControlSpec(**c) for c in d.get("controlnets", [])]
        return RenderJob(**d)


# ------------------------------------------------------------------ frames
def extract_frames(job: RenderJob, dest: Path) -> list[Path]:
    """Source frames at render size, numbered from 0 in extracted-frame units. `job.video` may also be a
    folder of PNGs (Refine: a run's output frames), resized with Lanczos."""
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.jpg"):
        old.unlink()
    nth = max(1, job.nth)
    if Path(job.video).is_dir():
        pngs = sorted(Path(job.video).glob("*.png"))[job.frame_start * nth::nth]
        if job.frame_end >= 0:
            pngs = pngs[:job.frame_end - job.frame_start + 1]
        for k, f in enumerate(pngs):
            Image.open(f).convert("RGB").resize((job.width, job.height), Image.LANCZOS).save(
                dest / f"{k:06d}.jpg", quality=95)
        if not pngs:
            raise RuntimeError(f"no frames in {job.video}")
        return sorted(dest.glob("*.jpg"))
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


def shape_frames(job: RenderJob, dest: Path) -> list[Path]:
    """The Shape mask's frames: a picture (one frame), a folder of images, or a video (every nth frame from
    its start, at render size)."""
    src = Path(job.shape)
    if src.is_dir():
        return sorted(f for f in src.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))[::max(1, job.nth)]
    if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        return [src]
    dest.mkdir(parents=True, exist_ok=True)
    vf = f"select=not(mod(n\\,{max(1, job.nth)})),scale={job.width}:{job.height}:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-vsync", "vfr",
                    "-q:v", "2", "-start_number", "0", str(dest / "%06d.jpg")], check=True)
    frames = sorted(dest.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"no frames in the shape video {src.name}")
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
FLOW_MAX = 1024                          # long side RAFT measures at; the flow is scaled back up


class Flow:
    """RAFT-large from torchvision. flow(a, b)[p] = where pixel p of a moved to in b."""

    def __init__(self, device):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.device = device
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(device)

    @torch.inference_mode()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2:]
        # RAFT wants dims divisible by 8 and inputs in [-1, 1]; big frames are measured at FLOW_MAX
        k = min(1.0, FLOW_MAX / max(h, w))
        H, W = math.ceil(h * k / 8) * 8, math.ceil(w * k / 8) * 8
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
            elif kind == "hed":                  # the SD 2.1 edge ControlNet was trained on HED maps
                det = ca.HEDdetector.from_pretrained("lllyasviel/Annotators")
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
        if kind == "tile":                       # no detector: the tile net reads the frame as it is
            return Image.fromarray(img)
        det = self._get(kind)
        res = min(h, w)
        if kind == "canny":
            out = det(Image.fromarray(img), low_threshold=100, high_threshold=200,
                      detect_resolution=res, image_resolution=res)
        elif kind in ("softedge", "hed"):
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
UNION_MODE = {"depth": 1, "softedge": 2, "canny": 3, "lineart": 3, "tile": 6, "shape": 1}   # shape = a mask read as depth


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


# Samplers: diffusers class + fixed kwargs, and the noise schedules each supports (first = its default).
SAMPLERS = {
    "dpmpp_2m":     ("DPMSolverMultistepScheduler", {}, ("karras", "normal", "exponential", "beta", "trailing")),
    "dpmpp_2m_sde": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++"},
                     ("karras", "normal", "exponential", "beta", "trailing")),
    "euler":        ("EulerDiscreteScheduler", {}, ("karras", "normal", "exponential", "beta", "trailing")),
    "euler_a":      ("EulerAncestralDiscreteScheduler", {}, ("normal", "trailing")),
    "unipc":        ("UniPCMultistepScheduler", {}, ("karras", "normal", "exponential", "beta", "trailing")),
    "ddim":         ("DDIMScheduler", {}, ("normal", "trailing")),
    "heun":         ("HeunDiscreteScheduler", {}, ("karras", "normal", "exponential", "beta", "trailing")),
    "lcm":          ("LCMScheduler", {}, ("normal", "trailing")),
}
SCHEDULE_KW = {"karras": {"use_karras_sigmas": True}, "exponential": {"use_exponential_sigmas": True},
               "beta": {"use_beta_sigmas": True}, "trailing": {"timestep_spacing": "trailing"}, "normal": {}}


def make_scheduler(base_config: dict, sampler: str, schedule: str):
    """A fresh scheduler for this render from the model's own config (so v-prediction etc. carry over);
    swapping it never reloads the model. An unknown sampler falls back to DPM++ 2M, an unsupported
    schedule to the sampler's first."""
    import diffusers
    cls, fixed, schedules = SAMPLERS.get(sampler, SAMPLERS["dpmpp_2m"])
    kw = SCHEDULE_KW[schedule if schedule in schedules else schedules[0]]
    # the stored config carries the load-time spacing (Karras on): clear every spacing flag first, or
    # Exponential / Beta would ask for two at once
    base = {**base_config, "use_karras_sigmas": False, "use_exponential_sigmas": False, "use_beta_sigmas": False}
    return getattr(diffusers, cls).from_config(base, **fixed, **kw)


def camera_move(img: torch.Tensor, job: RenderJob) -> torch.Tensor:
    """Text -> Video's motion: the previous frame zoomed / rotated / panned by the camera, for as many
    clip frames as one drawing spans (nth). Edges reflect instead of going black."""
    n = max(1, job.nth)
    zoom, ang = job.cam_zoom ** n, math.radians(job.cam_rotate * n)
    tx, ty = -2 * job.cam_x * n, -2 * job.cam_y * n          # grid units: the image moves the other way
    c, s_ = math.cos(ang) / zoom, math.sin(ang) / zoom
    theta = torch.tensor([[c, -s_, tx], [s_, c, ty]], dtype=img.dtype, device=img.device)[None]
    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    return F.grid_sample(img, grid, mode="bicubic", padding_mode="reflection", align_corners=False).clamp(0, 1)


def shape_follow(img: torch.Tensor, prev: torch.Tensor, cur: torch.Tensor, amount: float) -> torch.Tensor:
    """Move img the way the Shape mask moved from prev to cur: its centre and size (image moments), so
    what was painted on the shape travels and grows with it. amount 0..1 = how far along."""
    _, _, h, w = cur.shape
    ys = torch.linspace(-1, 1, h, device=cur.device)[:, None]
    xs = torch.linspace(-1, 1, w, device=cur.device)[None, :]

    def moments(m):
        a = m[0, 0].sum()
        return a, (m[0, 0] * xs).sum() / a, (m[0, 0] * ys).sum() / a
    a0, x0, y0 = moments(prev)
    a1, x1, y1 = moments(cur)
    if a0 < 1 or a1 < 1:                        # an empty mask: nothing to follow
        return img
    k = float((a1 / a0).sqrt()) ** amount      # size ratio
    x1, y1 = x0 + (x1 - x0) * amount, y0 + (y1 - y0) * amount
    # each output point samples where it was before the move: prev centre + (p - new centre) / k
    theta = torch.tensor([[1 / k, 0, float(x0 - x1 / k)], [0, 1 / k, float(y0 - y1 / k)]],
                         dtype=img.dtype, device=img.device)[None]
    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    return F.grid_sample(img, grid, mode="bicubic", padding_mode="reflection", align_corners=False).clamp(0, 1)


CAM_KEYS = ("cam_zoom", "cam_rotate", "cam_x", "cam_y", "cam_yaw", "cam_pitch", "cam_3d")   # Live can change
CAM_EASE = 0.25                          # Live camera: share of the way to a new value per frame (~90% in 8)
CAM_FOV = 50.0                           # 3D camera: horizontal field of view, degrees
CAM_NEAR, CAM_FAR = 1.0, 6.0             # depth range the MiDaS map spans (nearest = 1): far / near = parallax
CAM_REF = 2.0                            # the depth where Zoom and Pan match the 2D camera's amounts


def camera_move_3d(img: torch.Tensor, disparity: torch.Tensor, job: RenderJob) -> torch.Tensor:
    """3D Text -> Video: each pixel of the previous frame placed at its depth (MiDaS: 1 = near), the
    camera moved / turned, the frame projected back. Near things slide and grow faster than far ones.
    Backward: every new pixel takes the depth found at its own spot (as Deforum does), so no holes."""
    n = max(1, job.nth)
    _, _, h, w = img.shape
    dev, dt = img.device, torch.float32
    z = 1.0 / (disparity.to(dt).clamp(0, 1) * (1 / CAM_NEAR - 1 / CAM_FAR) + 1 / CAM_FAR)   # 1xHxW-ish
    fx = 1.0 / math.tan(math.radians(CAM_FOV) / 2)
    fy = fx * w / h
    ys, xs = torch.meshgrid((torch.arange(h, device=dev, dtype=dt) + 0.5) / h * 2 - 1,
                            (torch.arange(w, device=dev, dtype=dt) + 0.5) / w * 2 - 1, indexing="ij")
    z = z.reshape(h, w)
    p = torch.stack([xs * z / fx, ys * z / fy, z], -1)                     # HxWx3, the new camera's view
    r, yw, pt = (math.radians(a * n) for a in (job.cam_rotate, job.cam_yaw, job.cam_pitch))
    rz = torch.tensor([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]], dtype=dt)
    ry = torch.tensor([[math.cos(yw), 0, math.sin(yw)], [0, 1, 0], [-math.sin(yw), 0, math.cos(yw)]], dtype=dt)
    rx = torch.tensor([[1, 0, 0], [0, math.cos(pt), -math.sin(pt)], [0, math.sin(pt), math.cos(pt)]], dtype=dt)
    rot = (ry @ rx @ rz).to(dev)
    # + zoom = the camera moves forward; + pan = the image drifts right / down, so the camera goes the other way
    t = torch.tensor([-2 * job.cam_x * n * CAM_REF / fx, -2 * job.cam_y * n * CAM_REF / fy,
                      (job.cam_zoom ** n - 1) * CAM_REF], dtype=dt, device=dev)
    q = p @ rot.T + t                                                        # the same points, old camera
    qz = q[..., 2].clamp(min=0.05)
    grid = torch.stack([fx * q[..., 0] / qz, fy * q[..., 1] / qz], -1)[None].to(img.dtype)
    return F.grid_sample(img, grid, mode="bicubic", padding_mode="reflection", align_corners=False).clamp(0, 1)


def tile_spans(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets of tiles covering `length`, evenly spread, neighbours sharing >= `overlap`."""
    if length <= tile:
        return [0]
    n = math.ceil((length - overlap) / (tile - overlap))
    return [round(k * (length - tile) / (n - 1) / 8) * 8 for k in range(n)]


def tile_weight(h: int, w: int, y: int, x: int, H: int, W: int, overlap: int, device) -> torch.Tensor:
    """1x1xhxw feather for a tile at (y, x) in an HxW frame: ramps over `overlap` on inner edges only."""
    def ramp(n, lo_edge, hi_edge):
        r = torch.ones(n, device=device)
        o = min(overlap, n // 2)
        up = (torch.arange(o, device=device) + 0.5) / o
        if not lo_edge:
            r[:o] = up
        if not hi_edge:
            r[n - o:] = torch.minimum(r[n - o:], up.flip(0))
        return r
    wy, wx = ramp(h, y == 0, y + h >= H), ramp(w, x == 0, x + w >= W)
    return (wy[:, None] * wx[None, :])[None, None]


def fetch_missing(job: RenderJob, log) -> None:
    """Models that download on first use (DWARP installs without them): a ControlNet with a `repo` and no
    file yet is fetched into its folder once, then it's just there."""
    for c in job.controlnets:
        p = Path(c.path)
        if c.repo and not p.is_file():
            from huggingface_hub import hf_hub_download
            log(f"downloading {p.name} from {c.repo} (first use, once)…")
            t = time.time()
            hf_hub_download(c.repo, p.name, local_dir=str(p.parent))
            log(f"downloaded {p.name} ({p.stat().st_size / 1e6:.0f} MB) in {time.time() - t:.0f}s")


def fetch_file(path: str, repo: str, log) -> None:
    """A model file that downloads on first use: `repo` is "hf-repo::path/in/repo", saved as `path`."""
    p = Path(path)
    if p.is_file() or not repo:
        return
    import shutil
    from huggingface_hub import hf_hub_download
    rid, rfile = repo.split("::")
    log(f"downloading {p.name} from {rid} (first use, once)…")
    t = time.time()
    got = hf_hub_download(rid, rfile)
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(got, p)
    log(f"downloaded {p.name} ({p.stat().st_size / 1e6:.0f} MB) in {time.time() - t:.0f}s")


# Style ref: which attention layers get the picture. None = all of them (content and style).
STYLE_LAYERS = {"sdxl": {"up": {"block_0": [0.0, 1.0, 0.0]}},    # InstantStyle's style block
                "sd15": {"up": {"block_1": [0.0, 1.0, 0.0]}}}


def style_embeds(pipe, job: RenderJob, device, log):
    """The job's style ref, ready to pass to every call as ip_adapter_image_embeds (None = no style ref).
    The IP-Adapter stays loaded between jobs; the image encoder only visits the GPU to encode the picture."""
    cur = getattr(pipe, "_dwarp_ipa", "")
    if not job.style_image:
        if cur:
            pipe.unload_ip_adapter()
            pipe._dwarp_ipa = ""
        return None
    if cur != job.ip_adapter:
        from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
        from safetensors.torch import load_file
        if cur:
            pipe.unload_ip_adapter()
        fetch_file(job.ip_adapter, job.ip_adapter_repo, log)
        fetch_file(job.image_encoder, job.image_encoder_repo, log)
        t = time.time()
        if _CACHE.get("img_enc_path") != job.image_encoder:
            cfg = CLIPVisionConfig(hidden_size=1280, intermediate_size=5120, num_hidden_layers=32, num_attention_heads=16,
                                   patch_size=14, image_size=224, projection_dim=1024, hidden_act="gelu")
            enc = CLIPVisionModelWithProjection(cfg)
            enc.load_state_dict(load_file(job.image_encoder), strict=False)
            _CACHE["img_enc"], _CACHE["img_enc_path"] = enc.to(torch.float16).eval(), job.image_encoder
        pipe.image_encoder, pipe.feature_extractor = _CACHE["img_enc"], CLIPImageProcessor()
        ipa = Path(job.ip_adapter)
        pipe.load_ip_adapter(str(ipa.parent), subfolder="", weight_name=ipa.name, image_encoder_folder=None)
        pipe._dwarp_ipa = job.ip_adapter
        log(f"style ref: {ipa.stem} loaded in {time.time() - t:.1f}s")
    layers = STYLE_LAYERS.get(job.family) if job.style_only else None
    pipe.set_ip_adapter_scale(_scaled(layers, job.style_scale) if layers else job.style_scale)
    enc = pipe.image_encoder.to(device)
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=Image.open(job.style_image).convert("RGB"), ip_adapter_image_embeds=None, device=device,
        num_images_per_prompt=1, do_classifier_free_guidance=job.cfg > 1)
    enc.to("cpu")                               # ~1.2 GB back for the frames
    torch.cuda.empty_cache()
    log(f"style ref {Path(job.style_image).name} at {job.style_scale}" + (" (style only)" if layers else ""))
    return embeds


def _scaled(layers, k: float):
    """STYLE_LAYERS with every weight times k."""
    if isinstance(layers, dict):
        return {n: _scaled(v, k) for n, v in layers.items()}
    if isinstance(layers, list):
        return [v * k for v in layers]
    return layers * k


def load_pipeline(job: RenderJob, device, dtype=torch.float16):
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler

    paths = list(dict.fromkeys(c.path for c in job.controlnets))
    union = len(paths) == 1 and is_union(paths[0])
    # disable_mmap: memory-mapping multi-GB checkpoints crashes the process on Windows with an
    # access violation (seen with 7 GB SDXL files on a busy HDD); a plain read into RAM doesn't.
    nets = [load_union(paths[0], dtype)] if union else \
        [ControlNetModel.from_single_file(c.path, torch_dtype=dtype, disable_mmap=True,
                                          # SD 2.1 nets: diffusers would guess an SD 1.5 layout; thibaud's
                                          # diffusers repo carries the 2.1 one (same for all his nets)
                                          **({"config": "thibaud/controlnet-sd21-depth-diffusers"} if job.family == "sd2" else {}))
         for c in job.controlnets]
    import diffusers
    # No ControlNets = plain img2img: lighter and faster, structure comes from the flow warp alone.
    name = "StableDiffusionXLControlNetUnionImg2ImgPipeline" if union else \
           {("sdxl", True): "StableDiffusionXLControlNetImg2ImgPipeline",
            ("sdxl", False): "StableDiffusionXLImg2ImgPipeline",
            ("sd15", True): "StableDiffusionControlNetImg2ImgPipeline",
            ("sd15", False): "StableDiffusionImg2ImgPipeline"}[("sdxl" if job.family == "sdxl" else "sd15", bool(nets))]
    Pipe = getattr(diffusers, name)
    kwargs = dict(torch_dtype=dtype, disable_mmap=True)
    if nets:
        kwargs["controlnet"] = nets if len(nets) != 1 else nets[0]
    if job.family != "sdxl":
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    elif is_ssd1b(job.checkpoint):
        kwargs["unet"] = load_ssd1b_unet(job.checkpoint, dtype)
    source = job.checkpoint
    if job.family == "sd2":
        from models import converted_path, ckpt_to_safetensors, sd2_prediction
        # Stability took its SD 2.1 repos off the Hub: the architecture configs come from the community
        # mirror (512 base / 768-v). Old SD 2 .ckpt pickles fail diffusers' strict loader, so a .ckpt is
        # converted once to fp16 safetensors under models/converted.
        v = sd2_prediction(Path(job.checkpoint)) == "v_prediction"
        kwargs["config"] = "sd2-community/stable-diffusion-2-1" if v else "sd2-community/stable-diffusion-2-1-base"
        if job.checkpoint.lower().endswith(".ckpt"):
            conv = converted_path(Path(job.checkpoint), Path(__file__).resolve().parents[1] / "models" / "converted")
            source = str(conv if conv.is_file() else ckpt_to_safetensors(Path(job.checkpoint), conv))
    pipe = Pipe.from_single_file(source, **kwargs)
    extra = {}
    if job.family == "sd2":                     # 768 models predict v, 512 base models noise (models.py)
        from models import sd2_prediction
        extra["prediction_type"] = sd2_prediction(Path(job.checkpoint))
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, **extra)
    pipe._dwarp_scheduler_config = dict(pipe.scheduler.config)   # make_scheduler starts from this each render
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


EMBED_EXTS = (".pt", ".safetensors", ".bin")


def embedding_files(dirs) -> dict[str, Path]:
    """Embedding name (the file name, as typed in a prompt) -> file, first folder wins."""
    found: dict[str, Path] = {}
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in EMBED_EXTS and f.stem not in found:
                    found[f.stem] = f
    return found


def lora_family(f: Path) -> str | None:
    """What a LoRA was trained for, from a cross-attention layer's input width (the text features):
    768 = SD 1.5, 1024 = SD 2.x, 2048 = SDXL. Reads only the safetensors header. None = not an SD LoRA
    (Flux, video models…) or unreadable."""
    try:
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n)) if 0 < n < 100_000_000 else {}
    except (OSError, ValueError):
        return None
    for k, v in header.items():
        low = k.lower()
        if "attn2" in low and ("to_k" in low or "k_proj" in low) and ("lora_down" in low or "lora_a" in low)                 and isinstance(v, dict) and len(v.get("shape", [])) == 2:
            return {768: "sd15", 1024: "sd2", 2048: "sdxl"}.get(v["shape"][1])
    return None


def load_lora(pipe, path: str) -> None:
    """pipe.load_lora_weights, with one fix: transformers 5 dropped CLIPTextModel's `text_model` level
    (CLIPTextModelWithProjection, SDXL's second encoder, kept it),
    but diffusers still names text-encoder LoRA layers `text_model.encoder...`, so they matched nothing
    (IndexError in get_peft_kwargs). Renamed here before loading."""
    # unet_config: maps LoRAs in Stability's layout (input_blocks.8.1...) onto diffusers' names, as
    # load_lora_weights does; without it those layers matched nothing (NoMatchingPeftModuleError)
    sd, alphas = pipe.lora_state_dict(path, unet_config=pipe.unet.config)
    for name in ("text_encoder", "text_encoder_2"):    # CLIPTextModel lost it; ...WithProjection kept it
        te = getattr(pipe, name, None)
        if te is not None and not hasattr(te, "text_model"):
            fix = lambda d, n=name: {k.replace(f"{n}.text_model.", f"{n}."): v for k, v in d.items()}
            sd, alphas = fix(sd), alphas and fix(alphas)
    pipe.load_lora_into_unet(sd, alphas, pipe.unet, adapter_name="style", _pipeline=pipe)
    for name in ("text_encoder", "text_encoder_2"):
        te = getattr(pipe, name, None)
        if te is not None and any(k.startswith(name + ".") for k in sd):
            pipe.load_lora_into_text_encoder(sd, alphas, te, prefix=name, adapter_name="style", _pipeline=pipe)


def apply_lora(pipe, job: RenderJob, log) -> None:
    """The job's LoRA in the loaded pipeline, swapped without reloading the model: the old one unloaded,
    the new one loaded once, its strength set every render."""
    cur = getattr(pipe, "_dwarp_lora", "")
    if cur != job.lora:
        if cur:
            pipe.unload_lora_weights()
        if job.lora:
            t = time.time()
            try:
                load_lora(pipe, job.lora)
            except Exception:                   # a half-loaded LoRA would block the next one's name
                pipe.unload_lora_weights()
                pipe._dwarp_lora = ""
                raise
            log(f"LoRA {Path(job.lora).stem} loaded in {time.time() - t:.1f}s")
        pipe._dwarp_lora = job.lora
    if job.lora:
        pipe.set_adapters(["style"], [job.lora_weight])
        log(f"LoRA {Path(job.lora).stem} at {job.lora_weight}")


def embedding_family(f: Path) -> str | None:
    """What an embedding was trained for, from its vector width: 768 = SD 1.5, 1024 = SD 2.x, clip_g /
    clip_l pairs = SDXL. None if unreadable. Only the tensors' shapes matter; files are small."""
    try:
        if f.suffix.lower() == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(f), "pt") as st:
                keys = list(st.keys())
                if any("clip_g" in k for k in keys):
                    return "sdxl"
                widths = {st.get_slice(k).get_shape()[-1] for k in keys}
        else:
            d = torch.load(str(f), map_location="cpu", weights_only=True)
            d = d.get("string_to_param", d) if isinstance(d, dict) else {}
            widths = {v.shape[-1] for v in d.values() if hasattr(v, "shape")}
    except Exception:
        return None
    return "sd15" if widths == {768} else "sd2" if widths == {1024} else None


def load_embeddings(pipe, job: RenderJob, texts, log) -> None:
    """Load the embeddings named in these prompts into the pipeline's text encoder (once per loaded
    pipeline). A name matches as a whole word: `charcoalstyle-1000`, not `charcoalstyle-100` inside it."""
    files = embedding_files(job.embeddings_dir)    # a small folder: rescanned, so new ones show up
    if not files:
        return
    text = "\n".join(t for t in texts if t)
    named = [n for n in files if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text)]
    if not named:
        return
    if job.family == "sdxl":
        log(f"embeddings {named} are for SD 1.5 / SD 2.x — ignored with SDXL")
        return
    done = pipe.__dict__.setdefault("_dwarp_embeddings", set())
    for n in named:
        if n in done:
            continue
        try:
            pipe.load_textual_inversion(str(files[n]), token=n)
            done.add(n)
            log(f"embedding {n} loaded")
        except Exception as e:                  # an SDXL / broken file: say so, keep rendering
            done.add(n)
            log(f"embedding {n} skipped: {str(e).splitlines()[0][:120]}")


def encode_prompts(pipe, job: RenderJob, device, prompts: list[str], log=lambda m: None) -> list[dict]:
    """Encode each prompt once per job. SDXL's text encoders visit the GPU just for this."""
    load_embeddings(pipe, job, [*prompts, job.negative], log)
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

    def _init_amount(self):
        if self.held is None:                   # first step: everything below amount 1 is held
            self.amount = F.interpolate(self.amount_px.float(), self.orig.shape[-2:], mode="area").to(self.orig)
            self.held = self.was_held = self.amount < 1

    def _convert(self, *a, **k):
        x0 = self._orig_convert(*a, **k)
        self._init_amount()
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
        self._orig_prepare = self.pipe.prepare_latents
        self.pipe.prepare_latents = self._prepare_latents
        # multistep solvers (DPM++, UniPC) predict x0 through convert_model_output: pin held pixels there
        # too; Euler, DDIM, Heun, LCM have no such step and rely on the per-step holding alone
        self.hooked = hasattr(sch, "convert_model_output")
        if self.hooked:
            self._orig_convert = sch.convert_model_output
            sch.convert_model_output = self._convert
        return self

    def __exit__(self, *exc):
        del self.pipe.prepare_latents           # back to the class methods
        if self.hooked:
            del self.pipe.scheduler.convert_model_output

    def __call__(self, pipe, i, t, kw):
        # After step i the scheduler sits at step i+1; add_noise uses that level (sigma 0 at the end).
        n, k = pipe.num_timesteps, i + 1
        self._init_amount()
        self.held = self.amount < 1 - k / n
        if self.held.any():
            sch = pipe.scheduler
            if getattr(sch, "step_index", None) is None:
                # DDIM / LCM noise to the timestep they're given, not their own step: hand them the next one
                ts = sch.timesteps
                at = (ts == t).nonzero()
                nxt = ts[at[0, 0] + 1] if len(at) and at[0, 0] + 1 < len(ts) else ts[-1]
                t = nxt
            ref = sch.add_noise(self.orig, self.noise, t.reshape(1))
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
    and/or "now": a prompt that becomes a keyframe at the frame about to render (Live mode), and/or
    "camera": new Text / Image -> Video camera values ({cam_zoom, cam_rotate, ...}), eased in over a few frames.
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
    if job.t2v_frames and job.cam_3d:           # the depth each 3D camera move used
        (out / "control" / "depth").mkdir(parents=True, exist_ok=True)
    (out / "job.json").write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")

    t0 = time.time()
    shapes, shape_hint = [], None
    if job.shape:
        shapes = shape_frames(job, out / "shape_src")
        (out / "control" / "shape").mkdir(parents=True, exist_ok=True)
        # Pull: the mask is a depth hint. With the clip's depth net already on, it joins that hint (the
        # nearer of the two); otherwise it's a net of its own
        if job.shape_pull > 0:
            if any(c.kind == "depth" for c in job.controlnets):
                shape_hint = "merge"
            elif job.shape_cn:
                job.controlnets = [*job.controlnets, ControlSpec("shape", job.shape_cn, job.shape_pull, repo=job.shape_cn_repo)]
                shape_hint = "own"
        log(f"shape {Path(job.shape).name}: {len(shapes)} frame{'s' * (len(shapes) > 1)}, pull {job.shape_pull}"
            f"{' (joins the depth hint)' if shape_hint == 'merge' else ''}, repaint {job.shape_repaint}, follow {job.shape_follow}, nudge {job.shape_nudge}")
    t2v = job.t2v_frames > 0
    if t2v:                                     # no clip: the frames come from the prompt and the camera
        src_paths = [None] * job.t2v_frames
        total = job.t2v_frames
        log(f"{'image' if job.init_image else 'text'} -> video: {total} frames at {job.width}x{job.height}, {'3D ' if job.cam_3d else ''}camera zoom {job.cam_zoom} rotate "
            f"{job.cam_rotate} pan {job.cam_x}/{job.cam_y}" + (f" turn {job.cam_yaw} tilt {job.cam_pitch}" if job.cam_3d else "")
            + " per frame" + (f", on {job.nth}s" if job.nth > 1 else ""))
    else:
        progress("extracting", 0, 0, "extracting frames")
        src_paths = extract_frames(job, out / "src")
        total = len(src_paths)
        log(f"extracted {total} frames at {job.width}x{job.height} (every {job.nth}) in {time.time() - t0:.1f}s")

    progress("loading", 0, total, "loading models")
    fetch_missing(job, log)
    pipe = cached_pipeline(job, device, log)
    apply_lora(pipe, job, log)
    ip_embeds = style_embeds(pipe, job, device, log)
    pipe.scheduler = make_scheduler(pipe._dwarp_scheduler_config, job.sampler, job.schedule)
    log(f"sampler {job.sampler}, schedule {job.schedule} ({type(pipe.scheduler).__name__})")
    if total > 1 and not job.fresh and not t2v and "flow" not in _CACHE:
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
            encoded.update(zip(new, encode_prompts(pipe, job, device, new, log)))
        return [encoded[p] for _, p in ks]

    load_embeddings(pipe, job, [p for _, p in keys] + [job.prompt, job.negative], log)   # the plain-text path too
    embeds = embeds_for(keys) if job.family == "sdxl" or len(keys) > 1 else None
    if len(keys) > 1:
        log(f"prompt travel: {len(keys)} keyframes at source frames {[f for f, _ in keys]}, blend {blend}")
    edits = []                                  # live prompt edits, kept in job.json next to the start state
    cam_target = {k: getattr(job, k) for k in CAM_KEYS}   # Live camera: where each value is heading
    on_keys(keys, blend)
    last_cond = None
    union = type(getattr(pipe, "controlnet", None)).__name__ == "ControlNetUnionModel"
    written, prev_src, prev_out, first_out, prev_smask = [], None, None, None, None
    style_next = job.style_next if job.style_next >= 0 else round(job.style * 0.65, 3)
    if t2v:
        log(f"frame 1 from {'the image ' + Path(job.init_image).name if job.init_image else 'the prompt'}, then denoise {job.style} on the moved previous frame; colour match {job.color_match}")
    elif job.fresh:
        log(f"boil: every frame repainted from its source (no warp), denoise {job.style}; colour match {job.color_match}")
    else:
        log(f"denoise {job.style} on frame 0, {style_next} after{' (untrusted pixels: full style)' if job.diff else ''}; "
            f"colour match {job.color_match}")
    for i, sp in enumerate(src_paths):
        if cancelled():
            log("cancelled")
            break
        upd = live()
        if upd and upd.get("camera"):             # Live camera: new targets; the 3D switch flips at once
            cam = {k: v for k, v in upd.pop("camera").items() if k in CAM_KEYS}
            cam_target.update(cam)
            if "cam_3d" in cam:
                job.cam_3d = bool(cam["cam_3d"])
                if job.cam_3d:
                    (out / "control" / "depth").mkdir(parents=True, exist_ok=True)
            edits.append({"from_frame": i, "camera": cam})
            (out / "job.json").write_text(json.dumps(asdict(job) | {"live_edits": edits}, indent=2), encoding="utf-8")
            log(f"live camera from frame {i + 1}: " + ", ".join(f"{k[4:]} {v}" for k, v in cam.items()))
        for k in CAM_KEYS[:-1]:                 # ease: a share of the way each frame, so moves don't jerk
            setattr(job, k, getattr(job, k) + (float(cam_target[k]) - getattr(job, k)) * CAM_EASE)
        if upd and any(k in upd for k in ("prompt_keys", "prompt_blend", "now")):
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
        smask = None
        if shapes:                              # this frame's Shape mask, 1x1xHxW 0..1 (white = the subject)
            sm = Image.open(shapes[min(i + job.shape_start // max(1, job.nth), len(shapes) - 1)]).convert("L").resize((job.width, job.height), Image.LANCZOS)
            smask = torch.from_numpy(np.asarray(sm, dtype=np.float32) / 255).to(device)[None, None]
            if job.shape_invert:
                smask = 1 - smask
            smask = gaussian_blur(smask, int(job.shape_blur))
        if t2v:                                 # the "source" is where this frame starts: blank, then the camera
            if prev_out is None and job.init_image:     # Image -> Video: the picture, at render size
                pic = Image.open(job.init_image).convert("RGB").resize((job.width, job.height), Image.LANCZOS)
                src = to_tensor(np.asarray(pic), device)
            elif prev_out is None:
                src = torch.full((1, 3, job.height, job.width), 0.5, device=device)
                if smask is not None and job.shape_nudge > 0:
                    # black outside: what the latent mask holds there. Neutral grey inside: a black start
                    # leaks into step 1 (SD 2.1 then paints the shape black too); grey is near zero in latent
                    core = ((smask - 0.8) / 0.2).clamp(0, 1)     # the soft edge starts black too: no grey halo
                    src = src * torch.lerp(torch.ones_like(smask), core, job.shape_nudge)
            elif job.cam_3d:                    # the previous frame's own depth carries it into space
                prev_np = (prev_out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                dmap = ann("depth", prev_np).convert("L")
                dmap.save(out / "control" / "depth" / f"{i:06d}.jpg", quality=90)
                disp = torch.from_numpy(np.asarray(dmap, dtype=np.float32) / 255).to(device)[None, None]
                src = camera_move_3d(prev_out, gaussian_blur(disp, 2), job)   # soft steps, fewer tears
            else:
                src = camera_move(prev_out, job)
            if prev_out is not None and prev_smask is not None and job.shape_follow > 0:   # it travels with the shape
                src = shape_follow(src, prev_smask, smask, job.shape_follow)
            src_np = (src[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(src_np).save(out / "src" / f"{i:06d}.jpg", quality=90)   # the viewer's Source layer
        else:
            src_np = load_rgb(sp)
            src = to_tensor(src_np, device)

        amount = None                           # per-pixel strength, as a share of `strength`
        trust = False                           # job.diff: untrusted pixels get full style
        if prev_out is None or job.fresh or t2v:   # fresh / t2v: every frame starts from its own start
            init = src
        else:
            fb = flow(src, prev_src)            # current -> previous
            ff = flow(prev_src, src)            # previous -> current
            warped = warp(prev_out, fb)
            mask = soften_mask(consistency(fb, ff), job.mask_dilate, job.mask_blur)
            init = torch.lerp(src, warped, mask * job.flow_blend)
            # Uncovered pixels (things moving away, new content) have no paint to carry: they start from the
            # raw source, and at a low Next frames the footage shows through and stays. They get the full
            # Denoise, the rest style_next. Only when there is some: those frames cost the full step count.
            if job.diff and style_next < job.style and (mask < 0.5).float().mean() > 0.001:
                trust = True
                amount = torch.lerp(torch.ones_like(mask), torch.full_like(mask, style_next / job.style), mask)
            if i % 10 == 1:                     # a few debug snapshots, not every frame
                to_image(mask.expand(-1, 3, -1, -1)).save(out / "debug" / f"mask_{i:06d}.png")
                to_image(init).save(out / "debug" / f"init_{i:06d}.png")

        if smask is not None:
            if job.shape_nudge > 0:             # a bright subject on dark: the start already has the layout
                init = init * (1 - job.shape_nudge * (1 - smask))
        blank = t2v and i == 0 and not job.init_image     # t2v frame 1: from the prompt alone (grey start)
        hints = [Image.fromarray((smask[0, 0].cpu().numpy() * 255).astype(np.uint8)).convert("RGB") if c.kind == "shape"
                 else ann(c.kind, src_np) for c in job.controlnets]
        if shape_hint == "merge":
            hints = [Image.fromarray(np.maximum(np.asarray(h), (smask[0, 0].cpu().numpy() * 255 * job.shape_pull).astype(np.uint8)[..., None]))
                     if c.kind == "depth" else h for c, h in zip(job.controlnets, hints)]
        if smask is not None:
            to_image(smask.expand(-1, 3, -1, -1)).save(out / "control" / "shape" / f"{i:06d}.jpg", quality=90)
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
        grow = None                             # Repaint / Follow: per-pixel strength, the run's own set below
        if smask is not None and (job.shape_repaint > 0 or job.shape_follow > 0) and not blank:   # a blank start has nothing to hold
            # Repaint: white at the frame's strength, black holds; where the mask newly spreads (a wipe's edge,
            # a shape growing) gets near-full repaint so what's revealed is drawn fresh.
            # Follow: the inside also gets enough to re-form at the shape's new size
            ring = (smask - prev_smask).clamp(0, 1) if prev_smask is not None else torch.zeros_like(smask)
            ring = gaussian_blur((ring * 4).clamp(0, 1), 6)
            base_s = job.style if t2v or i == 0 or job.fresh else style_next
            top = max(base_s, 0.5 + 0.45 * max(job.shape_repaint, job.shape_follow))
            inside = base_s + (max(base_s, 0.8) - base_s) * job.shape_follow
            per = torch.lerp(torch.full_like(smask, base_s * (1 - job.shape_repaint)), torch.full_like(smask, inside), smask)
            per = torch.maximum(per, ring * top)
            grow = float(per.max())
            keep = per / grow
            amount = keep if amount is None else amount * keep
        prev_smask = smask
        if smask is not None and blank and job.shape_nudge > 0:
            # frame 1 from pure noise with the mask in the latent: inside generates from the first step,
            # outside stays held at black (full Nudge: all of it; less lets some of the outside generate too)
            amount = torch.lerp(torch.ones_like(smask), smask, job.shape_nudge)
        if amount is not None and amount.min() >= 1:
            amount = None
        g = torch.Generator(device="cpu").manual_seed(int(job.seed) + i)
        if embeds is None:
            text = dict(prompt=job.prompt, negative_prompt=job.negative or None)
        else:
            text, cond = travel(keys, embeds, blend, (job.frame_start + i) * max(1, job.nth))
            if len(keys) > 1 and cond != last_cond and not cond.endswith("%"):
                log(f"frame {i + 1}: {cond}")
            last_cond = cond
        strength = (1.0 if blank else   # Nudge: the shape shows through
                    max(grow, 1.001 / job.steps) if grow is not None else
                    # at least one step: diffusers refuses strength * steps < 1 (Denoise near 0 = the source back)
                    max(job.style if i == 0 or trust or job.fresh or t2v else style_next, 1.001 / job.steps))
        # ControlNet start/end are fractions of the run's steps. A differential run is longer (full
        # style), so rescale them to switch at the same noise levels the trusted pixels saw before.
        share = style_next / job.style if trust else 1.0
        cn = job.controlnets
        starts = [c.start if c.start == 0 else 1 - share + c.start * share for c in cn]
        ends = [1 - share + c.end * share for c in cn]

        def denoise(init_t, hint_imgs, amt, gen):
            """One img2img pass over init_t (a whole frame or a tile) -> 1x3xHxW tensor."""
            kw = dict(**text, image=to_image(init_t), strength=strength, num_inference_steps=job.steps,
                      guidance_scale=job.cfg, generator=gen)
            if ip_embeds is not None:
                kw["ip_adapter_image_embeds"] = ip_embeds
            if init_t.shape[2:] != (job.height, job.width):
                kw.update(height=init_t.shape[2], width=init_t.shape[3])
            dd = DiffDiffusion(pipe, amt) if amt is not None else contextlib.nullcontext()
            if amt is not None:
                kw.update(callback_on_step_end=dd, callback_on_step_end_tensor_inputs=["latents"])
            if union:
                # one network, every hint at once, each tagged with its condition type
                kw.update(control_image=hint_imgs, control_mode=[UNION_MODE[c.kind] for c in cn],
                          controlnet_conditioning_scale=[c.weight for c in cn],
                          control_guidance_start=starts, control_guidance_end=ends)
            elif cn:
                single = len(cn) == 1
                kw.update(control_image=hint_imgs[0] if single else hint_imgs,
                          controlnet_conditioning_scale=cn[0].weight if single else [c.weight for c in cn],
                          control_guidance_start=starts[0] if single else starts,
                          control_guidance_end=ends[0] if single else ends)
            with dd:
                return to_tensor(np.asarray(pipe(**kw).images[0]), device)

        H, W = init.shape[2:]
        if job.tile and max(H, W) > job.tile:
            # Tiled: each tile denoised on its own at a size SD knows, blended back with feathered seams.
            # Every tile starts from the same init, so the warp's coherence carries through each one.
            acc, wsum = torch.zeros_like(init), torch.zeros_like(init[:, :1])
            th, tw = min(job.tile, H), min(job.tile, W)
            ys, xs = tile_spans(H, th, job.tile_overlap), tile_spans(W, tw, job.tile_overlap)
            for ti, (y, x) in enumerate((y, x) for y in ys for x in xs):
                sl = (slice(None), slice(None), slice(y, y + th), slice(x, x + tw))
                crops = [h.crop((x, y, x + tw, y + th)) for h in hints]
                gt = torch.Generator(device="cpu").manual_seed(int(job.seed) + i + 7919 * ti)
                t_out = denoise(init[sl], crops, None if amount is None else amount[sl], gt)
                wt = tile_weight(th, tw, y, x, H, W, job.tile_overlap, device)
                acc[sl] += t_out * wt
                wsum[sl] += wt
                del t_out, crops
                torch.cuda.empty_cache()        # each tile's scratch back before the next: full SDXL + CN is ~11 GB
            res = acc / wsum.clamp(min=1e-6)
            if i == 0:
                log(f"tiled: {len(ys) * len(xs)} tiles of {tw}x{th}, overlap {job.tile_overlap}")
        else:
            res = denoise(init, hints, amount, g)
        if first_out is None:
            first_out = res
        else:
            res = match_color(res, first_out, job.color_match)
        path = out / "frames" / f"{i:06d}.png"
        # written aside, then renamed: the viewer lists frames/*.png and must never get a half-written one
        to_image(res).save(path.with_suffix(".tmp"), format="PNG")
        os.replace(path.with_suffix(".tmp"), path)
        written.append(path)
        prev_src, prev_out = src, res
        # Hand this frame's scratch memory back every frame. Without it the caching allocator's
        # footprint crept up until the Windows driver spilled into system RAM (30 s -> 235 s/frame).
        del init, hints
        torch.cuda.empty_cache()
        log(f"frame {i + 1}/{total}: {time.time() - tf:.1f}s")
        progress("rendering", i + 1, total, f"frame {i + 1}/{total}")

    return written


def assemble(out_dir: str, fps: float) -> Path:
    """frames/ -> video.mp4 at `fps`; a run rendered on twos / threes (job.hold) holds each frame so the
    file plays at the clip's own rate and drops into its edit timeline."""
    out = Path(out_dir)
    target = out / "video.mp4"
    try:
        hold = max(1, int(json.loads((out / "job.json").read_text(encoding="utf-8")).get("hold", 1)))
    except (OSError, ValueError):
        hold = 1
    rate = ["-r", f"{fps * hold:g}"] if hold > 1 else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:g}", "-i", str(out / "frames" / "%06d.png"),
                    *rate, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                   check=True)
    return target
