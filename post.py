"""Post-process a finished run: RIFE frame interpolation, then model upscale, streamed
straight into ffmpeg. Nothing is written to disk but the final mp4 (in <run>/post/).

Interpolation runs first, at render resolution, where RIFE is cheap; every frame it
produces is then upscaled. One post job at a time — they share the GPU with renders.
"""
import glob
import os
import re
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
RIFE_WEIGHTS = HERE / "models" / "rife" / "rife49.pth"
RIFE_ARCH = "4.7"                      # rife47/rife49 share the 4.7 architecture
MODEL_EXTS = (".pth", ".safetensors", ".pt", ".ckpt")
TILE, OVERLAP = 512, 32

_jobs: dict = {}
_lock = threading.Lock()
_models: dict = {}                     # small cache: rife + last upscaler


def list_upscalers(model_dir: str) -> list:
    if not model_dir or not os.path.isdir(model_dir):
        return []
    # LTX "spatial upscalers" live in the same Comfy folder but are latent-space
    # video-model parts, not image upscalers.
    return sorted(name for name in os.listdir(model_dir)
                  if name.lower().endswith(MODEL_EXTS) and "ltx" not in name.lower()
                  and os.path.isfile(os.path.join(model_dir, name)))


def active_job():
    return next((j for j in _jobs.values() if j["state"] in ("queued", "running")), None)


def get_job(job_id: str):
    return _jobs.get(job_id)


def cancel(job_id: str):
    job = _jobs.get(job_id)
    if job:
        job["cancel"] = True
    return job


def run_frames(run_dir: Path, batch: str = "warpbox") -> list:
    """Output frames of a run in frame order. Engine runs keep them in frames/NNNNNN.png;
    legacy VibeWarp runs name them <batch>(<n>)_<frame>.png (<n> is not the run number)."""
    if (run_dir / "frames").is_dir():
        return sorted(str(p) for p in (run_dir / "frames").glob("*.png"))
    pat = re.compile("^" + re.escape(batch) + r"\(\d+\)_(\d+)\.png$")
    found = []
    for path in glob.glob(str(run_dir / "*.png")):
        m = pat.search(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return [p for _, p in sorted(found)]


def start(run_dir: Path, batch: str, *, fps: float, smooth: int, upscaler_path: str | None,
          scale: int, slowmo: bool = False) -> dict:
    """smooth: RIFE factor. The in-between frames raise the fps (same length, smoother), or with
    slowmo keep the run's fps, so the clip plays `smooth` times longer."""
    with _lock:
        if active_job():
            raise RuntimeError("A post-process job is already running")
        frames = run_frames(run_dir, batch)
        if len(frames) < 2:
            raise RuntimeError("This run has fewer than 2 frames")
        parts = []
        if upscaler_path:
            parts.append(f"x{scale}-{Path(upscaler_path).stem}")
        slowmo = slowmo and smooth > 1
        out_fps = fps if slowmo else fps * smooth
        if slowmo:
            parts.append(f"slowmo{smooth}x")
        parts.append(f"{out_fps:g}fps")
        name = re.sub(r"[^\w.\-]+", "_", "_".join(parts))[:120] + ".mp4"
        job = {
            "id": uuid.uuid4().hex[:12], "run_id": run_dir.name, "state": "queued",
            "progress": 0.0, "message": "queued", "error": None, "output": name,
            "frames_in": len(frames), "frames_out": (len(frames) - 1) * smooth + 1,
            "started": time.time(), "cancel": False,
        }
        _jobs[job["id"]] = job
    threading.Thread(target=_run, args=(job, frames, run_dir / "post" / name, out_fps, smooth,
                                        upscaler_path, scale), daemon=True).start()
    return job


# ------------------------------------------------------------------ models
def _rife():
    if "rife" not in _models:
        from vendor.rife_arch import IFNet
        net = IFNet(arch_ver=RIFE_ARCH)
        net.load_state_dict(torch.load(RIFE_WEIGHTS, map_location="cpu", weights_only=True))
        _models["rife"] = net.eval().cuda()
    return _models["rife"]


def _upscaler(path: str):
    key = ("up", path)
    if key not in _models:
        from spandrel import ModelLoader
        for k in [k for k in _models if k != "rife"]:   # keep one upscaler resident at most
            del _models[k]
        torch.cuda.empty_cache()
        desc = ModelLoader(device="cuda").load_from_file(path)
        desc.eval()
        half = bool(getattr(desc, "supports_half", False))
        if half:
            desc.model.half()
        _models[key] = (desc, half)
    return _models[key]


@torch.inference_mode()
def _interp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return _rife()(a, b, timestep=t, scale_list=[8, 4, 2, 1], training=False,
                   fastmode=True, ensemble=False).clamp(0, 1)


@torch.inference_mode()
def _upscale(x: torch.Tensor, path: str, scale: int) -> torch.Tensor:
    """Tiled model upscale, then resize to exactly `scale`x if the model's factor differs."""
    desc, half = _upscaler(path)
    _, _, h, w = x.shape
    s = desc.scale
    inp = x.half() if half else x
    if h <= TILE and w <= TILE:
        out = desc(inp).float()
    else:
        out = torch.zeros((1, 3, h * s, w * s), device=x.device)
        weight = torch.zeros_like(out)
        step = TILE - OVERLAP
        for y0 in range(0, max(h - OVERLAP, 1), step):
            for x0 in range(0, max(w - OVERLAP, 1), step):
                y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                y0c, x0c = max(0, y1 - TILE), max(0, x1 - TILE)
                tile = desc(inp[:, :, y0c:y1, x0c:x1]).float()
                out[:, :, y0c * s:y1 * s, x0c * s:x1 * s] += tile
                weight[:, :, y0c * s:y1 * s, x0c * s:x1 * s] += 1
        out = out / weight
    if s != scale:
        out = F.interpolate(out, size=(h * scale, w * scale), mode="bicubic", antialias=True)
    return out.clamp(0, 1)


# ------------------------------------------------------------------ worker
def _load(path: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()


def _run(job, frames, out_path: Path, fps, smooth, upscaler_path, scale):
    proc = None
    try:
        job["state"], job["message"] = "running", "loading models"
        first = _load(frames[0])
        if smooth > 1:
            _rife()
        factor = scale if upscaler_path else 1
        h, w = first.shape[2] * factor, first.shape[3] * factor
        w2, h2 = w - w % 2, h - h % 2                  # yuv420p wants even sizes
        out_path.parent.mkdir(exist_ok=True)
        tmp = out_path.with_suffix(".part.mp4")
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}",
             "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        done, total = 0, job["frames_out"]
        t0 = time.time()

        def emit(frame: torch.Tensor):
            nonlocal done
            if job["cancel"]:
                raise InterruptedError
            if upscaler_path:
                frame = _upscale(frame, upscaler_path, scale)
            frame = frame[:, :, :h2, :w2]
            buf = (frame[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
            proc.stdin.write(buf.tobytes())
            done += 1
            job["progress"] = done / total * 100
            per = (time.time() - t0) / done
            job["message"] = f"frame {done}/{total} · {per:.2f} s/f · {max(0, total - done) * per:.0f}s left"

        prev = first
        emit(prev)
        for path in frames[1:]:
            cur = _load(path)
            for k in range(1, smooth):
                emit(_interp(prev, cur, k / smooth))
            emit(cur)
            prev = cur

        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()[-500:]}")
        os.replace(tmp, out_path)
        job["state"], job["progress"] = "completed", 100.0
        job["message"] = f"done · {total} frames · {time.time() - t0:.0f}s"
    except InterruptedError:
        job["state"], job["message"] = "cancelled", "cancelled"
    except Exception as exc:
        job["state"], job["error"] = "failed", f"{exc}"
        job["message"] = "failed"
        job["traceback"] = traceback.format_exc()
    finally:
        if proc and proc.poll() is None:
            try:
                proc.stdin.close()
            except OSError:
                pass
            proc.kill()
        for leftover in out_path.parent.glob("*.part.mp4") if out_path.parent.exists() else []:
            try:
                leftover.unlink()
            except OSError:
                pass
        torch.cuda.empty_cache()
