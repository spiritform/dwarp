"""Post-process a finished run: frame interpolation (RIFE or FILM), then model upscale, streamed
straight into ffmpeg. Nothing is written to disk but the final mp4 (in <run>/post/).

Interpolation runs first, at render resolution, where it is cheap; every frame it
produces is then upscaled. One post job at a time — they share the GPU with renders.

A cloud upscale (start_cloud) is the same job in the same slot, with the work happening on Comfy
Cloud instead: the run's mp4 goes up, a bigger one comes back into the same post folder and shows
as the same chip. It holds no GPU here, so unlike the local one it can run while a render does.
"""
import glob
import hashlib
import os
import re
import shutil
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
# FILM (Google Research): slower than RIFE, steadier on large motion. TorchScript port by dajes,
# fetched on first use.
FILM_WEIGHTS = HERE / "models" / "film" / "film_net_fp32.pt"
FILM_URL = "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/film_net_fp32.pt"
FILM_SHA = "10aadc0d25ad586e3af7dd7703f75ec1186fbfe2e11c0d219d30ecfe2db10aa3"
INTERPS = ("rife", "film")
MODEL_EXTS = (".pth", ".safetensors", ".pt", ".ckpt")
TILE, OVERLAP = 512, 32

_jobs: dict = {}
_lock = threading.Lock()
_models: dict = {}                     # small cache: rife / film + last upscaler


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
          scale: int, slowmo: bool = False, interp: str = "rife") -> dict:
    """smooth: interpolation factor (interp: rife | film). The in-between frames raise the fps (same
    length, smoother), or with slowmo keep the run's fps, so the clip plays `smooth` times longer."""
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
        if smooth > 1 and interp == "film":
            parts.append("film")
        if slowmo:
            parts.append(f"slowmo{smooth}x")
        parts.append(f"{out_fps:g}fps")
        name = re.sub(r"[^\w.\-]+", "_", "_".join(parts))[:120] + ".mp4"
        job = {
            "id": uuid.uuid4().hex[:12], "run_id": run_dir.name, "state": "queued",
            "progress": 0.0, "message": "queued", "error": None, "output": name, "logs": [],
            "frames_in": len(frames), "frames_out": (len(frames) - 1) * smooth + 1,
            "started": time.time(), "cancel": False,
        }
        bits = ([f"upscale x{scale} with {Path(upscaler_path).stem}"] if upscaler_path else []) + \
               ([f"{interp.upper()} {smooth}x" + (" as slow motion" if slowmo else "")] if smooth > 1 else [])
        say(job, f"{' + '.join(bits)} on your card · {len(frames)} frames "
                 f"→ {(len(frames) - 1) * smooth + 1} at {out_fps:g} fps")
        _jobs[job["id"]] = job
    threading.Thread(target=_run, args=(job, frames, run_dir / "post" / name, out_fps, smooth,
                                        upscaler_path, scale, interp), daemon=True).start()
    return job


def start_cloud(run_dir: Path, wid: str, params: dict | None = None, *, batch: str = "warpbox") -> dict:
    """Send the run's own mp4 to a Comfy Cloud upscaler. Nothing here loads a model or takes the card."""
    from engine import cloud

    src = run_dir / "video.mp4"
    if not src.is_file():                        # a run the page can play always has one
        raise RuntimeError("This run has no mp4 to upscale")
    w = cloud.workflow(wid, kind="upscale")
    values = cloud.control_values(w, params)
    with _lock:
        if active_job():
            raise RuntimeError("A post-process job is already running")
        # only the short knobs belong in a file name: a prompt pasted into one is unreadable,
        # and the full settings are kept on the run this becomes
        short = {c["id"] for c in (w.get("controls") or [])
                 if c.get("kind") in ("number", "choice", "toggle")}
        bits = [w.get("short") or w["id"]] + [f"{k}{v}" for k, v in values.items() if k in short]
        name = re.sub(r"[^\w.\-]+", "_", "-".join(str(b) for b in bits))[:120] + ".mp4"
        job = {
            "id": uuid.uuid4().hex[:12], "run_id": run_dir.name, "state": "queued", "kind": "cloud",
            "progress": 0.0, "message": "queued", "error": None, "output": name, "logs": [],
            "workflow": w["id"], "label": w.get("label") or w["id"], "params": values,
            "started": time.time(), "cancel": False,
        }
        _jobs[job["id"]] = job
    threading.Thread(target=_guarded, args=(_run_cloud, job, run_dir, w["id"], values, name),
                     daemon=True).start()
    return job


def _rate(run_dir: Path) -> float:
    """The new run plays at the source's rate: an upscale changes the frame, never the clock."""
    import runs as rr
    try:
        return rr.fps(run_dir.name)
    except Exception:                            # noqa: BLE001 - a rate we cannot read is not fatal
        return 24.0


def _run_cloud(job, run_dir: Path, wid: str, values: dict, name: str):
    from engine import cloud

    job["state"] = "running"

    def note(message):
        """The poll line: it changes every second with the clock, so it is the status, not the log."""
        if job["cancel"]:
            raise InterruptedError
        say(job, message, keep=False)

    try:
        cloud.upscale_cloud(wid, run_dir / "video.mp4", run_dir / "post" / name, values,
                            progress=note, cancelled=lambda: job["cancel"],
                            log=lambda m: say(job, m))
    except InterruptedError:
        job["state"], job["message"] = "cancelled", "cancelled"
        return
    except cloud.CloudError as exc:
        if job["cancel"]:
            job["state"], job["message"] = "cancelled", "cancelled"
            return
        job["state"] = "failed"
        job["error"] = str(exc)
        say(job, str(exc))
        return
    # a post job reports 0-100 (the local path does); the cloud reports no percentage at all,
    # so it stays at 0 and the elapsed time in the message is the honest signal
    detail = " · ".join(f"{c.get('label') or c['id']} {values[c['id']]}"
                        for c in (cloud.workflow(wid, kind="upscale").get("controls") or [])
                        if c["id"] in values and c.get("kind") != "text")
    promote(job, run_dir / "post" / name, run_dir, "upscale", detail, _rate(run_dir))
    job["state"], job["progress"] = "completed", 100.0


# ------------------------------------------------------------------ models
def start_tracks(src: str, dest_dir: Path, *, points: int = 24, length: int = 81,
                 width: int = 720, height: int = 720, trim: list | None = None,
                 fps: float = 16.0) -> dict:
    """Track a video on the post queue. It is GPU work like an upscale, and it shares that queue so the
    two cannot collide; the page polls it at /api/post/{id} exactly as it polls an enhance."""
    with _lock:
        if active_job():
            raise RuntimeError("A post-process job is already running")
        job = {
            "id": uuid.uuid4().hex[:12], "run_id": "", "kind": "tracks", "state": "queued",
            "progress": 0.0, "message": "queued", "error": None, "output": Path(src).name,
            "frames_in": 0, "frames_out": int(length), "started": time.time(), "cancel": False,
            "result": None,
        }
        _jobs[job["id"]] = job
    threading.Thread(target=_guarded, daemon=True, args=(
        _run_tracks, job, src, Path(dest_dir), int(points), int(length), int(width), int(height),
        trim, float(fps))).start()
    return job


def promote(job: dict, made: Path, source: Path, verb: str, detail: str, rate: float) -> str:
    """Put a finished post video in the runs list as its own run.

    It gets its own frames, so it has a thumbnail, scrubs and compares like any other run, and a label
    saying what it came from — "upscale · wanmove" beside the "wanmove · render" it was made from.
    The source run's settings ride along, so "use settings" still gets you back to what made it.
    """
    import runs as rr
    from engine import cloud

    was = rr.read_meta(source.name)
    came = (was.get("label") or source.name).split(" · ")[0].strip() or source.name
    rid, path = rr.new_run()
    try:
        shutil.move(str(made), str(path / "video.mp4"))
        say(job, "extracting frames for the run")
        cloud.extract(path / "video.mp4", path / "frames")
        rr.write_meta(rid, label=f"{verb} · {came}", fps=float(rate),
                      ui={**(was.get("ui") or {}), "kind": verb, "source_run": source.name,
                          "post": detail})
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)    # a half-made run in the list is worse than none
        raise
    job["run"] = rid
    say(job, f"run {rid} · {verb} · {came}" + (f" · {detail}" if detail else ""))
    return rid


def say(job: dict, message: str, keep: bool = True) -> None:
    """One line of progress. It becomes the job's message, and — unless it is the per-frame counter,
    which would bury everything else — it is kept in the log the page shows."""
    message = str(message).rstrip()                # the right side only: the cloud log indents its
    if not message.strip():                        # detail lines, and that is the structure of it
        return
    job["message"] = message.strip()
    if keep:
        log = job.setdefault("logs", [])
        last = log[-1].split("] ", 1)[-1] if log else None    # past the timestamp this line will get
        if last != message:                        # a poll that repeats itself is not news
            log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
            del log[:-400]


def _guarded(fn, job, *args):
    """Nothing may leave a job sitting on "queued". A worker thread that dies before its own try --
    a bad import, the wrong arguments -- prints to the console and is never heard from again, and the
    page waits for a worker that is already gone. This catches the lot and marks the job failed."""
    try:
        fn(job, *args)
    except BaseException as exc:                 # noqa: BLE001 - the page shows whatever went wrong
        traceback.print_exc()
        job["state"] = "failed"
        job["error"] = job["message"] = f"{exc.__class__.__name__}: {exc}".strip(": ")
    if job.get("state") in (None, "queued", "running"):      # returned without saying how it went
        job["state"], job["message"] = "failed", "the worker stopped without finishing"


def _run_tracks(job, src, dest_dir, points, length, width, height, trim, fps):
    # everything inside the try, the import included: a thread that dies above it leaves the job on
    # "queued" for ever and the page waits on a worker that is already gone
    try:
        from engine import tracks as tk
        job["state"], job["message"] = "running", "reading the clip"

        def on(done, total):
            if job["cancel"]:
                raise InterruptedError
            job["progress"] = done / max(1, total)
            job["message"] = f"tracking {done}/{total}"

        job["result"] = tk.make(src, dest_dir, points=points, length=length, width=width,
                                height=height, trim=trim, fps=fps, on=on,
                                log=lambda m: job.__setitem__("message", m.strip()),
                                cancelled=lambda: job["cancel"])
        job["state"], job["progress"], job["message"] = "done", 1.0, "done"
    except InterruptedError:
        job["state"], job["message"] = "cancelled", "cancelled"
    except BaseException as exc:                 # noqa: BLE001 - the page shows whatever went wrong
        traceback.print_exc()
        job["state"] = "cancelled" if job["cancel"] else "failed"
        job["error"] = job["message"] = f"{exc.__class__.__name__}: {exc}".strip(': ')


def _rife():
    if "rife" not in _models:
        from vendor.rife_arch import IFNet
        net = IFNet(arch_ver=RIFE_ARCH)
        net.load_state_dict(torch.load(RIFE_WEIGHTS, map_location="cpu", weights_only=True))
        _models["rife"] = net.eval().cuda()
    return _models["rife"]


def _film(job=None):
    if "film" not in _models:
        if not FILM_WEIGHTS.is_file():
            _fetch_film(job)
        _models["film"] = torch.jit.load(str(FILM_WEIGHTS), map_location="cpu").eval().cuda()
    return _models["film"]


def _fetch_film(job=None):
    import urllib.request
    FILM_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    part, digest, got = FILM_WEIGHTS.with_suffix(".part"), hashlib.sha256(), 0
    with urllib.request.urlopen(FILM_URL) as r, open(part, "wb") as fh:
        size = int(r.headers.get("Content-Length") or 0)
        while chunk := r.read(1 << 20):
            fh.write(chunk); digest.update(chunk); got += len(chunk)
            if job is not None:
                job["message"] = f"downloading FILM weights · {got >> 20} / {size >> 20} MB"
    if digest.hexdigest() != FILM_SHA:
        part.unlink(missing_ok=True)
        raise RuntimeError("FILM weights download did not match its checksum")
    os.replace(part, FILM_WEIGHTS)


def _upscaler(path: str):
    key = ("up", path)
    if key not in _models:
        from spandrel import ModelLoader
        for k in [k for k in _models if k not in INTERPS]:   # keep one upscaler resident at most
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
def _inbetweens(a: torch.Tensor, b: torch.Tensor, n: int, interp: str) -> list:
    """The n - 1 frames between a and b, in order. RIFE takes each time directly. FILM does best near
    the middle of its pair, so each new frame comes from the two closest frames already made (the
    order ComfyUI-Frame-Interpolation uses): 2x = one midpoint, 4x = midpoint then quarters."""
    if interp != "film":
        return [_interp(a, b, k / n) for k in range(1, n)]
    film = _film()
    made = {0: a, n: b}                    # known frames by index 0..n
    todo = list(range(1, n))
    while todo:
        keys = sorted(made)
        best = None                        # (distance from its pair's middle, k, left, right)
        for k in todo:
            lo = max(i for i in keys if i < k); hi = min(i for i in keys if i > k)
            d = abs((k - lo) / (hi - lo) - 0.5)
            if best is None or d < best[0]:
                best = (d, k, lo, hi)
        _, k, lo, hi = best
        dt = made[lo].new_full((1, 1), (k - lo) / (hi - lo))
        made[k] = film(made[lo], made[hi], dt).clamp(0, 1)
        todo.remove(k)
    return [made[k] for k in range(1, n)]


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


def _run(job, frames, out_path: Path, fps, smooth, upscaler_path, scale, interp="rife"):
    proc = None
    try:
        job["state"], job["message"] = "running", "loading models"
        first = _load(frames[0])
        if smooth > 1:
            _film(job) if interp == "film" else _rife()
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
            # the per-frame counter is the status line's job, not the log's
            say(job, f"frame {done}/{total} · {per:.2f} s/f · {max(0, total - done) * per:.0f}s left", keep=False)

        prev = first
        emit(prev)
        for path in frames[1:]:
            cur = _load(path)
            for mid in _inbetweens(prev, cur, smooth, interp):
                emit(mid)
            emit(cur)
            prev = cur

        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()[-500:]}")
        os.replace(tmp, out_path)
        say(job, f"made {total} frames in {time.time() - t0:.0f}s")
        bits = ([f"{scale}x {Path(upscaler_path).stem}"] if upscaler_path else []) + \
               ([f"{interp} {smooth}x"] if smooth > 1 else [])
        promote(job, out_path, out_path.parent.parent, "upscale" if upscaler_path else "smooth",
                " · ".join(bits), fps)
        job["state"], job["progress"] = "completed", 100.0
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
