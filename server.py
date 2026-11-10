"""DWARP server: the page, the render queue, run history and post-processing.

    .venv\\Scripts\\python server.py [--port 8013] [--no-browser]
"""
from __future__ import annotations

# The console window is empty until uvicorn binds, and a silent black window reads as a hang --
# which is exactly what it looked like while torch was being imported at start-up. Say something.
if __name__ == "__main__":
    print("DWARP starting…", flush=True)

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import uuid
import webbrowser
from pathlib import Path
from threading import Timer

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import modelscan  # noqa: E402
import post  # noqa: E402
import runs  # noqa: E402
import shapelib  # noqa: E402
import downloads  # noqa: E402
from jobs import TERMINAL, JobManager  # noqa: E402
from models import list_checkpoints  # noqa: E402
from engine import cloud  # noqa: E402
from engine.cloud import api_key as h3_key  # noqa: E402


# Long enough for the page's own start-up questions to get the disk first. Warming torch straight
# away made the first load slower than it needed to be: reading a few hundred model filenames off a
# spinning disk and loading torch's DLLs off the same one are not friends.
WARM_AFTER = 20.0


def warm() -> None:
    """Pull torch in while the server is already answering.

    post.py leaves torch until something actually needs a tensor (see post._LazyTorch), which is
    what lets the port bind in seconds instead of minutes. That only moves the cost, so it is spent
    here, on a background thread, against the odds that you enhance something before it lands.
    Failures are silent: this is a head start, not a requirement.
    """
    try:
        import torch        # noqa: F401
    except Exception:
        pass


INPUTS = HERE / "inputs"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}          # Image -> Video's picture
jobs = JobManager()


def config() -> dict:
    """Machine-specific settings (config.json, not committed)."""
    f = HERE / "config.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def settings() -> dict:
    """presets.json with model file names resolved against config.json's models_root."""
    d = json.loads((HERE / "presets.json").read_text(encoding="utf-8"))
    root = (config().get("models_root") or str(HERE / "models")).rstrip("/\\")
    d["models_root"] = root
    d["checkpoint_dir"] = f"{root}/checkpoints"
    d["upscale_dir"] = f"{root}/upscale_models"
    for mode in d.get("modes", {}).values():
        sr = mode.get("style_ref")
        for c in [sr, sr["encoder"]] if sr else []:   # IP-Adapter + its image encoder: missing ones download on first use
            paths = [str(HERE / "models" / c["dir"] / n) for n in c["file"]] + [f"{root}/{c['dir']}/{n}" for n in c["file"]]
            c["path"] = next((q for q in paths if os.path.isfile(q)), str(HERE / "models" / c["dir"] / c.get("save_as", c["file"][0])))
            c["have"] = os.path.isfile(c["path"])
        refine_cn = [mode["refine"]["controlnet"]] if mode.get("refine", {}).get("controlnet") else []
        illusion = [mode["illusion"]] if mode.get("illusion") else []   # QR Code Monster, for the Shape card
        for c in list(mode.get("controlnets", {}).values()) + refine_cn + illusion:
            names = c["file"] if isinstance(c["file"], list) else [c["file"]]
            # DWARP's own models/controlnet first (e.g. the SD 2.1 nets), then the models folder's
            paths = [str(HERE / "models" / "controlnet" / n) for n in names] + [f"{root}/controlnet/{n}" for n in names]
            c["path"] = next((q for q in paths if os.path.isfile(q)), paths[0])   # missing: DWARP's folder
            c["have"] = os.path.isfile(c["path"])
            c["fetch"] = 0 if c["have"] else downloads.size(c["path"])   # downloads on first use: this many bytes
    d["fetchable"] = {name: k[2] for name, k in downloads.KNOWN.items()}   # checkpoints etc. the panel can promise
    return d


def run_or_404(run_id: str) -> Path:
    try:
        return runs.run_dir(run_id)
    except runs.RunNotFound:
        raise HTTPException(404, detail="Unknown run")


# ------------------------------------------------------------------ video helpers
def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,codec_name:format=duration",
                          "-of", "json", path], capture_output=True, text=True)
    if out.returncode != 0:
        raise ValueError(out.stderr.strip()[:300] or "ffprobe failed")
    data = json.loads(out.stdout)
    st = (data.get("streams") or [{}])[0]
    num, _, den = (st.get("r_frame_rate") or "24/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 24.0
    duration = float((data.get("format") or {}).get("duration") or 0)
    frames = int(st.get("nb_read_packets") or round(duration * fps))
    if not duration or fps > 240:          # e.g. a browser-recorded WebM: no duration, "1000 fps" = its clock
        fps, duration = _measured_rate(path, frames) or (fps, duration)
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": fps, "frames": frames,
            "duration": duration, "codec": st.get("codec_name", "")}


def _measured_rate(path: str, frames: int) -> tuple[float, float] | None:
    """fps and duration from the frames' own timestamps, for files whose header doesn't say."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", path], capture_output=True, text=True)
    times = sorted(float(t) for t in (line.strip().rstrip(",") for line in out.stdout.splitlines())
                   if t and t != "N/A")
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    fps = round((len(times) - 1) / (times[-1] - times[0]), 3)
    return fps, round(max(frames, len(times)) / fps, 3)


def fit(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Longest side -> max_size, aspect kept, both multiples of 8."""
    scale = max_size / max(width, height)
    return max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8)


def build_app() -> FastAPI:
    runs.RENDERS.mkdir(exist_ok=True)
    INPUTS.mkdir(exist_ok=True)
    app = FastAPI(title="DWARP")

    # -------------------------------------------------------------- page
    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/fonts/{name}")
    def font(name: str):
        """The wordmark's font, bundled so the page works offline (vendor/fonts, OFL)."""
        f = HERE / "vendor" / "fonts" / Path(name).name
        if f.suffix != ".woff2" or not f.is_file():
            raise HTTPException(404, detail="No such font")
        return FileResponse(f, media_type="font/woff2", headers={"Cache-Control": "max-age=86400"})

    @app.get("/vendor/{name}")
    def vendor_js(name: str):
        """Bundled page scripts (anime.js, MIT), so the page works offline."""
        f = HERE / "vendor" / Path(name).name
        if f.suffix != ".js" or not f.is_file():
            raise HTTPException(404, detail="No such script")
        return FileResponse(f, media_type="text/javascript", headers={"Cache-Control": "max-age=86400"})

    @app.get("/presets.json")
    def presets():
        return JSONResponse(settings(), headers={"Cache-Control": "no-store"})

    @app.get("/api/info")
    def info():
        return {"output_dir": str(runs.RENDERS), "batch_name": "warpbox", "engine": "dwarp"}

    @app.get("/api/fs/exists")
    def exists(path: str = ""):
        p = Path(path.strip().strip('"')) if path.strip() else None
        return {"path": path, "checked": bool(p), "exists": bool(p and p.exists())}

    def embedding_dirs() -> list[str]:
        """Textual-inversion embeddings: the folder Settings names, else DWARP's own models/embeddings
        (a hand-picked set — a shared ComfyUI folder holds a lot)."""
        return [config().get("embeddings_dir") or str(HERE / "models" / "embeddings")]

    def lora_dirs() -> list[Path]:
        """LoRAs: the folder Settings names, else DWARP's own models/loras."""
        return [Path(config().get("loras_dir") or HERE / "models" / "loras")]

    # -------------------------------------------------------------- settings (config.json, re-read per request)
    @app.get("/api/comfy/workflows")
    def comfy_workflows():
        """The Comfy Cloud graphs this install has, and what each one takes. The panel builds itself from
        this — no references slot in a manifest means no References card — so dropping a graph into
        engine/comfy/ and naming it in workflows.json is the whole job of adding one."""
        try:
            # "hidden" parks a graph without deleting it: out of the picker, still in the
            # manifest and still renderable by id, so a direct submit can keep testing it.
            return {"workflows": [cloud.caps(w) for w in cloud.workflows("render")
                                  if not w.get("hidden")]}
        except cloud.CloudError as e:
            return {"workflows": [], "error": str(e)}

    @app.get("/api/config")
    def get_config(count: bool = False):
        """`count` walks the checkpoints folder to total them. Settings asks for that; start-up does
        not, and it used to pay for the walk anyway on a call it makes before the page can paint."""
        c, root = config(), settings()["models_root"]
        return {"models_root": root, "own_models": (HERE / "models").as_posix(),
                "embeddings_dir": c.get("embeddings_dir", ""), "loras_dir": c.get("loras_dir", ""),
                "checkpoints": len(list_checkpoints(f"{root}/checkpoints")) if count else None,
                "root_ok": os.path.isdir(root),
                # H3 renders on Comfy Cloud. Whether a key is set, never the key: it is a live
                # credential, so it goes into config.json (gitignored) and is never read back out.
                "comfy_key": bool(h3_key(c)), "comfy_key_env": bool(h3_key({}))}

    @app.post("/api/config")
    async def set_config(request: Request):
        body = await request.json()
        c = config()
        if "models_root" in body:
            root = str(body["models_root"]).strip().strip('"').replace("\\", "/").rstrip("/")
            if not root or not os.path.isdir(root):
                raise HTTPException(422, detail=[{"message": f"No folder at {root!r}"}])
            c["models_root"] = root
        for k in ("embeddings_dir", "loras_dir"):   # blank = DWARP's own folder
            if k in body:
                d = str(body[k]).strip().strip('"').replace("\\", "/").rstrip("/")
                if d and not os.path.isdir(d):
                    raise HTTPException(422, detail=[{"message": f"No folder at {d!r}"}])
                c[k] = d
        if "comfy_api_key" in body:                 # blank clears it
            key = str(body["comfy_api_key"]).strip().strip('"')
            if key:
                c["comfy_api_key"] = key
            else:
                c.pop("comfy_api_key", None)
        for k in ("extra_embeddings", "extra_loras"):   # the old "also use the models folder's" switches
            c.pop(k, None)
        (HERE / "config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
        return get_config()

    @app.post("/api/open-models")
    def open_models(which: str = "own"):
        """Opens a Settings folder in Explorer: own (DWARP's models), root, embeddings or loras."""
        folder = {"root": settings()["models_root"], "embeddings": embedding_dirs()[0],
                  "loras": str(lora_dirs()[0]), "renders": str(runs.RENDERS)}.get(which, str(HERE / "models"))
        if which == "own":
            (HERE / "models").mkdir(exist_ok=True)
        if not os.path.isdir(folder):
            raise HTTPException(404, detail=f"No folder at {folder}")
        os.startfile(folder)  # Windows-only; this is a local tool
        return {"ok": True}

    @app.get("/api/loras")
    def loras():
        """LoRAs from DWARP's own models/loras (a hand-picked set), each with the family it was trained for;
        ones for other model types (Flux, video…) are left out."""
        seen, out = set(), []
        for folder in lora_dirs():
            for f in modelscan.lora_files(folder):
                if f.stem not in seen and (fam := modelscan.lora_family(f)):
                    seen.add(f.stem)
                    out.append({"name": f.stem, "path": f.as_posix(), "family": fam})
        return {"loras": sorted(out, key=lambda e: e["name"].lower())}

    @app.get("/api/embeddings")
    def embeddings():
        """Embeddings by name (what a prompt types), with the model family they were trained for."""
        out = [{"name": name, "family": fam} for name, f in modelscan.embedding_files(embedding_dirs()).items()
               if (fam := modelscan.embedding_family(f))]
        return {"embeddings": sorted(out, key=lambda e: e["name"].lower())}

    _sys = {"t": 0.0, "v": None}

    @app.get("/api/system")
    def system():
        """The header's GPU meter: VRAM, load, temperature, power (nvidia-smi), plus CPU and RAM. Cached a second."""
        import time as _t
        if _t.time() - _sys["t"] < 1.0 and _sys["v"]:
            return _sys["v"]
        out = {"gpu": None}
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,"
                                "power.draw,power.limit,power.default_limit", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=3)
            f = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
            num = lambda x: float(x) if x.replace(".", "", 1).isdigit() else None
            out["gpu"] = {"name": f[0], "vram_used": num(f[1]), "vram_total": num(f[2]), "load": num(f[3]),
                          "temp": num(f[4]), "power": num(f[5]), "power_limit": num(f[6]), "power_default": num(f[7])}
        except (OSError, IndexError, subprocess.SubprocessError):
            pass
        try:
            import psutil
            vm = psutil.virtual_memory()
            out["cpu"] = psutil.cpu_percent(interval=None)
            out["ram_used"], out["ram_total"] = vm.used / 2 ** 30, vm.total / 2 ** 30
        except Exception:
            pass
        _sys.update(t=_t.time(), v=out)
        return out

    @app.get("/api/motion")
    def motion():
        """Motion (AnimateDiff): motion modules and motion LoRAs, the models folder's (ComfyUI layout) and DWARP's own."""
        root = Path(settings()["models_root"])
        exts = (".safetensors", ".ckpt", ".pth")

        def scan(*dirs):
            found = {}
            for d in dirs:
                if d.is_dir():
                    for f in sorted(d.iterdir()):
                        if f.suffix.lower() in exts and f.name not in found:
                            found[f.name] = {"name": f.stem, "path": f.as_posix()}
            return list(found.values())
        # SD 1.5 motion modules only: not SDXL ones, the v3 adapter (a LoRA) or SparseCtrl. The page picks from
        # these by name (TemporalDiff, v2, AnimateLCM); AnimateLCM also needs its spatial LoRA.
        skip = re.compile(r"sdxl|adapter|sparsectrl", re.I)
        modules = [m for m in scan(HERE / "models" / "animatediff", root / "animatediff_models")
                   if not skip.search(m["name"])]
        for m in modules:
            m["fetch"] = 0
        have = {Path(m["path"]).name for m in modules}
        for name in ("temporaldiff-v1-animatediff.safetensors", "mm_sd_v15_v2.ckpt", "AnimateLCM_sd15_t2v.ckpt"):
            if name not in have:                  # not on disk: listed anyway, downloads on first use
                modules.append({"name": Path(name).stem, "path": (HERE / "models" / "animatediff" / name).as_posix(),
                                "fetch": downloads.size(name)})
        lcm = modelscan.animatelcm_lora(str(root / "animatediff_models" / "x"))
        return {"modules": modules,
                "loras": scan(HERE / "models" / "animatediff_motion_lora", root / "animatediff_motion_lora"),
                "lcm_lora": lcm or str(HERE / "models" / "loras" / "AnimateLCM_sd15_t2v_lora.safetensors"),
                "lcm_lora_fetch": 0 if lcm else downloads.size("AnimateLCM_sd15_t2v_lora.safetensors")}

    @app.get("/api/checkpoints")
    def checkpoints():
        root = settings().get("checkpoint_dir", "")
        return {"root": root, "checkpoints": list_checkpoints(root)}

    # -------------------------------------------------------------- input video
    @app.post("/api/upload")
    async def upload(request: Request, filename: str = "clip.mp4"):
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTS | IMAGE_EXTS:
            raise HTTPException(422, detail=f"Not a video or image file: {filename}")
        stem = re.sub(r"[^\w\-]+", "_", Path(filename).stem)[:60] or "clip"
        target = INPUTS / f"{stem}-{uuid.uuid4().hex[:6]}{suffix}"
        size = 0
        with open(target, "wb") as fh:
            async for chunk in request.stream():
                fh.write(chunk)
                size += len(chunk)
        if size == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(422, detail="Empty upload")
        if suffix in IMAGE_EXTS:                 # a picture: its size sets Image -> Video's aspect
            from PIL import Image
            try:
                with Image.open(target) as im:
                    w, h = im.size
            except Exception:
                target.unlink(missing_ok=True)
                raise HTTPException(422, detail=f"Could not read that image: {filename}")
            return {"path": str(target), "name": filename, "bytes": size, "image": True, "width": w, "height": h}
        return {"path": str(target), "name": filename, "bytes": size}

    # -------------------------------------------------------------- the Shape library (shapelib.py)
    @app.get("/api/shapes")
    def shapes_list():
        """Presets (drawn on first use: ready is false until they're on disk) and the user's own masks."""
        return shapelib.listing()

    @app.post("/api/shapes/upload")
    async def shapes_upload(request: Request, filename: str = "mask.png"):
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTS | IMAGE_EXTS:
            raise HTTPException(422, detail=f"Not a video or image file: {filename}")
        shapelib.YOURS.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w\- ]+", "_", Path(filename).stem)[:60].strip() or "mask"
        target, n = shapelib.YOURS / f"{stem}{suffix}", 2
        while target.exists():                   # keep both: "name 2.png"
            target, n = shapelib.YOURS / f"{stem} {n}{suffix}", n + 1
        size = 0
        with open(target, "wb") as fh:
            async for chunk in request.stream():
                fh.write(chunk)
                size += len(chunk)
        if size == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(422, detail="Empty upload")
        return {"path": str(target), "name": target.stem, "video": suffix in VIDEO_EXTS}

    @app.delete("/api/shapes")
    def shapes_delete(path: str):
        """Only the user's own masks (shapes/yours) can be deleted; presets come back anyway."""
        p = Path(path).resolve()
        if p.parent != shapelib.YOURS.resolve() or not p.is_file():
            raise HTTPException(404, detail="Not one of your shapes")
        p.unlink()
        return {"deleted": str(p)}

    @app.post("/api/shapes/open")
    def shapes_open():
        shapelib.YOURS.mkdir(parents=True, exist_ok=True)
        os.startfile(shapelib.YOURS)  # Windows-only; this is a local tool
        return {"ok": True}

    @app.get("/api/image/file")
    def image_file(path: str = ""):
        """Image -> Video's picture, for the panel and the stage."""
        if not path or not os.path.isfile(path) or Path(path).suffix.lower() not in IMAGE_EXTS:
            raise HTTPException(404, detail="No image at that path")
        return FileResponse(path)

    @app.get("/api/video/probe")
    def video_probe(path: str = "", max_size: int = 0, extract_nth_frame: int = 1):
        if not path or not os.path.isfile(path):
            raise HTTPException(404, detail=[{"message": "No video at that path"}])
        try:
            info = probe(path)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, detail=[{"message": f"Could not read that video: {exc}"}])
        w, h = fit(info["width"], info["height"], max_size) if max_size > 0 else (info["width"], info["height"])
        nth = max(1, extract_nth_frame)
        renderable = -(-info["frames"] // nth)
        return {"source": info, "render": {"width": w, "height": h}, "extract_nth_frame": nth,
                "renderable_frames": renderable, "max_frame": max(0, renderable - 1)}

    @app.get("/api/video/file")
    def video_file(path: str = ""):
        """The clip itself, for playing the source in clip view (range requests, so it can seek)."""
        if not path or not os.path.isfile(path) or Path(path).suffix.lower() not in VIDEO_EXTS:
            raise HTTPException(404, detail="No video at that path")
        return FileResponse(path)

    @app.get("/api/video/thumbnail")
    def video_thumbnail(path: str = "", frame: int = 0):
        if not path or not os.path.isfile(path):
            raise HTTPException(404, detail="No video at that path")
        vf = f"select=eq(n\\,{max(0, frame)}),scale=480:480:force_original_aspect_ratio=decrease"
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-frames:v", "1",
                              "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"], capture_output=True)
        if out.returncode != 0 or not out.stdout:
            raise HTTPException(422, detail="Could not read that frame")
        return StreamingResponse(iter([out.stdout]), media_type="image/jpeg",
                                 headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/diff/base")
    def diff_base(path: str = "", frame: int = 0, width: int = 512, height: int = 512, kind: str = "luma",
                  run: str = "", depth_model: str = "midas", fit: str = "stretch"):
        """One source frame, as the DepthDiff mask starts from it: the frame itself (luma) or its
        depth map. Either a clip frame at render size (path + frame + size; also the stage's clip
        view), or the frame a run started from (run + that run's frame number). JPEG like the
        engine's own source frames; depth as PNG. The page shapes it into the mask live."""
        if run:
            run_or_404(run)
            src = runs.layer_path(run, "init", frame)
            if not src:
                raise HTTPException(404, detail="No source frame for that run/frame")
            data = src.read_bytes()
        else:
            if not path or not os.path.isfile(path):
                raise HTTPException(404, detail="No video at that path")
            from engine.warp import scale_vf
            vf = f"select=eq(n\\,{max(0, frame)})," + scale_vf(max(8, width), max(8, height), fit)
            out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-frames:v", "1",
                                  "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"], capture_output=True)
            if out.returncode != 0 or not out.stdout:
                raise HTTPException(422, detail="Could not read that frame")
            data = out.stdout
        if kind != "depth":
            return StreamingResponse(iter([data]), media_type="image/jpeg", headers={"Cache-Control": "max-age=600"})
        if kind == "depth":
            import io
            import numpy as np
            from PIL import Image
            from engine.warp import depth_map
            buf = io.BytesIO()
            depth_map(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), depth_model).convert("RGB").save(buf, "PNG")
            data = buf.getvalue()
        return StreamingResponse(iter([data]), media_type="image/png", headers={"Cache-Control": "max-age=600"})

    # -------------------------------------------------------------- jobs
    @app.post("/api/jobs", status_code=202)
    async def submit(request: Request):
        body = await request.json()
        job, meta = body.get("job") or {}, body.get("meta") or {}
        job["embeddings_dir"] = embedding_dirs()   # names typed in a prompt load from here
        if job.get("lora") and not os.path.isfile(job["lora"]):
            raise HTTPException(422, detail=[{"message": f"LoRA not found: {job['lora']!r}"}])
        if job.get("family") in ("comfy", "h3"):
            # H3 renders on Comfy Cloud: no local checkpoint, and the clip is a reference (h3_video),
            # not the thing being repainted. Either a clip or a picture is enough.
            for key in ("h3_video", *[f"h3_refs[{i}]" for i in range(len(job.get("h3_refs") or []))]):
                path = job.get("h3_video") if key == "h3_video" else job["h3_refs"][int(key[8:-1])]
                if path and not os.path.isfile(path):
                    raise HTTPException(422, detail=[{"message": f"{'clip' if key == 'h3_video' else 'reference'} not found: {path!r}"}])
            if job.get("comfy_tracks") and not os.path.isfile(job["comfy_tracks"]):
                raise HTTPException(422, detail=[{"message": f"tracks not found: {job['comfy_tracks']!r}"}])
            if not job.get("h3_video") and not (job.get("h3_refs") or []):
                raise HTTPException(422, detail=[{"message": "this workflow needs a clip, or a reference picture"}])
        else:
            for key in (("checkpoint",) + (("init_image",) if job.get("init_image") else ()) if job.get("t2v_frames")
                        else ("video", "checkpoint")):   # text / image -> video: no clip
                if not job.get(key) or not (os.path.isfile(job[key]) or key == "video" and os.path.isdir(job[key])
                                            or key == "checkpoint" and downloads.missing(job[key])):
                    raise HTTPException(422, detail=[{"message": f"{key} not found: {job.get(key)!r}"}])
        for key in ("motion", "motion_lora", "shape_qr_cn"):    # Motion's module / LoRA, QR Code Monster
            if job.get(key) and not os.path.isfile(job[key]) and not downloads.missing(job[key]):
                raise HTTPException(422, detail=[{"message": f"{key} not found: {job[key]!r}"}])
        if job.get("style_image") and not os.path.isfile(job["style_image"]):
            raise HTTPException(422, detail=[{"message": f"style image not found: {job['style_image']!r}"}])
        if job.get("shape") and not os.path.exists(job["shape"]):
            raise HTTPException(422, detail=[{"message": f"shape mask not found: {job['shape']!r}"}])
        for c in job.get("controlnets", []):
            if not os.path.isfile(c.get("path", "")) and not c.get("repo") and not downloads.missing(c.get("path", "")):   # downloads on first use
                raise HTTPException(422, detail=[{"message": f"ControlNet not found: {c.get('path')!r}"}])
        run_id, _ = runs.new_run()
        runs.write_meta(run_id, label=meta.get("label", ""), ui=meta.get("ui", {}), fps=meta.get("fps", 24))
        return jobs.submit("render", run_id, {"job": job, "fps": meta.get("fps", 24)}).snapshot()

    @app.get("/api/jobs")
    def list_jobs():
        return jobs.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, detail="Unknown job")
        return job.snapshot()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = jobs.cancel(job_id)
        if not job:
            raise HTTPException(404, detail="Unknown job")
        return job.snapshot()

    @app.post("/api/jobs/{job_id}/prompts")
    async def job_prompts(job_id: str, request: Request):
        """Live prompt travel: the running render uses these from its next frame on."""
        body = await request.json()
        upd = {}
        if "prompt_keys" in body:
            upd["prompt_keys"] = body["prompt_keys"] or []
        if "prompt_blend" in body:
            upd["prompt_blend"] = int(body["prompt_blend"])
        if body.get("now"):                     # Live mode: a keyframe at the frame about to render
            upd["now"] = str(body["now"])
        if isinstance(body.get("camera"), dict):  # Live camera (Text / Image -> Video)
            cam = body["camera"]
            upd["camera"] = {k: (bool(v) if k == "cam_3d" else float(v)) for k, v in cam.items()
                             if k in ("cam_zoom", "cam_rotate", "cam_x", "cam_y", "cam_yaw", "cam_pitch", "cam_3d")}
        job = jobs.set_prompts(job_id, upd) if upd else jobs.get(job_id)
        if not job:
            raise HTTPException(404, detail="Unknown job")
        if isinstance(body.get("ui"), dict) and job.state not in TERMINAL:   # the run remembers its latest prompts
            ui = runs.read_meta(job.run_id).get("ui") or {}
            runs.write_meta(job.run_id, ui=ui | body["ui"])
        return job.snapshot()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, detail="Unknown job")

        async def stream():
            revision = -1
            while True:
                snap = await asyncio.to_thread(jobs.wait_for_change, job, revision)
                if snap["revision"] != revision:
                    revision = snap["revision"]
                    yield f"data: {json.dumps(snap)}\n\n"
                else:
                    yield ": keepalive\n\n"
                if snap["state"] in TERMINAL:
                    break
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # -------------------------------------------------------------- runs
    @app.get("/api/runs")
    def list_runs():
        return {"runs": runs.list_runs(), "root": str(runs.RENDERS)}

    # the Runs section: every run (hidden too), sizes, hide / show / delete from disk
    @app.get("/api/runs/catalog")
    def runs_catalog():
        return {"runs": runs.catalog(), "root": str(runs.RENDERS)}

    @app.get("/api/runs/sizes")
    def runs_sizes(recount: bool = False):
        return runs.sizes(recount)

    def _busy_runs() -> set[str]:
        return {j["run_id"] for j in jobs.list() if j["state"] not in TERMINAL}

    @app.post("/api/runs/visibility")
    async def runs_visibility(request: Request):
        body = await request.json()
        done = []
        for rid in body.get("ids", []):
            try:
                runs.hide(rid) if body.get("hidden") else runs.unhide(rid)
                done.append(rid)
            except runs.RunNotFound:
                pass
        return {"changed": done}

    @app.post("/api/runs/purge")
    async def runs_purge(request: Request):
        """Delete runs from disk for good (the Runs section asks first)."""
        ids, busy = (await request.json()).get("ids", []), _busy_runs()
        deleted, failed, freed = [], [], 0
        for rid in ids:
            if rid in busy:
                failed.append({"id": rid, "error": "still rendering"}); continue
            try:
                freed += runs.purge(rid); deleted.append(rid)
            except runs.RunNotFound:
                failed.append({"id": rid, "error": "not found"})
            except (PermissionError, OSError) as e:
                failed.append({"id": rid, "error": str(e)})
        return {"deleted": deleted, "failed": failed, "freed": freed}

    @app.get("/api/runs/{run_id}")
    def describe_run(run_id: str):
        run_or_404(run_id)
        return runs.describe(run_id)

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        run_or_404(run_id)
        if any(j["run_id"] == run_id and j["state"] not in TERMINAL for j in jobs.list()):
            raise HTTPException(409, detail="That run is still rendering — cancel it first")
        runs.hide(run_id)                   # off the list; the files stay on disk
        return {"removed": run_id}

    @app.get("/api/runs/{run_id}/image")
    def run_image(run_id: str, layer: str, frame: int):
        run_or_404(run_id)
        path = runs.layer_path(run_id, layer, frame)
        if not path:
            raise HTTPException(404, detail="No image for that layer/frame")
        return FileResponse(path)

    @app.get("/api/runs/{run_id}/thumbnail")
    def run_thumbnail(run_id: str, layer: str, frame: int):
        run_or_404(run_id)
        path = runs.thumbnail(run_id, layer, frame)
        if not path:
            raise HTTPException(404, detail="No image for that layer/frame")
        return FileResponse(path, headers={"Cache-Control": "max-age=3600"})   # the URL changes when the run does

    @app.get("/api/runs/{run_id}/video")
    def run_video(run_id: str):
        run_or_404(run_id)
        path = runs.video_path(run_id)
        if not path:
            raise HTTPException(404, detail="This run has no video yet")
        return FileResponse(path, media_type="video/mp4")

    @app.post("/api/runs/{run_id}/video", status_code=202)
    def make_video(run_id: str):
        run_or_404(run_id)
        if runs.is_legacy(run_id):
            raise HTTPException(422, detail="Old VibeWarp runs can't be re-assembled here")
        if not runs.output_frames(run_id):
            raise HTTPException(422, detail="This run has no frames")
        return jobs.submit("video", run_id, {"fps": runs.fps(run_id)}).snapshot()

    @app.post("/api/runs/{run_id}/as_input")
    def run_as_input(run_id: str, frame: int | None = None):
        """A run fed back in (dragged from the stage onto the source slot): its mp4 as a new clip, or with
        `frame`, that output frame as a picture. Copied into inputs/, so it outlives the run."""
        import shutil
        run_or_404(run_id)
        seed = runs.summary(run_id).get("seed")
        stem = f"run{run_id}" + (f"_seed{seed}" if seed is not None else "")
        if frame is not None:
            src = runs.output_frames(run_id).get(frame)
            if not src:
                raise HTTPException(404, detail=f"Run {run_id} has no frame {frame}")
            name = f"{stem}_{frame:06d}.png"
        else:
            src = runs.video_path(run_id)
            if not src:
                raise HTTPException(404, detail="This run has no video yet")
            name = f"{stem}.mp4"
        target = INPUTS / f"{Path(name).stem}-{uuid.uuid4().hex[:6]}{src.suffix}"
        shutil.copyfile(src, target)
        if frame is None:
            return {"path": str(target), "name": name}
        from PIL import Image
        with Image.open(target) as im:
            w, h = im.size
        return {"path": str(target), "name": name, "image": True, "width": w, "height": h}

    @app.put("/api/runs/{run_id}/label")
    async def set_label(run_id: str, request: Request):
        run_or_404(run_id)
        label = " ".join(str((await request.json()).get("label") or "").split())[:120]
        runs.write_meta(run_id, label=label)
        return {"id": run_id, "label": label}

    @app.get("/api/runs/{run_id}/job")
    def get_job_json(run_id: str):
        """What Refine starts from: size, prompt, seed and the frames on disk.

        Engine runs keep a job.json and it answers all of that. A cloud render doesn't write one -
        the frames are still there, so read their size off the first one and take the prompt and seed
        from the page state saved beside them. Refining a Comfy result through a local model is a
        real thing to want: the graph gave it the motion, diffusion gives it the look.
        """
        path = run_or_404(run_id)
        frames = sorted((path / "frames").glob("*.png"))
        try:
            job = json.loads((path / "job.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if not frames:
                raise HTTPException(404, detail="This run has no frames")
            from PIL import Image
            with Image.open(frames[0]) as im:
                width, height = im.size
            ui = runs.read_meta(run_id).get("ui") or {}
            seed = ui.get("seed")
            job = {"engine": False, "width": width, "height": height, "hold": 1,
                   "prompt": ui.get("prompt") or "", "negative": ui.get("negative") or "",
                   "seed": int(seed) if isinstance(seed, (int, float)) else None}
        return {"engine": True} | job | {"frames_dir": str(path / "frames"), "frame_count": len(frames)}

    @app.get("/api/runs/{run_id}/ui")
    def get_ui(run_id: str):
        run_or_404(run_id)
        return runs.read_meta(run_id).get("ui") or {}

    @app.put("/api/runs/{run_id}/ui")
    async def put_ui(run_id: str, request: Request):
        run_or_404(run_id)
        runs.write_meta(run_id, ui=await request.json())
        return {"ok": True}

    @app.post("/api/runs/{run_id}/open")
    def open_run(run_id: str):
        os.startfile(run_or_404(run_id))  # Windows-only; this is a local tool
        return {"ok": True}

    # -------------------------------------------------------------- post-process
    @app.get("/api/post/options")
    def post_options():
        """What Enhance can offer: the upscaling models on disk, and the cloud upscalers the manifest
        installs. Each cloud one carries its own controls, so the panel draws them from this and adding
        another upscaler stays a JSON entry plus a graph."""
        try:
            on_cloud = [cloud.upscale_caps(w) for w in cloud.workflows("upscale")]
        except cloud.CloudError:
            on_cloud = []
        return {"upscalers": post.list_upscalers(settings().get("upscale_dir", "")),
                "cloud": on_cloud, "cloud_key": bool(h3_key(config())),
                "rife": post.RIFE_WEIGHTS.is_file(), "film": post.FILM_WEIGHTS.is_file(), "active": post.active_job()}

    @app.post("/api/runs/{run_id}/post")
    async def post_run(run_id: str, request: Request):
        body = await request.json()
        path = run_or_404(run_id)
        # A cloud upscale holds nothing here, so it is allowed to run while a render has the card.
        if body.get("cloud"):
            wid = str(body["cloud"])
            if wid not in {w["id"] for w in cloud.workflows("upscale")}:
                raise HTTPException(422, detail=f"Unknown cloud upscaler {wid}")
            params = body.get("params")
            if params is not None and not isinstance(params, dict):
                raise HTTPException(422, detail="params must be an object")
            try:
                return post.start_cloud(path, wid, params or {})
            except (RuntimeError, cloud.CloudError) as exc:
                raise HTTPException(409, detail=str(exc))
        if jobs.busy():
            raise HTTPException(409, detail="Wait for the render to finish — both need the GPU")
        smooth, scale, name = int(body.get("smooth", 1)), int(body.get("scale", 2)), body.get("upscaler") or None
        if smooth not in (1, 2, 3, 4) or scale not in (2, 4):
            raise HTTPException(422, detail="smooth must be 1–4, scale 2 or 4")
        if smooth == 1 and not name:
            raise HTTPException(422, detail="Pick an upscaler or a smoothing factor")
        interp = body.get("interp") or "rife"
        if interp not in post.INTERPS:
            raise HTTPException(422, detail=f"Unknown interpolation {interp}")
        up_dir = settings().get("upscale_dir", "")
        up_path = None
        if name:
            if name not in post.list_upscalers(up_dir):
                raise HTTPException(422, detail=f"Unknown upscaler {name}")
            up_path = os.path.join(up_dir, name)
        try:
            return post.start(path, "warpbox", fps=runs.fps(run_id), smooth=smooth, upscaler_path=up_path, scale=scale,
                              slowmo=bool(body.get("slowmo")), interp=interp)
        except RuntimeError as exc:
            raise HTTPException(409, detail=str(exc))

    @app.post("/api/tracks")
    async def make_tracks(request: Request):
        """Track a video's motion into path data a motion-control model can follow. The clip can be
        anything on disk — your footage, or a warp render, which is the interesting one."""
        body = await request.json()
        src = body.get("path") or ""
        if not src or not os.path.isfile(src):
            raise HTTPException(422, detail=f"no video at {src!r}")
        if jobs.busy():
            raise HTTPException(409, detail="Wait for the render to finish — both need the GPU")
        try:
            return post.start_tracks(
                src, INPUTS / "tracks",
                points=max(1, min(128, int(body.get("points", 24)))),
                length=max(2, min(1024, int(body.get("length", 81)))),
                width=max(16, min(4096, int(body.get("width", 720)))),
                height=max(16, min(4096, int(body.get("height", 720)))),
                trim=body.get("trim") or None, fps=float(body.get("fps", 16)))
        except RuntimeError as exc:
            raise HTTPException(409, detail=str(exc)) from None

    @app.post("/api/tracks/refit")
    async def refit_tracks(request: Request):
        """The same measurement at a different Length, output box or path count. All of the cost of
        tracking is in the chase, which was saved beside the JSON, so this is arithmetic — no GPU, and
        it runs happily while a render has the card."""
        from engine import tracks as tk
        body = await request.json()
        folder = (INPUTS / "tracks").resolve()
        chase = Path(str(body.get("chase") or "")).resolve()
        if folder not in chase.parents or not chase.is_file():
            raise HTTPException(422, detail="no saved measurement to refit — measure it again")
        try:
            return tk.refit(
                chase,
                points=max(1, min(128, int(body.get("points", 24)))),
                length=max(2, min(1024, int(body.get("length", 81)))),
                width=max(16, min(4096, int(body.get("width", 720)))),
                height=max(16, min(4096, int(body.get("height", 720)))))
        except tk.TrackError as exc:
            raise HTTPException(422, detail=str(exc)) from None

    @app.get("/api/fields")
    def field_kinds():
        """The authored motion fields this install offers, with the dials each one reads. The panel
        builds its controls from this, so a new field is an entry in engine/fields.py and no page edit."""
        from engine import fields
        return {"kinds": fields.KINDS}

    @app.post("/api/fields")
    async def make_field(request: Request):
        """A motion field authored rather than measured off a clip — from a formula, or from the
        lines the user dragged on the preview, which arrive as `strokes`. Milliseconds of numpy,
        so unlike tracking it answers in the request instead of going on the queue, and it hands the
        points back with the file so the page can animate them without reading it again."""
        from engine import fields
        body = await request.json()
        try:
            return fields.make(
                INPUTS / "tracks", kind=str(body.get("kind") or "pan"),
                points=max(1, min(128, int(body.get("points", 24)))),
                length=max(2, min(1024, int(body.get("length", 81)))),
                width=max(16, min(4096, int(body.get("width", 720)))),
                height=max(16, min(4096, int(body.get("height", 720)))),
                amount=max(-1.0, min(1.0, float(body.get("amount", 0.45)))),
                angle=float(body.get("angle", 0)) % 360.0,
                seed=int(body.get("seed", 0)),
                strokes=body.get("strokes"))        # the Draw kind: lines dragged on the preview
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, detail=f"could not draw that field: {exc}") from None

    @app.get("/api/masks")
    def mask_patterns():
        """The colour masks this install can draw. The panel builds its picker from this list, so a
        new pattern is an entry in engine/masks.py and no page edit."""
        from engine import masks
        return {"patterns": masks.PATTERNS, "colours": masks.NAMES}

    @app.post("/api/layers/plate")
    async def layers_plate(request: Request):
        """The layer stack composited into one clip, so the stage can show what will actually be
        rendered rather than the clip that happens to be at the bottom of it.

        Named after what is in it (every path, mode, opacity, level and mtime), so the same stack is
        flattened once and moving a slider back finds the old plate still there.
        """
        from engine import layers as layerlib
        body = await request.json()
        stack = layerlib.clean(body.get("layers") or [])
        for L in stack:
            if not os.path.isfile(L["path"]):
                raise HTTPException(404, detail=f"no layer at {L['path']}")
        if not layerlib.plan(stack):
            raise HTTPException(422, detail="a plate needs a layer that is not a matte")
        try:
            return layerlib.plate(INPUTS / "plates", stack,
                                  width=max(0, min(4096, int(body.get("width", 0)))),
                                  height=max(0, min(4096, int(body.get("height", 0)))),
                                  frames=max(0, min(4096, int(body.get("frames", 0)))),
                                  fps=max(0.0, min(120.0, float(body.get("fps", 0)))))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(422, detail=f"could not flatten the layers: {exc}") from None

    @app.get("/api/masks/preview")
    def mask_preview(pattern: str = "circle", width: int = 256, height: int = 144,
                     amount: float = 1.0):
        """One pattern as a small loop, for the picker. A mask you cannot see before you pick it is
        a guess, and the swatches never showed the shape. Drawn once, then read off disk."""
        from engine import masks
        try:
            out = masks.preview(INPUTS / "masks" / "_preview", pattern, width, height, amount)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, detail=f"could not draw that preview: {exc}") from None
        return FileResponse(out, media_type="video/mp4", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/masks")
    async def make_mask(request: Request):
        """One colour mask, drawn at the size and length of the render it is for. Flat colour and
        PIL, so it answers in the request like a field does rather than going on the queue."""
        from engine import masks
        body = await request.json()
        try:
            return masks.make(
                INPUTS / "masks", pattern=str(body.get("pattern") or "circle"),
                width=max(16, min(4096, int(body.get("width", 848)))),
                height=max(16, min(4096, int(body.get("height", 480)))),
                frames=max(2, min(1024, int(body.get("frames", 124)))),
                fps=max(1.0, min(120.0, float(body.get("fps", 24)))),
                amount=float(body.get("amount", 1.0)))
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, detail=f"could not draw that mask: {exc}") from None

    @app.get("/api/tracks/file")
    def tracks_file(path: str):
        """One tracks JSON, to show or to copy. Only out of the tracks folder: this reads a file by
        path, so it is pinned to the one directory DWARP writes them into."""
        folder = (INPUTS / "tracks").resolve()
        target = Path(path).resolve()
        if folder not in target.parents or target.suffix != ".json" or not target.is_file():
            raise HTTPException(404, detail="Not a tracks file")
        return FileResponse(target, media_type="application/json",
                            headers={"Cache-Control": "max-age=600"})

    @app.get("/api/post/{job_id}")
    def post_status(job_id: str):
        job = post.get_job(job_id)
        if not job:
            raise HTTPException(404, detail="Unknown post job")
        return job

    @app.post("/api/post/{job_id}/cancel")
    def post_cancel(job_id: str):
        job = post.cancel(job_id)
        if not job:
            raise HTTPException(404, detail="Unknown post job")
        return job

    def hidden_outputs(folder: Path) -> dict:
        """Enhanced mp4s taken off the list: {name: its mtime then}. The files stay; a newer file of the
        same name (enhanced again) shows up again."""
        try:
            return json.loads((folder / ".hidden").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @app.get("/api/runs/{run_id}/outputs")
    def outputs(run_id: str):
        folder = run_or_404(run_id) / "post"
        if not folder.is_dir():
            return []
        hidden = hidden_outputs(folder)
        files = [f for f in folder.glob("*.mp4") if not f.name.endswith(".part.mp4")
                 and hidden.get(f.name) != int(f.stat().st_mtime)]
        return [{"name": f.name, "bytes": f.stat().st_size, "modified": f.stat().st_mtime}
                for f in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)]

    @app.delete("/api/runs/{run_id}/post/{name}")
    def hide_output(run_id: str, name: str):
        """Take an enhanced mp4 off the list — the file stays in the run's post folder."""
        if not re.fullmatch(r"[\w.\-]+\.mp4", name):
            raise HTTPException(400, detail="Bad file name")
        folder = run_or_404(run_id) / "post"
        path = folder / name
        if not path.is_file():
            raise HTTPException(404, detail="No such output")
        hidden = hidden_outputs(folder)
        hidden[name] = int(path.stat().st_mtime)
        (folder / ".hidden").write_text(json.dumps(hidden, indent=2), encoding="utf-8")
        return {"removed": name}

    @app.get("/api/runs/{run_id}/post/{name}")
    def output_file(run_id: str, name: str):
        if not re.fullmatch(r"[\w.\-]+\.mp4", name):
            raise HTTPException(400, detail="Bad file name")
        path = run_or_404(run_id) / "post" / name
        if not path.is_file():
            raise HTTPException(404, detail="No such output")
        return FileResponse(path, media_type="video/mp4")

    return app


def main():
    parser = argparse.ArgumentParser(prog="dwarp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    import uvicorn
    app = build_app()
    warming = Timer(WARM_AFTER, warm)            # torch loads once the page has had the disk
    warming.daemon = True                        # never hold up a ctrl-c on account of a head start
    warming.start()
    if not args.no_browser:
        Timer(1.2, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    print(f"DWARP  →  http://{args.host}:{args.port}   (ctrl-c to stop)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
