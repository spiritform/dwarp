"""DWARP server: the page, the render queue, run history and post-processing.

    .venv\\Scripts\\python server.py [--port 8013] [--no-browser]
"""
from __future__ import annotations

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

import post  # noqa: E402
import runs  # noqa: E402
from jobs import TERMINAL, JobManager  # noqa: E402
from models import list_checkpoints  # noqa: E402

INPUTS = HERE / "inputs"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}
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
        for c in mode.get("controlnets", {}).values():
            names = c["file"] if isinstance(c["file"], list) else [c["file"]]
            paths = [f"{root}/controlnet/{n}" for n in names]
            c["path"] = next((q for q in paths if os.path.isfile(q)), paths[0])
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
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": fps, "frames": frames,
            "duration": duration, "codec": st.get("codec_name", "")}


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

    @app.get("/api/checkpoints")
    def checkpoints():
        root = settings().get("checkpoint_dir", "")
        return {"root": root, "checkpoints": list_checkpoints(root)}

    # -------------------------------------------------------------- input video
    @app.post("/api/upload")
    async def upload(request: Request, filename: str = "clip.mp4"):
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTS:
            raise HTTPException(422, detail=f"Not a video file: {filename}")
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
        return {"path": str(target), "name": filename, "bytes": size}

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
    def diff_base(path: str = "", frame: int = 0, width: int = 512, height: int = 512, kind: str = "luma"):
        """One source frame at render size, as the DepthDiff mask starts from it: the frame itself
        (luma) or its depth map. The page shapes it into the mask live."""
        if not path or not os.path.isfile(path):
            raise HTTPException(404, detail="No video at that path")
        vf = f"select=eq(n\\,{max(0, frame)}),scale={max(8, width)}:{max(8, height)}:flags=lanczos"
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-frames:v", "1",
                              "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
        if out.returncode != 0 or not out.stdout:
            raise HTTPException(422, detail="Could not read that frame")
        data = out.stdout
        if kind == "depth":
            import io
            import numpy as np
            from PIL import Image
            from engine.warp import depth_map
            buf = io.BytesIO()
            depth_map(np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))).convert("RGB").save(buf, "PNG")
            data = buf.getvalue()
        return StreamingResponse(iter([data]), media_type="image/png", headers={"Cache-Control": "max-age=600"})

    # -------------------------------------------------------------- jobs
    @app.post("/api/jobs", status_code=202)
    async def submit(request: Request):
        body = await request.json()
        job, meta = body.get("job") or {}, body.get("meta") or {}
        for key in ("video", "checkpoint"):
            if not job.get(key) or not os.path.isfile(job[key]):
                raise HTTPException(422, detail=[{"message": f"{key} not found: {job.get(key)!r}"}])
        for c in job.get("controlnets", []):
            if not os.path.isfile(c.get("path", "")):
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

    @app.get("/api/runs/{run_id}")
    def describe_run(run_id: str):
        run_or_404(run_id)
        return runs.describe(run_id)

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        run_or_404(run_id)
        if any(j["run_id"] == run_id and j["state"] not in TERMINAL for j in jobs.list()):
            raise HTTPException(409, detail="That run is still rendering — cancel it first")
        runs.delete(run_id)
        return {"deleted": run_id}

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
        return FileResponse(path)

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
        return jobs.submit("video", run_id, {"fps": run_fps(run_id)}).snapshot()

    @app.put("/api/runs/{run_id}/label")
    async def set_label(run_id: str, request: Request):
        run_or_404(run_id)
        label = " ".join(str((await request.json()).get("label") or "").split())[:120]
        runs.write_meta(run_id, label=label)
        return {"id": run_id, "label": label}

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
    def run_fps(run_id: str) -> float:
        meta = runs.read_meta(run_id)
        if meta.get("fps"):
            return float(meta["fps"])
        ui = meta.get("ui") or {}
        src = ((ui.get("clip") or {}).get("source") or {}).get("fps") or 24
        return src / (2 if ui.get("kind") == "test" else 1)

    @app.get("/api/post/options")
    def post_options():
        return {"upscalers": post.list_upscalers(settings().get("upscale_dir", "")),
                "rife": post.RIFE_WEIGHTS.is_file(), "active": post.active_job()}

    @app.post("/api/runs/{run_id}/post")
    async def post_run(run_id: str, request: Request):
        body = await request.json()
        path = run_or_404(run_id)
        if jobs.busy():
            raise HTTPException(409, detail="Wait for the render to finish — both need the GPU")
        smooth, scale, name = int(body.get("smooth", 1)), int(body.get("scale", 2)), body.get("upscaler") or None
        if smooth not in (1, 2, 3, 4) or scale not in (2, 4):
            raise HTTPException(422, detail="smooth must be 1–4, scale 2 or 4")
        if smooth == 1 and not name:
            raise HTTPException(422, detail="Pick an upscaler or a smoothing factor")
        up_dir = settings().get("upscale_dir", "")
        up_path = None
        if name:
            if name not in post.list_upscalers(up_dir):
                raise HTTPException(422, detail=f"Unknown upscaler {name}")
            up_path = os.path.join(up_dir, name)
        try:
            return post.start(path, "warpbox", fps=run_fps(run_id), smooth=smooth, upscaler_path=up_path, scale=scale)
        except RuntimeError as exc:
            raise HTTPException(409, detail=str(exc))

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

    @app.get("/api/runs/{run_id}/outputs")
    def outputs(run_id: str):
        folder = run_or_404(run_id) / "post"
        if not folder.is_dir():
            return []
        files = [f for f in folder.glob("*.mp4") if not f.name.endswith(".part.mp4")]
        return [{"name": f.name, "bytes": f.stat().st_size, "modified": f.stat().st_mtime}
                for f in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)]

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
    if not args.no_browser:
        Timer(1.2, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
