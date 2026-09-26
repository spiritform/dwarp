"""Run folders on disk.

Engine runs:   renders/<n>/  frames/NNNNNN.png  src/NNNNNN.jpg  control/<kind>/NNNNNN.jpg  debug/  job.json  meta.json
               video.mp4  post/*.mp4  .thumbs/
Legacy runs:   renders/warpbox/<n>/  (made by VibeWarp) — listed read-only as id "v<n>".
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
RENDERS = HERE / "renders"
LEGACY = RENDERS / "warpbox"
_LEGACY_FRAME = re.compile(r"^warpbox\(\d+\)_(\d+)\.png$")


class RunNotFound(Exception):
    pass


def run_dir(run_id: str) -> Path:
    if re.fullmatch(r"\d+", run_id):
        path = RENDERS / run_id
    elif re.fullmatch(r"v\d+", run_id):
        path = LEGACY / run_id[1:]
    else:
        raise RunNotFound(run_id)
    if not path.is_dir():
        raise RunNotFound(run_id)
    return path


def is_legacy(run_id: str) -> bool:
    return run_id.startswith("v")


def new_run() -> tuple[str, Path]:
    RENDERS.mkdir(exist_ok=True)
    nums = [int(p.name) for p in RENDERS.iterdir() if p.is_dir() and p.name.isdigit()]
    n = str(max(nums, default=-1) + 1)
    path = RENDERS / n
    path.mkdir()
    return n, path


# ------------------------------------------------------------------ meta
def read_meta(run_id: str) -> dict:
    path = run_dir(run_id)
    if is_legacy(run_id):
        ui = json.loads((path / "warpbox.json").read_text(encoding="utf-8")) if (path / "warpbox.json").is_file() else {}
        lab = path / ".vibewarp-label"          # where VibeWarp kept run labels
        label = lab.read_text(encoding="utf-8").strip() if lab.is_file() else ""
        return {"label": label, "ui": ui}
    f = path / "meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def write_meta(run_id: str, **fields) -> dict:
    path = run_dir(run_id)
    meta = read_meta(run_id)
    meta.update(fields)
    if is_legacy(run_id):
        if "ui" in fields:
            (path / "warpbox.json").write_text(json.dumps(fields["ui"], indent=2), encoding="utf-8")
        if "label" in fields:
            (path / ".vibewarp-label").write_text(fields["label"] or "", encoding="utf-8")
    else:
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ------------------------------------------------------------------ frames / layers
def output_frames(run_id: str) -> dict[int, Path]:
    path = run_dir(run_id)
    if not is_legacy(run_id):
        return {int(p.stem): p for p in (path / "frames").glob("*.png")} if (path / "frames").is_dir() else {}
    found = {}
    for p in path.glob("*.png"):
        m = _LEGACY_FRAME.match(p.name)
        if m:
            found[int(m.group(1))] = p
    return found


def init_frames(run_id: str) -> dict[int, Path]:
    path = run_dir(run_id)
    if not is_legacy(run_id):
        return {int(p.stem): p for p in (path / "src").glob("*.jpg")} if (path / "src").is_dir() else {}
    # VibeWarp numbers extracted frames from 1
    vf = path / "video_frames"
    return {int(p.stem) - 1: p for p in vf.glob("*.jpg")} if vf.is_dir() else {}


# ControlNet hints the engine saved per frame: control/<kind>/NNNNNN.jpg -> layer "control_<kind>"
CONTROL_LABELS = {"depth": "Depth", "softedge": "Edge", "hed": "Edge", "canny": "Canny", "lineart": "Lineart",
                  "diff": "Diff"}      # diff = the DepthDiff per-pixel strength mask


def control_frames(run_id: str, kind: str) -> dict[int, Path]:
    d = run_dir(run_id) / "control" / kind
    if kind not in CONTROL_LABELS or is_legacy(run_id) or not d.is_dir():
        return {}
    return {int(p.stem): p for p in d.glob("*.jpg")}


def layer_path(run_id: str, layer: str, frame: int) -> Path | None:
    if layer.startswith("control_"):
        return control_frames(run_id, layer[len("control_"):]).get(frame)
    frames =output_frames(run_id) if layer == "output" else init_frames(run_id) if layer == "init" else {}
    return frames.get(frame)


def thumbnail(run_id: str, layer: str, frame: int) -> Path | None:
    src = layer_path(run_id, layer, frame)
    if not src:
        return None
    cache = run_dir(run_id) / ".thumbs"
    dest = cache / f"{layer}_{frame:06d}.jpg"
    if not dest.is_file() or dest.stat().st_mtime < src.stat().st_mtime:
        cache.mkdir(exist_ok=True)
        img = Image.open(src).convert("RGB")
        img.thumbnail((320, 320))
        img.save(dest, quality=84)
    return dest


def video_path(run_id: str) -> Path | None:
    path = run_dir(run_id)
    for name in ("video.mp4", "warpbox.mp4"):
        if (path / name).is_file():
            return path / name
    return None


# ------------------------------------------------------------------ listing
def timing(run_id: str) -> dict | None:
    """Where the run's frames sit in its clip: frame n is source frame (frame_start + n) * nth, at the
    clip's fps — so the viewer can show clip time, the keyframe lane's units. None for old runs."""
    try:
        job = json.loads((run_dir(run_id) / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    nth = max(1, int(job.get("nth", 1)))
    src = (((read_meta(run_id).get("ui") or {}).get("clip") or {}).get("source") or {}).get("fps")
    if job.get("t2v_frames"):                 # text -> video: its own clock, not the panel's clip
        src = None
    return {"frame_start": int(job.get("frame_start", 0)), "nth": nth, "fps": float(src or fps(run_id) * nth),
            "video": job.get("video", "")}


def describe(run_id: str) -> dict:
    out, init = output_frames(run_id), init_frames(run_id)
    return {"run": run_id, "timing": timing(run_id), "layers": [
        {"id": "output", "label": "Output", "frames": sorted(out)},
        {"id": "init", "label": "Source", "frames": sorted(init)},
    ] + [{"id": f"control_{kind}", "label": label, "frames": sorted(frames)}
         for kind, label in CONTROL_LABELS.items() if (frames := control_frames(run_id, kind))]}


def fps(run_id: str, meta: dict | None = None) -> float:
    """The run's playback rate: saved at submit; older runs derive it from the clip (Preview = every 2nd frame)."""
    meta = read_meta(run_id) if meta is None else meta
    if meta.get("fps"):
        return float(meta["fps"])
    ui = meta.get("ui") or {}
    src = ((ui.get("clip") or {}).get("source") or {}).get("fps") or 24
    return src / (2 if ui.get("kind") == "test" else 1)


def summary(run_id: str) -> dict:
    path = run_dir(run_id)
    frames = output_frames(run_id)
    meta = read_meta(run_id)
    video = video_path(run_id)
    mtimes = [p.stat().st_mtime for p in frames.values()] or [path.stat().st_mtime]
    return {
        "id": run_id, "label": meta.get("label", ""), "modified": max(mtimes),
        "frames": len(frames), "last_frame": max(frames) if frames else None,
        "prompt": (meta.get("ui") or {}).get("prompt", ""),
        "video_available": bool(video), "video_modified": video.stat().st_mtime if video else None,
        "legacy": is_legacy(run_id), "fps": fps(run_id, meta),
    }


def list_runs() -> list[dict]:
    ids = []
    if RENDERS.is_dir():
        ids += [p.name for p in RENDERS.iterdir() if p.is_dir() and p.name.isdigit()]
    if LEGACY.is_dir():
        ids += ["v" + p.name for p in LEGACY.iterdir() if p.is_dir() and p.name.isdigit()]
    runs = []
    for rid in ids:
        try:
            if is_hidden(rid):
                continue
            runs.append(summary(rid))
        except (RunNotFound, OSError):
            pass
    return sorted(runs, key=lambda r: r["modified"], reverse=True)


HIDDEN = ".hidden"                         # a removed run: off the list, files kept


def hide(run_id: str) -> None:
    """Take a run off the list. Nothing is deleted: its folder stays in renders/, and removing the
    marker file brings it back."""
    (run_dir(run_id) / HIDDEN).touch()


def is_hidden(run_id: str) -> bool:
    return (run_dir(run_id) / HIDDEN).exists()
