"""Run folders on disk.

Engine runs:   renders/<YYYYMMDD>/<n>/  frames/NNNNNN.png  src/NNNNNN.jpg  control/<kind>/NNNNNN.jpg
               debug/  job.json  meta.json  video.mp4  post/*.mp4  .thumbs/
               A run is still named by its number alone — the day folder is only where it sits, so a
               run keeps its id when the folders are tidied, and runs made before the day folders
               existed still lie flat in renders/<n>/ and are found exactly the same way.
Legacy runs:   renders/warpbox/<n>/  (made by VibeWarp) — listed read-only as id "v<n>".
Deleted runs:  renders/.trash/ — a purge moves the folder there whole (one rename) and a background
               thread empties the bin, so deleting a hundred runs is as quick as deleting one.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
RENDERS = HERE / "renders"
LEGACY = RENDERS / "warpbox"
TRASH = RENDERS / ".trash"
_LEGACY_FRAME = re.compile(r"^warpbox\(\d+\)_(\d+)\.png$")
_DAY = re.compile(r"^20\d{6}$")          # a day folder, 20261125


class RunNotFound(Exception):
    pass


# ------------------------------------------------------------------ where a run lives
# A run's id is its number; the folder it sits in is not part of it. New runs go under the day they
# were made, older ones lie flat in renders/, and both are found through one index built from a
# single listing — so nothing else in DWARP has to know which layout a run is in, and a run keeps
# its id (and every link, label and source_run pointing at it) when the folders are tidied.
_INDEX: dict[str, Path] = {}
_INDEX_AT = 0.0
_INDEX_LOCK = threading.Lock()


def _is_day(path: Path) -> bool:
    """A day folder, not a run that happens to be numbered like one: a run carries its own files."""
    return not any((path / f).exists() for f in ("meta.json", "job.json", "frames"))


def _scan() -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not RENDERS.is_dir():
        return found
    with os.scandir(RENDERS) as it:
        for e in it:
            if not e.is_dir() or not e.name.isdigit():
                continue
            path = Path(e.path)
            if _DAY.match(e.name) and _is_day(path):
                with os.scandir(path) as day:
                    for d in day:
                        if d.is_dir() and d.name.isdigit():
                            found[d.name] = Path(d.path)
            else:
                found[e.name] = path
    return found


def index(fresh: bool = False) -> dict[str, Path]:
    """Run id -> folder. Built on demand; a miss rebuilds it once (a run can arrive from the engine,
    or be moved, while the server is up) but not oftener than once a second."""
    global _INDEX_AT
    with _INDEX_LOCK:
        if fresh or not _INDEX:
            _INDEX.clear()
            _INDEX.update(_scan())
            _INDEX_AT = time.time()
        return dict(_INDEX)


def run_dir(run_id: str) -> Path:
    if re.fullmatch(r"v\d+", run_id):
        path = LEGACY / run_id[1:]
        if not path.is_dir():
            raise RunNotFound(run_id)
        return path
    if not re.fullmatch(r"\d+", run_id):
        raise RunNotFound(run_id)
    path = _INDEX.get(run_id) or index().get(run_id)
    if path is None or not path.is_dir():
        if time.time() - _INDEX_AT > 1.0:           # don't rescan once per miss in a tight loop
            path = index(fresh=True).get(run_id)
        if path is None or not path.is_dir():
            raise RunNotFound(run_id)
    return path


def is_legacy(run_id: str) -> bool:
    return run_id.startswith("v")


def day_dir(when: float | None = None) -> Path:
    """Today's folder, made if it isn't there."""
    path = RENDERS / time.strftime("%Y%m%d", time.localtime(when))
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_run() -> tuple[str, Path]:
    RENDERS.mkdir(exist_ok=True)
    with _INDEX_LOCK:
        _INDEX.clear()
        _INDEX.update(_scan())                  # the highest number on disk, not the highest we remember
        n = str(max((int(k) for k in _INDEX), default=-1) + 1)
        path = day_dir() / n
        path.mkdir()
        _INDEX[n] = path
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


# ------------------------------------------------------------------ asking a file about itself
def probe_field(src: Path | str, *args: str) -> str:
    """One ffprobe field, or "" if ffprobe has nothing to say about it. Here rather than in the
    engine because the engine, post and the registry all need to ask, and asking must not cost a
    torch import."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "default=nw=1:nk=1", str(src)],
                           capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    return lines[0] if lines else ""


def has_audio(src: Path | str) -> bool:
    """Does this file carry a soundtrack?"""
    return bool(probe_field(src, "-select_streams", "a:0", "-show_entries", "stream=codec_name"))


def adopt(video: Path, *, came: str, verb: str, detail: str = "", rate: float = 24.0,
          ui: dict | None = None, source: str = "", move: bool = True) -> str:
    """Take a finished video and make it a run of its own.

    Anything DWARP produces that you would want to scrub, thumbnail or compare belongs in the strip
    rather than in a row of file chips: an upscale, a smooth, the track visualiser a motion graph
    writes. It gets its own frames so the card has a picture and the transport works, and a label
    naming what it came from — "upscale · wanmove" beside the "wanmove · render" that made it.
    """
    from engine.cloud import extract          # runs.py is imported by the engine, so this stays lazy

    rid, path = new_run()
    try:
        dest = path / "video.mp4"
        shutil.move(str(video), str(dest)) if move else shutil.copy2(str(video), str(dest))
        extract(dest, path / "frames")
        # It arrives as a finished file: if there is sound on it, the run says so, the same way the
        # assembler marks a render it has put the clip's audio back on.
        if has_audio(dest):
            (path / "sound.json").write_text(json.dumps({"from": "adopted"}), encoding="utf-8")
        write_meta(rid, label=f"{verb} · {came}".strip(" · "), fps=float(rate),
                   ui={**(ui or {}), "kind": verb, "source_run": source, "post": detail})
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)    # a half-made run in the list is worse than none
        raise
    return rid


def came_from(run_id: str) -> str:
    """What to call a run in the label of something made out of it: "wanmove · render" -> "wanmove"."""
    try:
        label = (read_meta(run_id).get("label") or "").split(" · ")[0].strip()
    except (RunNotFound, OSError, ValueError):
        label = ""
    return label or run_id


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
                  "diff": "Diff", "shape": "Shape", "qr": "QR Code Monster"}      # diff = the DepthDiff per-pixel strength mask


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
    if layer == "output" and not is_legacy(run_id):   # the usual case: the file by name, no listing of every frame
        src = run_dir(run_id) / "frames" / f"{frame:06d}.png"
        src = src if src.is_file() else None
    else:
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


GIF_WIDE = 640          # the long side a shared gif is drawn at: a gif is a thing people paste


def gif_path(run_id: str, *, wide: int = GIF_WIDE) -> Path:
    """The run as a looping gif, built on demand and kept beside its mp4.

    It is made from video.mp4 rather than from the frames, so every timing decision the assembler
    already made -- the clip's own rate, a run held on twos or threes, where the trim started, a
    render stopped half way -- is baked into it and the gif cannot drift from the file you would
    have downloaded. A gif has no sound, so a run with a soundtrack simply loses it.

    Two passes over the clip: palettegen reads it and chooses 256 colours for the whole thing,
    paletteuse maps the frames onto them. That is the difference between a gif that looks like the
    render and one that looks like 1996. The dither is ordered rather than error-diffused because
    error diffusion re-rolls its noise every frame -- it crawls on screen, and since each frame then
    differs everywhere it also roughly doubles the file.
    """
    src = video_path(run_id)
    if not src:
        raise RunNotFound(f"Run {run_id} has no video to make a gif from")
    wide = max(64, min(1920, int(wide or GIF_WIDE)))
    target = run_dir(run_id) / f"video_{wide}.gif"
    if target.is_file() and target.stat().st_mtime >= src.stat().st_mtime:
        return target                      # still the same mp4 underneath it
    w = int(probe_field(src, "-select_streams", "v:0", "-show_entries", "stream=width") or 0)
    h = int(probe_field(src, "-select_streams", "v:0", "-show_entries", "stream=height") or 0)
    # Fit the long side, and never enlarge: a 384 px run makes a 384 px gif.
    fit = (f"scale={wide}:-1:flags=lanczos," if w >= h else f"scale=-1:{wide}:flags=lanczos,") \
        if max(w, h) > wide else ""
    tmp = target.with_suffix(".tmp.gif")
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-filter_complex",
             f"[0:v]{fit}split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
             "-loop", "0", str(tmp)], check=True)
        os.replace(tmp, target)
    except (OSError, subprocess.CalledProcessError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not make a gif: {e}") from e
    return target


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
            "video": job.get("video", ""), "shape": job.get("shape", ""), "shape_start": int(job.get("shape_start", 0))}


def has_sound(run_id: str) -> bool:
    """Does the run's mp4 carry audio? A cloud render's soundtrack comes down with the picture. A
    local render is assembled from PNGs, which are silent -- but the clip's own audio goes back on
    at the end, and when it does the assembler leaves sound.json behind to say so."""
    if (run_dir(run_id) / "sound.json").is_file():
        return True
    try:
        job = json.loads((run_dir(run_id) / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return job.get("family") in ("comfy", "h3") and job.get("h3_sound", True)


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
        "sound": bool(video) and has_sound(run_id),
        "seed": (meta.get("ui") or {}).get("seed"),
    }


def list_runs() -> list[dict]:
    ids = list(index(fresh=True))
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


def unhide(run_id: str) -> None:
    """Put a hidden run back on the list."""
    (run_dir(run_id) / HIDDEN).unlink(missing_ok=True)


# ------------------------------------------------------------------ the Runs section: every run, sizes, delete
def _kind(meta: dict) -> str:
    kind = (meta.get("ui") or {}).get("kind") or ""
    if not kind and str(meta.get("label", "")).startswith("refine"):
        kind = "refine"
    return kind


class _nothing:
    def __enter__(self): return iter(())
    def __exit__(self, *a): return False


_CARDS: dict[str, tuple[tuple, dict]] = {}     # run id -> (the times it was read at, its catalog entry)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def catalog() -> list[dict]:
    """Every run, hidden ones too, kept light (no per-frame stat) so a thousand runs list quickly."""
    out = []
    ids = list(index(fresh=True))
    ids += ["v" + p.name for p in LEGACY.iterdir() if p.is_dir() and p.name.isdigit()] if LEGACY.is_dir() else []
    for rid in ids:
        try:
            path = run_dir(rid)
            fdir = path / "frames"
            # when its frames were written (the run folder's own time moves with thumbnails and hiding)
            stamp = next((f for f in (fdir, path / "meta.json") if f.exists()), path)
            # A finished run never changes again, so its whole card is remembered against the times of
            # the only two things that could change it. That is what makes the second open of the Runs
            # section instant rather than a thousand folders read through again.
            key = (stamp.stat().st_mtime, _mtime(path / "meta.json"))
            known = _CARDS.get(rid)
            if known and known[0] == key:
                out.append({**known[1], "hidden": is_hidden(rid)})
                continue
            meta = read_meta(rid)
            if is_legacy(rid):
                frames = sorted(output_frames(rid))
            else:                                  # a plain listing: far quicker than glob over a thousand runs
                with os.scandir(fdir) if fdir.is_dir() else _nothing() as it:
                    frames = sorted(int(e.name[:-4]) for e in it if e.name.endswith(".png") and e.name[:-4].isdigit())
            card = {"id": rid, "label": meta.get("label", ""), "kind": _kind(meta), "frames": len(frames),
                    "last_frame": frames[-1] if frames else None, "modified": key[0], "legacy": is_legacy(rid)}
            _CARDS[rid] = (key, card)
            out.append({**card, "hidden": is_hidden(rid)})
        except (RunNotFound, OSError, ValueError):
            pass
    for gone in set(_CARDS) - set(ids):
        _CARDS.pop(gone, None)
    # latest first by run number (new runs count up); old VibeWarp runs after
    return sorted(out, key=lambda r: (not r["legacy"], int(r["id"].lstrip("v"))), reverse=True)


def dir_size(path: Path) -> int:
    total, stack = 0, [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_size   # from the directory listing on Windows
                    except OSError:
                        pass
        except OSError:
            pass
    return total


# Sizes are counted in a background thread (a thousand folders of frames take a while on a hard disk), kept
# in renders/.sizes.json so the next start shows them at once; a run is re-counted when its frames change.
_SIZES: dict[str, tuple[float, int]] = {}
_SIZER: threading.Thread | None = None
_SIZES_FILE = ".sizes.json"


def _load_sizes() -> None:
    try:
        saved = json.loads((RENDERS / _SIZES_FILE).read_text(encoding="utf-8"))
        _SIZES.update({k: (float(v[0]), int(v[1])) for k, v in saved.items()})
    except (OSError, ValueError, TypeError, IndexError):
        pass


def _count_sizes() -> None:
    live = set()
    for r in catalog():
        live.add(r["id"])
        known = _SIZES.get(r["id"])
        if known and known[0] == r["modified"]:
            continue
        try:
            _SIZES[r["id"]] = (r["modified"], dir_size(run_dir(r["id"])))
        except RunNotFound:
            pass
    for gone in set(_SIZES) - live:              # deleted outside DWARP
        _SIZES.pop(gone, None)
    try:
        (RENDERS / _SIZES_FILE).write_text(json.dumps(_SIZES), encoding="utf-8")
    except OSError:
        pass


def sizes(recount: bool = False) -> dict:
    """Bytes per run as far as counted; done once no count is running. recount (asked by the Runs section a
    moment after it opens) starts a pass over new or changed runs; plain calls only report."""
    global _SIZER
    if not _SIZES and _SIZER is None:
        _load_sizes()
    if recount and (_SIZER is None or not _SIZER.is_alive()):
        _SIZER = threading.Thread(target=_count_sizes, daemon=True)
        _SIZER.start()
    return {"sizes": {rid: b for rid, (_, b) in list(_SIZES.items())}, "done": _SIZER is None or not _SIZER.is_alive()}


def _force(func, p, _exc):                     # read-only files (Windows) would stop rmtree
    os.chmod(p, stat.S_IWRITE)
    func(p)


_BINMAN: threading.Thread | None = None


def _empty_now() -> None:
    while True:
        try:
            with os.scandir(TRASH) as it:
                doomed = [Path(e.path) for e in it]
        except OSError:
            return
        went = False
        for p in doomed:
            try:
                shutil.rmtree(p, onexc=_force) if p.is_dir() else p.unlink()
                went = True
            except OSError:
                pass      # something still holds it: leave it, the next purge or the next start tries again
        if not went:      # the bin is empty, or what is left will not go
            return


def empty_trash() -> None:
    """Clear renders/.trash in the background. Called after a purge, and once at startup so a bin left
    behind by a crash (or by quitting mid-delete) is cleared the next time DWARP runs."""
    global _BINMAN
    if _BINMAN is None or not _BINMAN.is_alive():
        _BINMAN = threading.Thread(target=_empty_now, daemon=True)
        _BINMAN.start()


def purge(run_id: str) -> int:
    """Delete a run's folder from disk for good. Returns the bytes freed. Old VibeWarp runs are read-only.

    The folder is moved into renders/.trash and a background thread empties the bin, because a run is
    thousands of PNGs and deleting a stack of them takes minutes — minutes you would otherwise spend
    watching a Runs section that cannot answer. The move is one rename, so the run is gone from every
    listing the instant you confirm it, and whether the bytes have left the disk yet is the disk's
    own business. If a run cannot be moved (something still holds a file in it open) it is deleted
    where it lies, the slow way, exactly as it always was.
    """
    if is_legacy(run_id):
        raise PermissionError("old VibeWarp runs are read-only")
    path = run_dir(run_id)
    freed = _SIZES.pop(run_id, (0, None))[1]
    if freed is None:
        freed = dir_size(path)
    _CARDS.pop(run_id, None)
    try:
        TRASH.mkdir(parents=True, exist_ok=True)
        os.rename(path, TRASH / f"{run_id}-{int(time.time() * 1000):x}")
    except OSError:
        shutil.rmtree(path, onexc=_force)
    else:
        with _INDEX_LOCK:
            _INDEX.pop(run_id, None)
        if _DAY.match(path.parent.name):
            try:
                path.parent.rmdir()        # the last run of that day: don't leave the empty folder
            except OSError:
                pass
        empty_trash()
    return freed
