"""Noise as a layer: the kinds there are, and the clips drawn from them.

Injecting noise is a different lever from raising Denoise. Denoise asks the model to invent more;
noise changes the pixels it is looking at -- and with the warp carrying paint forward, the grain
becomes structure the next frame elaborates. What it is made of therefore matters: per-pixel static
and slow fractal cloud are not the same instruction at all, and the first of them cannot be turned
into the second by blurring it, because a blur takes the grain's own edge away with it.

Every kind rests at mid grey, so a layer at Grain 0 is a layer that is not there under overlay or
soft light, and the dial opens it either side of that. One generator, numpy into ffmpeg, no GPU.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

# Each kind is one idea of what grain is. Data, so another one is an entry here and a branch in
# _frame -- the picker draws whatever this list says.
PATTERNS = [
    {"id": "static", "label": "Static", "base": "grey",
     "note": "Per-pixel grain, redrawn every frame — the finest there is, and the one the warp "
             "turns into texture fastest."},
    {"id": "cells", "label": "Cells", "base": "grey",
     "note": "The same thing with a far bigger brush: blocks of grey, hard-edged, a new draw on "
             "every frame."},
    {"id": "clouds", "label": "Clouds", "base": "grey",
     "note": "Fractal noise boiling slowly — soft masses rather than grain, for when you want the "
             "picture pushed around rather than roughened."},
    {"id": "lines", "label": "Lines", "base": "grey",
     "note": "Whole rows at a time, the way a tape drops out. Horizontal structure the warp drags "
             "sideways."},
]
BY_ID = {p["id"]: p for p in PATTERNS}

CELL = 22               # how wide a block is, in pixels, before the frame's own size is considered
DRAW_VERSION = 1        # bump when _frame changes shape, so an old cache is not handed back


def spec(kind: str) -> dict:
    return BY_ID.get(str(kind or "").lower()) or PATTERNS[0]


def catalogue() -> dict:
    """What the picker draws itself from."""
    return {"patterns": PATTERNS}


def _frame(kind: str, w: int, h: int, grain: float, rng, st: dict, z: float) -> np.ndarray:
    """One frame as 0..1 greys, mid grey at rest."""
    if kind == "cells":
        # Nearest-neighbour on purpose: a block with a soft edge is just blurred static, and the
        # point of this one is that the warp has something with a shape to carry forward.
        gw = max(2, w // CELL)
        gh = max(2, h // CELL)
        lat = rng.random((gh, gw), dtype=np.float32)
        out = lat[np.arange(h) * gh // h][:, np.arange(w) * gw // w]
    elif kind == "clouds":
        from engine.pulse import _fbm          # one kind of value noise in this codebase, not two
        out = _fbm(st, w, h, z, rng, octaves=4, cells=4.0)
        out = 0.5 + (out - 0.5) * 1.7          # the field keeps to the middle of its range
    elif kind == "lines":
        out = np.repeat(rng.random((h, 1), dtype=np.float32), w, axis=1)
    else:                                       # static
        out = rng.random((h, w), dtype=np.float32)
    return np.clip(0.5 + (out - 0.5) * grain, 0.0, 1.0).astype(np.float32)


def clip(dest_dir: Path, *, kind: str = "static", width: int = 768, height: int = 768,
         frames: int = 124, fps: float = 24.0, grain: float = 1.0, seed: int = 0) -> dict:
    """One noise layer, named after what is in it, so the same ask is made once."""
    p = spec(kind)
    width, height = max(16, int(width)) & ~1, max(16, int(height)) & ~1
    frames, fps = max(1, int(frames)), max(1.0, float(fps))
    grain = min(1.0, max(0.05, float(grain)))
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    key = f"{p['id']}|{width}x{height}|{frames}|{fps:.4f}|{grain:.3f}|{int(seed)}|{DRAW_VERSION}"
    out = dest_dir / f"noise-{p['id']}-{hashlib.sha1(key.encode()).hexdigest()[:6]}.mp4"
    if not (out.is_file() and out.stat().st_size):
        # Drawn at the render's own size rather than small and blown up: grain that has been
        # through a scaler is not grain any more, it is a pattern of soft blobs.
        # ffmpeg writes to a part file and the finished clip is moved into place, because the name
        # is also the cache: anything else asking for this one checks `out.is_file()` and a file
        # being written passes that check happily. The flatten then opened a clip that was half a
        # clip -- "Invalid data found when processing input", from a layer that was fine seconds
        # later. os.replace is atomic, so a reader sees either nothing or the whole thing.
        tmp = out.with_name(out.name + ".part.mp4")   # still an mp4: ffmpeg reads the container off the name
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{width}x{height}", "-r", f"{fps:.4f}", "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", "-preset", "veryfast", str(tmp)]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        rng = np.random.default_rng(int(seed) or 13)
        st: dict = {}
        try:
            for i in range(frames):
                g = _frame(p["id"], width, height, grain, rng, st, i * 0.08)
                pipe.stdin.write((g * 255.0).astype(np.uint8).tobytes())
        finally:
            pipe.stdin.close()
            err = pipe.stderr.read().decode("utf-8", "replace")
            pipe.wait()
        if pipe.returncode or not (tmp.is_file() and tmp.stat().st_size):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg could not write the noise: {err.strip()[:200] or 'no output'}")
        os.replace(tmp, out)
    return {"path": str(out), "name": out.name, "kind": p["id"], "label": p["label"],
            "width": width, "height": height, "frames": frames, "fps": round(fps, 3),
            "grain": round(grain, 3)}


PREVIEW_W, PREVIEW_H, PREVIEW_FRAMES = 256, 144, 32


def preview(dest_dir: Path, kind: str = "static", width: int = PREVIEW_W,
            height: int = PREVIEW_H) -> Path:
    """One kind as a small loop for the picker. Grain is left at the top of the dial here: the card
    is showing you what the kind looks like, not what you have it set to."""
    p = spec(kind)
    width = max(32, min(640, int(width))) & ~1
    height = max(32, min(640, int(height))) & ~1
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{p['id']}-{width}x{height}-v{DRAW_VERSION}.mp4"
    if out.is_file() and out.stat().st_size:
        return out
    made = clip(dest_dir, kind=p["id"], width=width, height=height, frames=PREVIEW_FRAMES,
                fps=24.0, grain=1.0)
    Path(made["path"]).replace(out)
    return out
