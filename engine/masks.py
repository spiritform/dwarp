"""Colour masks: a video whose only content is which other video goes where.

H3 reads several whole videos as context at once, and a flat-colour video alongside them is how you
say *where* and *when* each one lands: red means <Video 1>, green means <Video 2>, blue means
<Video 3>, and the prompt spells that out in words. The model has no idea what red means until the
brief tells it, so the legend that comes back with a mask is as much a part of it as the file.

Everything here is deliberately blunt. The regions are flat, fully saturated and hard-edged, because a
gradient or a soft edge reads as content rather than as an instruction. The motion is eased the same
way the motion fields are, and the whole thing is drawn with PIL and piped into ffmpeg — no GPU, a
second or two, so a mask can be made and remade while you are deciding.
"""
from __future__ import annotations

import math
import subprocess
import uuid
from pathlib import Path

# <Video 1>, <Video 2>, <Video 3> — the order the panel numbers the slots in, so the legend reads
# the way the slots are stacked. Pure primaries: nothing in a photograph looks like these.
COLOURS = [(230, 25, 25), (25, 210, 60), (40, 90, 255)]
NAMES = ["red", "green", "blue"]

# Each pattern is one shape of instruction, and its own idea of what "more" means (`amount`).
PATTERNS = [
    {"id": "circle", "label": "Circle", "colours": 2,
     "note": "A green circle on red, opening out from the middle as the clip runs."},
    {"id": "wipe", "label": "Wipe", "colours": 2,
     "note": "Green sweeps across red, left to right — a straight hand-off."},
    {"id": "split", "label": "Split", "colours": 2,
     "note": "Half and half, with the dividing line travelling slowly across."},
    {"id": "orbit", "label": "Orbit", "colours": 2,
     "note": "A green disc circling the frame over red: one context visiting the other."},
    {"id": "thirds", "label": "Thirds", "colours": 3,
     "note": "Three vertical bands, one per video, sliding sideways."},
    {"id": "blobs", "label": "Blobs", "colours": 3,
     "note": "Two discs wandering over a red field, one green, one blue."},
]
BY_ID = {p["id"]: p for p in PATTERNS}


def spec(pattern: str) -> dict:
    return BY_ID.get(pattern) or PATTERNS[0]


def legend(pattern: str) -> str:
    """The sentence the prompt needs. A mask without one is a video of coloured shapes."""
    p = spec(pattern)
    return ", ".join(f"{NAMES[i]} is <Video {i + 1}>" for i in range(int(p.get("colours", 2))))


def _ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _draw(d, pattern: str, t: float, w: int, h: int, amount: float):
    """One frame of the mask, in pixels. `t` is 0..1 through the clip."""
    from PIL import ImageDraw                    # noqa: F401  (type only, d is the drawer)
    e, s = _ease(t), min(w, h)
    c0, c1, c2 = COLOURS
    if pattern == "wipe":
        d.rectangle([0, 0, w, h], fill=c0)
        d.rectangle([0, 0, int(w * e * amount), h], fill=c1)
    elif pattern == "split":
        d.rectangle([0, 0, w, h], fill=c0)
        x = int(w * (0.5 + (e - 0.5) * 0.6 * amount))
        d.rectangle([x, 0, w, h], fill=c1)
    elif pattern == "orbit":
        d.rectangle([0, 0, w, h], fill=c0)
        r = s * 0.22 * amount
        a = 2.0 * math.pi * e
        cx, cy = w / 2 + math.cos(a) * w * 0.26, h / 2 + math.sin(a) * h * 0.26
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c1)
    elif pattern == "thirds":
        d.rectangle([0, 0, w, h], fill=c0)
        shift = (e - 0.5) * w * 0.3 * amount
        d.rectangle([w / 3 + shift, 0, 2 * w / 3 + shift, h], fill=c1)
        d.rectangle([2 * w / 3 + shift, 0, w, h], fill=c2)
    elif pattern == "blobs":
        d.rectangle([0, 0, w, h], fill=c0)
        r = s * 0.2 * amount
        for colour, phase in ((c1, 0.0), (c2, math.pi)):
            cx = w * (0.5 + 0.3 * math.sin(2.0 * math.pi * e + phase))
            cy = h * (0.5 + 0.22 * math.sin(4.0 * math.pi * e + phase * 1.7))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
    else:                                        # circle
        d.rectangle([0, 0, w, h], fill=c0)
        r = s * (0.08 + 0.42 * e) * amount
        d.ellipse([w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r], fill=c1)


def make(dest_dir: Path, *, pattern: str = "circle", width: int = 848, height: int = 480,
         frames: int = 124, fps: float = 24.0, amount: float = 1.0) -> dict:
    """One mask written as <pattern>-<id>.mp4, with the legend that explains it."""
    from PIL import Image, ImageDraw
    p = spec(pattern)
    width, height = max(16, int(width)), max(16, int(height))
    frames, fps = max(2, int(frames)), max(1.0, float(fps))
    amount = min(1.5, max(0.2, float(amount)))
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{p['id']}-{uuid.uuid4().hex[:6]}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", f"{fps:.4f}", "-i", "-",
           # flat colour, so the encoder is told to keep the edges rather than smooth them away
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "veryfast", str(out)]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(frames):
            img = Image.new("RGB", (width, height), COLOURS[0])
            _draw(ImageDraw.Draw(img), p["id"], i / max(1, frames - 1), width, height, amount)
            pipe.stdin.write(img.tobytes())
    finally:
        pipe.stdin.close()
        err = pipe.stderr.read().decode("utf-8", "replace")
        pipe.wait()
    if pipe.returncode or not out.is_file():
        raise RuntimeError(f"ffmpeg could not write the mask: {err.strip()[:200] or 'no output'}")
    return {"path": str(out), "name": out.name, "pattern": p["id"], "label": p["label"],
            "colours": int(p.get("colours", 2)), "legend": legend(p["id"]),
            "width": width, "height": height, "frames": frames, "fps": round(fps, 3),
            "amount": round(amount, 3)}
