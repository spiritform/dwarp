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

import hashlib
import math
import subprocess
from pathlib import Path

# <Video 1>, <Video 2>, <Video 3> — the order the panel numbers the slots in, so the legend reads
# the way the slots are stacked. Pure primaries: nothing in a photograph looks like these.
COLOURS = [(230, 25, 25), (25, 210, 60), (40, 90, 255)]
NAMES = ["red", "green", "blue"]

# The same patterns in black and white, for use as a luma matte: out, in, and -- for a pattern
# that speaks for three regions -- a mid grey the Black / Mid / White levels can single out.
GREYS = [(0, 0, 0), (255, 255, 255), (128, 128, 128)]

# Each pattern is one shape of instruction, and its own idea of what "more" means (`amount`).
PATTERNS = [
    {"id": "circle", "label": "Circle", "colours": 2,
     "note": "A circle opening out from the middle as the clip runs."},
    {"id": "wipe", "label": "Wipe", "colours": 2,
     "note": "A straight sweep across the frame, left to right — a hand-off."},
    {"id": "split", "label": "Split", "colours": 2,
     "note": "Half and half, the dividing line drifting slowly either side of the middle."},
    {"id": "orbit", "label": "Orbit", "colours": 2,
     "note": "A disc circling the frame: one layer visiting the other."},
    {"id": "thirds", "label": "Thirds", "colours": 3,
     "note": "Three vertical bands sliding sideways — out, mid and in, so Black / Mid / White can pick one."},
    {"id": "blobs", "label": "Blobs", "colours": 3,
     "note": "Two discs circling opposite each other, one white and one mid grey."},
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


def _draw(d, pattern: str, t: float, w: int, h: int, amount: float, palette=None):
    """One frame of the mask, in pixels. `t` is 0..1 through the clip."""
    from PIL import ImageDraw                    # noqa: F401  (type only, d is the drawer)
    e, s = _ease(t), min(w, h)
    c0, c1, c2 = palette or COLOURS
    if pattern == "wipe":
        d.rectangle([0, 0, w, h], fill=c0)
        d.rectangle([0, 0, int(w * e * amount), h], fill=c1)
    elif pattern == "split":
        d.rectangle([0, 0, w, h], fill=c0)
        x = int(w * (0.5 + (e - 0.5) * 0.3 * amount))     # 35/65 .. 65/35: a half each, drifting
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
        r = s * 0.15 * amount
        a = 2.0 * math.pi * e
        # one drift for both, so the pair roams without the gap between them ever changing. Kept
        # small: it is the only thing that can carry a disc off the edge of a tall frame.
        dx, dy = w * 0.035 * math.sin(a * 0.5), h * 0.03 * math.cos(a * 0.5)
        for colour, phase in ((c1, 0.0), (c2, math.pi)):
            cx = w * 0.5 + dx + math.cos(a + phase) * w * 0.32
            cy = h * 0.5 + dy + math.sin(a + phase) * h * 0.26
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
    else:                                        # circle
        d.rectangle([0, 0, w, h], fill=c0)
        r = s * (0.08 + 0.42 * e) * amount
        d.ellipse([w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r], fill=c1)


# The picker shows every pattern moving at once. These never change, so they are drawn once and
# kept: small, flat colour, a few KB each.
PREVIEW_W, PREVIEW_H, PREVIEW_FRAMES = 256, 144, 48


def preview(dest_dir: Path, pattern: str = "circle", width: int = PREVIEW_W,
            height: int = PREVIEW_H, amount: float = 1.0) -> Path:
    """One pattern as a small loop, drawn on first ask and reused after.

    It takes a size because a mask is a map of the frame: a circle opening in 16:9 is not the circle
    you get in 9:16, and the picker should show the one you are about to render. h264 at yuv420p
    wants even sides, so the box is nudged to even rather than handed on and failing in ffmpeg.
    """
    p = spec(pattern)
    amount = min(1.5, max(0.2, float(amount)))
    width = max(32, min(640, int(width))) & ~1
    height = max(32, min(640, int(height))) & ~1
    dest_dir.mkdir(parents=True, exist_ok=True)
    # "matte" in the name, so the colour previews cached before these were black and white are not
    # handed back for the pattern they no longer look like.
    out = dest_dir / f"{p['id']}-matte-{width}x{height}-{int(round(amount * 100)):03d}.mp4"
    if out.is_file() and out.stat().st_size:
        return out
    made = make(dest_dir, pattern=p["id"], width=width, height=height,
                frames=PREVIEW_FRAMES, fps=24.0, amount=amount)
    Path(made["path"]).replace(out)
    return out


# Bump when _draw changes shape, so a mask cached under the old geometry is not handed back.
DRAW_VERSION = 2
DRAW_MAX = 512          # the long side these are actually painted at, before being blown back up


def _draw_size(width: int, height: int) -> tuple:
    """The size to paint at: the render's own, divided by whatever whole number brings the long side
    under DRAW_MAX. A whole number keeps the aspect exact, so the geometry is identical."""
    k = max(1, -(-max(width, height) // DRAW_MAX))
    return max(16, width // k), max(16, height // k), k


def make(dest_dir: Path, *, pattern: str = "circle", width: int = 848, height: int = 480,
         frames: int = 124, fps: float = 24.0, amount: float = 1.0, grey: bool = True) -> dict:
    """One mask, written under a name that says what is in it, so the same ask is not drawn twice.

    Two things made this take twelve seconds for a 768x1344 clip. The name held a random id, so a
    pattern you had already picked was redrawn from nothing every time; it is a hash of the request
    now, and a second ask for the same mask is a stat() call. And every frame was painted at the full
    render size -- a megapixel of flat colour per frame, 380 MB of raw video through a pipe. These are
    flat regions with hard edges, so they are painted small and scaled back up with nearest-neighbour,
    which reproduces hard edges exactly. (Anything smoother would invent colours along every edge that
    are neither red nor green, which is worse than blocky: the colour *is* the instruction.)
    """
    from PIL import Image, ImageDraw
    p = spec(pattern)
    width, height = max(16, int(width)), max(16, int(height))
    frames, fps = max(2, int(frames)), max(1.0, float(fps))
    amount = min(1.5, max(0.2, float(amount)))
    dest_dir.mkdir(parents=True, exist_ok=True)
    palette = GREYS if grey else COLOURS
    key = f"{p['id']}|{width}x{height}|{frames}|{fps:.4f}|{amount:.3f}|{'g' if grey else 'c'}|{DRAW_VERSION}"
    out = dest_dir / f"{p['id']}-{'matte-' if grey else ''}{hashlib.sha1(key.encode()).hexdigest()[:6]}.mp4"
    dw, dh, k = _draw_size(width, height)
    if not out.is_file():
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{dw}x{dh}", "-r", f"{fps:.4f}", "-i", "-"]
        if (dw, dh) != (width, height):
            cmd += ["-vf", f"scale={width}:{height}:flags=neighbor"]
        # flat colour, so the encoder is told to keep the edges rather than smooth them away
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "veryfast", str(out)]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i in range(frames):
                img = Image.new("RGB", (dw, dh), palette[0])
                _draw(ImageDraw.Draw(img), p["id"], i / max(1, frames - 1), dw, dh, amount, palette)
                pipe.stdin.write(img.tobytes())
        finally:
            pipe.stdin.close()
            err = pipe.stderr.read().decode("utf-8", "replace")
            pipe.wait()
        if pipe.returncode or not out.is_file():
            raise RuntimeError(f"ffmpeg could not write the mask: {err.strip()[:200] or 'no output'}")
    return {"path": str(out), "name": out.name, "pattern": p["id"], "label": p["label"],
            "colours": int(p.get("colours", 2)), "legend": legend(p["id"]), "grey": bool(grey),
            "width": width, "height": height, "frames": frames, "fps": round(fps, 3),
            "amount": round(amount, 3)}
