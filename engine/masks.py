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

# The same patterns in black and white, for use as a luma matte. A matte is read as a transition:
# it starts white -- the layer is there -- and ends black, with invert on the card to turn that
# round. So white leads, and what the pattern draws over it is black. With three regions the middle
# one is the mid grey the Black / Mid / White levels can single out, which is why the two orders
# differ: a three-region pattern walks c0 -> c1 -> c2 and has to pass through the grey on the way.
GREYS = [(255, 255, 255), (0, 0, 0), (128, 128, 128)]
GREYS3 = [(255, 255, 255), (128, 128, 128), (0, 0, 0)]

# Each pattern is one shape of instruction, and its own idea of what "more" means (`amount`).
# Every one of these runs from one end to the other over the clip: as a matte that is white to
# black, the layer there and then gone. A pattern that ends where it started says nothing -- it is
# a loop, and a matte that loops is a flicker.
PATTERNS = [
    {"id": "circle", "label": "Circle", "colours": 2,
     "note": "A disc opening out of the middle until it has swallowed the corners."},
    {"id": "wipe", "label": "Wipe", "colours": 2,
     "note": "A straight sweep across the frame, left to right — a hand-off."},
    {"id": "split", "label": "Split", "colours": 2,
     "note": "Two bands closing in from the edges until they meet in the middle."},
    {"id": "polygon", "label": "Polygon", "colours": 2,
     "note": "A six-sided shape opening out of the middle, turning as it grows."},
    {"id": "orbit", "label": "Orbit", "colours": 2,
     "note": "A disc going round the frame and swelling as it goes, until it has taken all of it."},
    {"id": "blobs", "label": "Blobs", "colours": 3,
     "note": "Discs arriving one after another and running together, every other one mid grey."},
    {"id": "thirds", "label": "Thirds", "colours": 3,
     "note": "The whole frame in three steps — white, then mid grey, then black, so the levels can "
             "single out any one of them."},
    {"id": "clouds", "label": "Clouds", "colours": 2, "smooth": True,
     "note": "Fractal noise eating the frame away: thick where the cloud is heaviest first, then "
             "everywhere."},
    {"id": "ink", "label": "Ink", "colours": 2, "smooth": True,
     "note": "A blot spreading out of the middle with its edge broken up by noise — round at heart "
             "and ragged at the rim."},
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


def _noise(img, pattern: str, e: float, w: int, h: int, amount: float, c0, c1, st: dict):
    """The two patterns that are a field rather than a shape, painted straight into the frame.

    The value noise is the pulse layers' own -- one module, one kind of noise, and a cloud here
    boils at the same rate as a cloud there. `st` carries the lattices from frame to frame, which is
    what makes it boil instead of flicker; it belongs to one clip and is made in make().
    """
    import numpy as np
    from PIL import Image

    from engine.pulse import _fbm              # the same value noise the pulse patterns are built on

    if "rng" not in st:
        st["rng"] = np.random.default_rng(11)
    v = _fbm(st, w, h, e * 4.0, st["rng"], octaves=4, cells=3.0)
    if pattern == "ink":
        y, x = np.ogrid[0:h, 0:w]
        far = max(1.0, math.hypot(w, h) / 2.0)
        r = np.hypot(x - (w - 1) / 2.0, y - (h - 1) / 2.0) / far
        # the blot is a circle the noise has chewed: how far out it has got, plus a wobble
        gone = (r + (v - 0.5) * 0.8) < e * 1.35 * amount
    else:
        # a share of the frame rather than a level, so the dissolve is even however the field drifts
        cover = min(1.0, e * amount)
        gone = (v <= float(np.quantile(v[::2, ::2], cover))) if cover > 0.002 else np.zeros_like(v, bool)
    rgb = np.where(gone[..., None], np.array(c1, np.uint8), np.array(c0, np.uint8)).astype(np.uint8)
    img.paste(Image.fromarray(rgb))


def _draw(img, d, pattern: str, t: float, w: int, h: int, amount: float, palette=None, st=None):
    """One frame of the mask, in pixels. `t` is 0..1 through the clip, and every pattern is
    somewhere on its way from all of c0 to all of c1 as that runs."""
    e, s = _ease(t), min(w, h)
    c0, c1, c2 = palette or COLOURS
    far = math.hypot(w, h) / 2.0                 # the middle to a corner: this big covers everything
    if pattern in ("clouds", "ink"):
        return _noise(img, pattern, e, w, h, amount, c0, c1, st if st is not None else {})
    if pattern == "wipe":
        d.rectangle([0, 0, w, h], fill=c0)
        d.rectangle([0, 0, int(w * e * amount), h], fill=c1)
    elif pattern == "split":
        # in from both edges until they meet: a half each at the end, not a line that wandered
        d.rectangle([0, 0, w, h], fill=c0)
        x = w * 0.5 * min(1.0, e * amount)
        d.rectangle([0, 0, int(x), h], fill=c1)
        d.rectangle([int(w - x), 0, w, h], fill=c1)
    elif pattern == "polygon":
        d.rectangle([0, 0, w, h], fill=c0)
        r, a = far * e * amount, math.pi * e * 0.6          # it turns a little as it opens
        if r > 0.5:
            d.polygon([(w / 2 + r * math.cos(a + k * math.pi / 3),
                        h / 2 + r * math.sin(a + k * math.pi / 3)) for k in range(6)], fill=c1)
    elif pattern == "orbit":
        # still one layer visiting the other, but the visit ends with it having taken the place
        d.rectangle([0, 0, w, h], fill=c0)
        r = s * 0.14 * amount + far * e * e * amount
        a = 2.0 * math.pi * e
        cx = w / 2 + math.cos(a) * w * 0.26 * (1 - e)       # it comes home as it grows
        cy = h / 2 + math.sin(a) * h * 0.26 * (1 - e)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c1)
    elif pattern == "thirds":
        # three steps in time rather than three bands in space: the whole frame white, then the mid
        # grey, then black. Not eased -- a step is the point, and an eased step is a slow step.
        d.rectangle([0, 0, w, h], fill=(c0, c1, c2)[min(2, int(t * 3.0))])
    elif pattern == "blobs":
        # discs arriving one after another and running together, every other one the mid grey. The
        # places are fixed and the arrivals are staggered, so the frame fills in rather than throbs.
        d.rectangle([0, 0, w, h], fill=c0)
        spots = [(0.30, 0.34), (0.68, 0.30), (0.50, 0.58), (0.22, 0.72), (0.78, 0.70), (0.50, 0.14)]
        for n, (fx, fy) in enumerate(spots):
            start = n / (len(spots) + 1.0)
            k = max(0.0, min(1.0, (e - start) / max(1e-3, 1.0 - start)))
            r = far * 0.62 * k * amount
            if r <= 0.5:
                continue
            cx, cy = w * fx, h * fy
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c2 if n % 2 else c1)
    else:                                        # circle
        d.rectangle([0, 0, w, h], fill=c0)
        r = far * e * amount                     # all the way out: at the end the corners are gone
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
    out = dest_dir / f"{p['id']}-matte-{width}x{height}-{int(round(amount * 100)):03d}-v{DRAW_VERSION}.mp4"
    if out.is_file() and out.stat().st_size:
        return out
    made = make(dest_dir, pattern=p["id"], width=width, height=height,
                frames=PREVIEW_FRAMES, fps=24.0, amount=amount)
    Path(made["path"]).replace(out)
    return out


# Bump when _draw changes shape, so a mask cached under the old geometry is not handed back.
DRAW_VERSION = 3
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
    # Three regions walk c0 -> c1 -> c2 and need the grey in the middle of that walk; two regions
    # want white then black with the grey left over and unused.
    palette = (GREYS3 if int(p.get("colours", 2)) >= 3 else GREYS) if grey else COLOURS
    key = f"{p['id']}|{width}x{height}|{frames}|{fps:.4f}|{amount:.3f}|{'g' if grey else 'c'}|{DRAW_VERSION}"
    out = dest_dir / f"{p['id']}-{'matte-' if grey else ''}{hashlib.sha1(key.encode()).hexdigest()[:6]}.mp4"
    dw, dh, k = _draw_size(width, height)
    if not out.is_file():
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{dw}x{dh}", "-r", f"{fps:.4f}", "-i", "-"]
        if (dw, dh) != (width, height):
            # Nearest-neighbour everywhere else: these are flat regions with hard edges, and
            # anything smoother invents colours along every edge that are neither red nor green --
            # and the colour *is* the instruction. A noise matte has no legend to protect and an
            # edge in every pixel, so as a matte it is brought up smoothly instead.
            smooth = bool(p.get("smooth")) and grey
            cmd += ["-vf", f"scale={width}:{height}:flags={'bicubic' if smooth else 'neighbor'}"]
        # flat colour, so the encoder is told to keep the edges rather than smooth them away
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "veryfast", str(out)]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        st: dict = {}                            # the noise patterns' lattices, one clip's worth
        try:
            for i in range(frames):
                img = Image.new("RGB", (dw, dh), palette[0])
                _draw(img, ImageDraw.Draw(img), p["id"], i / max(1, frames - 1), dw, dh, amount,
                      palette, st)
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
