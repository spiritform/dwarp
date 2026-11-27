"""Noise as a layer: the kinds there are, and how each of the three compositors draws them.

Injecting noise is a different lever from raising Denoise. Denoise asks the model to invent more;
noise changes the pixels it is looking at -- and with the warp carrying paint forward, the grain
becomes structure the next frame elaborates. What it is made of therefore matters: per-pixel static
and slow fractal cloud are not the same instruction at all, and the first of them cannot be turned
into the second by blurring it, because a blur takes the grain's own edge away with it.

Nothing here is written to disk for a render. A noise layer used to be an mp4 drawn at the render's
size and length before you could see any of it, so a nudge of the grain dial was ffmpeg over the
whole clip and then a frame extraction after it -- for a layer whose entire content is `random()`.
It is generated at the moment it is composited instead, by whichever compositor is doing the
compositing:

  * the page draws it into a canvas, so the dials move under your hand at the frame rate;
  * the local render draws it with numpy, about a millisecond a frame, inside the per-frame stack;
  * the flatten hands ffmpeg an `-f lavfi` input, so the plate costs no pre-pass either.

Three implementations, so they have to agree about what the dials mean, and this module is where
that is written down: `feature_px` is how big the thing the noise is made of is, and grain is how
far it swings either side of mid grey. Every kind rests at mid grey, so a layer at Grain 0 is a
layer that is not there under overlay or soft light, and the dial opens it either side of that.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

# Each kind is one idea of what grain is, and says whether its own size is worth a dial -- static
# is per-pixel by definition and has nothing to set. Data, so another kind is an entry here, a
# branch in `draw` and a line in `lavfi`: the picker and the layer's dials both draw whatever this
# list says.
PATTERNS = [
    {"id": "static", "label": "Static", "base": "grey", "scale": None,
     "note": "Per-pixel grain, redrawn every frame — the finest there is, and the one the warp "
             "turns into texture fastest."},
    {"id": "cells", "label": "Cells", "base": "grey",
     "scale": {"label": "size", "default": 0.35,
               "note": "How big a block is — a coarse static at the bottom of the dial, slabs "
                       "the warp drags about whole at the top"},
     "note": "The same thing with a far bigger brush: blocks of grey, hard-edged, a new draw on "
             "every frame."},
    {"id": "clouds", "label": "Clouds", "base": "grey",
     "scale": {"label": "size", "default": 0.5,
               "note": "How big the masses are — a boiling froth at the bottom of the dial, a "
                       "handful of shapes moving through the frame at the top"},
     "note": "Fractal noise boiling slowly — soft masses rather than grain, for when you want "
             "the picture pushed around rather than roughened."},
    {"id": "lines", "label": "Lines", "base": "grey",
     "scale": {"label": "width", "default": 0.08,
               "note": "How deep a band is — hairlines at the bottom of the dial, tape "
                       "dropout at the top"},
     "note": "Whole rows at a time, the way a tape drops out. Horizontal structure the warp drags "
             "sideways."},
    # Not noise, and in here anyway: a solid is drawn the same way every one of these is -- no file,
    # no pre-pass, generated at the size of whatever it is going over by whichever compositor is
    # doing the compositing. That machinery is this module, so a second copy of it next door to say
    # "fill it in" would be the larger lie. `flat` is what keeps it out of the noise picker, where
    # a frame of one colour is not an answer to the question being asked.
    {"id": "solid", "label": "Solid", "base": "black", "scale": None, "flat": True,
     "level": {"label": "level", "default": 0.0,
               "note": "Black at the bottom of the dial, white at the top, every grey between"},
     "note": "One flat level, the whole frame. The thing to put underneath a stack that has "
             "nothing of its own to stand on \u2014 noise over black, a pulse over white."},
]
BY_ID = {p["id"]: p for p in PATTERNS}

# How big the thing the noise is made of is, as a share of the short side rather than in pixels, so
# a stack judged at preview size comes out the same when it is flattened four times larger -- the
# rule the blur dial already keeps. Squared, because the bottom of the dial is where the choosing
# happens: at 768 the travel from 0 to half the dial is 3 px to 45, and the top of it is a slab.
FEATURE_MIN, FEATURE_MAX = 0.004, 0.22
DRAW_VERSION = 2        # bump when a draw changes shape, so an old preview is not handed back


def spec(kind: str) -> dict:
    return BY_ID.get(str(kind or "").lower()) or PATTERNS[0]


def catalogue() -> dict:
    """What the picker and the layer's dials draw themselves from."""
    return {"patterns": PATTERNS}


def scale_default(kind: str) -> float:
    """Each kind rests where that kind looks like itself: cells at blocks, lines at hairlines."""
    s = spec(kind)["scale"]
    return float(s["default"]) if s else 0.0


def level_default(kind: str) -> float:
    """How light a flat layer rests at. Black: it is the floor of a stack far more often than the
    ceiling, and white over nothing is the one that looks like a bug."""
    lv = spec(kind).get("level")
    return float(lv["default"]) if lv else 0.0


def settings(row) -> dict:
    """A layer row as the generators read it. One place decides what a missing field means."""
    row = row if isinstance(row, dict) else {}
    kind = spec(row.get("nkind") or row.get("kind"))["id"]
    grain, scale, level = row.get("grain"), row.get("scale"), row.get("level")
    return {"kind": kind,
            "grain": min(1.0, max(0.0, 1.0 if grain is None else float(grain))),
            "scale": scale_default(kind) if scale is None else min(1.0, max(0.0, float(scale))),
            # Only a solid reads this, and only a solid is asked for it: the dial a kind does not
            # have is not drawn, so it has no value of its own to carry either.
            "level": level_default(kind) if level is None else min(1.0, max(0.0, float(level))),
            "seed": int(row.get("seed") or 0) & 0x7FFFFFFF}


def feature_px(scale: float, width: int, height: int) -> int:
    """The size of one block, one band, or the finest detail in a cloud, at this frame size."""
    s = min(1.0, max(0.0, float(scale)))
    return max(2, int(round((FEATURE_MIN + (FEATURE_MAX - FEATURE_MIN) * s * s) * min(width, height))))


def lattice(kind: str, scale: float, width: int, height: int) -> tuple:
    """How many cells across and down a kind is drawn from. Lines are one column by definition:
    the whole row is one value, which is what makes them lines rather than blocks."""
    px = feature_px(scale, width, height)
    if kind == "lines":
        return 1, max(2, int(round(height / px)))
    return max(2, int(round(width / px))), max(2, int(round(height / px)))


CLOUD_STEP = 0.08       # how far a cloud's own clock moves per frame: a boil, not a flicker
CLOUD_GAIN = 1.7        # fbm keeps to the middle of its range; this opens it back out


def _lat(seed: int, o: int, i: int, gh: int, gw: int) -> np.ndarray:
    """One octave's lattice at one step of its clock, drawn from the seed and the step rather than
    carried forward from the frame before -- so any frame can be drawn on its own, which is what
    lets the page draw the frame you scrubbed to without drawing the hundred before it."""
    rng = np.random.default_rng((int(seed) & 0x7FFFFFFF, int(o), max(0, int(i))))
    return rng.random((gh, gw), dtype=np.float32)


def _clouds(width: int, height: int, cells: float, z: float, seed: int) -> np.ndarray:
    """Octaves of value noise, each at half the amplitude and twice the frequency, lerped between
    two draws of each lattice so the field boils instead of flickering."""
    from engine.pulse import _up               # one value-noise upsample in this codebase, not two
    out = np.zeros((height, width), dtype=np.float32)
    amp, tot, g = 1.0, 0.0, max(1.0, float(cells))
    for o in range(4):
        gw = max(2, int(round(g)))
        gh = max(2, int(round(g * height / max(1, width))))
        zo = z * (1.0 + 0.35 * o)              # each octave on its own clock: they part as they boil
        i0 = int(np.floor(zo))
        f = float(zo - i0)
        f = f * f * (3.0 - 2.0 * f)            # cells, not facets
        lat = _lat(seed, o, i0, gh, gw) * (1.0 - f) + _lat(seed, o, i0 + 1, gh, gw) * f
        out += amp * _up(lat, width, height)
        tot += amp
        amp *= 0.5
        g *= 2.0
    return out / max(1e-6, tot)


def draw(kind: str, width: int, height: int, *, i: int = 0, grain: float = 1.0,
         scale: float = -1.0, seed: int = 0, level: float = 0.0) -> np.ndarray:
    """Frame `i` as 0..1 greys, mid grey at rest, drawn from nothing but these arguments."""
    kind = spec(kind)["id"]
    if scale < 0:
        scale = scale_default(kind)
    i = max(0, int(i))
    if kind == "solid":
        # One level, every pixel, every frame. Grain says nothing here: there is nothing to swing,
        # and what a solid is for is to be the thing underneath. Its opacity is how much of it
        # there is and its level is how light it is, which is the whole of it.
        return np.full((height, width), min(1.0, max(0.0, float(level))), dtype=np.float32)
    if kind == "clouds":
        # The lattice is counted off the finest detail rather than set outright: `cells` is how many
        # of the first octave fit across the frame, and four octaves ride on top of it -- so the
        # smallest thing in the picture is about the feature size, and the masses are a handful of
        # them across.
        cells = min(48.0, max(1.0, min(width, height) / (feature_px(scale, width, height) * 4.0)))
        out = 0.5 + (_clouds(width, height, cells, i * CLOUD_STEP, seed) - 0.5) * CLOUD_GAIN
    else:
        gw, gh = (width, height) if kind == "static" else lattice(kind, scale, width, height)
        rng = np.random.default_rng((int(seed) & 0x7FFFFFFF, i))
        lat = rng.random((gh, gw), dtype=np.float32)
        # Nearest-neighbour on purpose: a block with a soft edge is just blurred static, and the
        # point of these is that the warp has something with a shape to carry forward.
        out = lat[np.arange(height) * gh // height][:, np.arange(width) * gw // width]
    return np.clip(0.5 + (out - 0.5) * min(1.0, max(0.0, float(grain))), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ the same thing, as ffmpeg
# The flatten is a filtergraph, so its noise has to be one too: an `-f lavfi` input generating
# frames as they are consumed, which costs no file and no pre-pass. ffmpeg's `noise` filter is the
# only per-pixel random it has that is fast (`geq=random(1)` is minutes over a clip), and on mid
# grey at alls=100 it swings about +/-50 -- so the expansion below is what turns that into the
# black-to-white the dial promises. This is not the numpy draw pixel for pixel; it is the same kind
# at the same size, which is all anyone can ask of noise.
NOISE_SWING = 50.0                     # what alls=100 is worth, measured


def _expand(gain: float) -> str:
    """Open the swing out to what grain asks for. The commas are escaped because this lands in a
    filtergraph, where a comma is the next filter."""
    k = max(0.01, min(12.0, float(gain)))
    return f"lutyuv=y='clip(128+(val-128)*{k:.4f}\\,0\\,255)'"


def lavfi(kind: str, width: int, height: int, fps: float, *, grain: float = 1.0,
          scale: float = -1.0, seed: int = 0, level: float = 0.0) -> str:
    """This layer as a generated ffmpeg input, ready to hand to `-f lavfi -i`."""
    kind = spec(kind)["id"]
    if scale < 0:
        scale = scale_default(kind)
    fps = max(1.0, float(fps))
    if kind == "solid":
        v = int(round(min(1.0, max(0.0, float(level))) * 255.0))
        return (f"color=c=0x{v:02x}{v:02x}{v:02x}:s={width}x{height}:r={fps:.6f},format=gray")
    seed = int(seed) & 0x7FFFFFFF
    gain = (127.0 / NOISE_SWING) * min(1.0, max(0.0, float(grain)))
    src = f"noise=alls=100:allf=t+u:all_seed={seed}"
    if kind == "static":
        return f"color=c=gray:s={width}x{height}:r={fps:.6f},format=gray,{src},{_expand(gain)}"
    gw, gh = lattice(kind, scale, width, height)
    if kind == "clouds":
        # A boil, not a flicker: the lattice is drawn at its own slow rate and the frames between
        # are the average of the ones either side, which is a crossfade from one field to the next.
        # Averaging two independent fields costs about a third of the swing, hence the extra gain.
        rate = max(0.05, fps * CLOUD_STEP)
        span = max(2, int(round(fps / rate)))
        return (f"color=c=gray:s={gw}x{gh}:r={rate:.6f},format=gray,{src},"
                f"fps={fps:.6f},tmix=frames={span},{_expand(gain * 1.45)},"
                f"scale={width}:{height}:flags=bicubic")
    # cells and lines: a tiny lattice blown up with no interpolation at all
    return (f"color=c=gray:s={max(2, gw)}x{gh}:r={fps:.6f},format=gray,{src},{_expand(gain)},"
            f"scale={width}:{height}:flags=neighbor")


# ------------------------------------------------------------------ a clip of it, for the picker
def clip(dest_dir: Path, *, kind: str = "static", width: int = 768, height: int = 768,
         frames: int = 124, fps: float = 24.0, grain: float = 1.0, scale: float = -1.0,
         seed: int = 0) -> dict:
    """One kind written out as an mp4. Nothing in a render asks for this any more -- a noise layer
    is generated where it is composited -- but the picker's cards are little loops, and a loop is a
    file. Named after what is in it, so the same ask is made once."""
    p = spec(kind)
    width, height = max(16, int(width)) & ~1, max(16, int(height)) & ~1
    frames, fps = max(1, int(frames)), max(1.0, float(fps))
    grain = min(1.0, max(0.05, float(grain)))
    if scale < 0:
        scale = scale_default(p["id"])
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    key = f"{p['id']}|{width}x{height}|{frames}|{fps:.4f}|{grain:.3f}|{scale:.3f}|{int(seed)}|{DRAW_VERSION}"
    out = dest_dir / f"noise-{p['id']}-{hashlib.sha1(key.encode()).hexdigest()[:6]}.mp4"
    if not (out.is_file() and out.stat().st_size):
        # ffmpeg writes to a part file and the finished clip is moved into place, because the name
        # is also the cache: anything else asking for this one checks `out.is_file()` and a file
        # being written passes that check happily. os.replace is atomic, so a reader sees either
        # nothing or the whole thing.
        tmp = out.with_name(out.name + ".part.mp4")   # still an mp4: ffmpeg reads the container off the name
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{width}x{height}", "-r", f"{fps:.4f}", "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", "-preset", "veryfast", str(tmp)]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i in range(frames):
                g = draw(p["id"], width, height, i=i, grain=grain, scale=scale, seed=seed)
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
            "grain": round(grain, 3), "scale": round(scale, 3)}


PREVIEW_W, PREVIEW_H, PREVIEW_FRAMES = 256, 144, 32


def preview(dest_dir: Path, kind: str = "static", width: int = PREVIEW_W,
            height: int = PREVIEW_H) -> Path:
    """One kind as a small loop for the picker. Grain is left at the top of the dial here and the
    size at the kind's own default: the card is showing you what the kind looks like, not what you
    have it set to."""
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
