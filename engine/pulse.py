"""A layer that moves on a beat: the signal, and the pictures drawn from it.

A pulse layer is an ordinary clip on the stack -- drawn here at the render's size and length, then
blended, faded, blurred and matted like anything else. The only thing that makes it a pulse is that
its content follows one number per frame, from 0 (between hits) to 1 (on one).

That number comes from a count or from a sound. A BPM is exact and repeatable and matches whatever
you are going to cut this to later; a sound is whatever is actually in the file. Both end up as the
same array, so nothing below here knows or cares which you used.

The base of every pattern is what it looks like *off* the beat, and it is chosen so the layer
disappears under its own blend mode when nothing is happening: black under screen, mid grey under
overlay. A pulse layer at rest should cost you nothing.

Nothing here is written to disk for a render. A pulse layer used to be an mp4 drawn at the render's
size and length before you could see any of it, so a nudge of the scale dial was fractal noise over
the whole clip and then an x264 encode after it -- seconds of waiting, and a few hundred kilobytes
left on disk, for a layer whose entire content is a formula. `draw` below makes frame `i` out of
nothing but its arguments instead, and whichever compositor is doing the compositing calls it:

  * the page draws it into a canvas, so the dials move under your hand at the frame rate;
  * the local render draws it with numpy, inside the per-frame stack, as it paints each frame.

The one thing that is NOT recomputed per frame is the envelope. It costs an audio analysis when the
beat comes from a track, and it is one number per frame either way, so it is worked out once when
the beat changes and carried on the layer from then on. Everything a dial moves is the picture.
"""
from __future__ import annotations

import hashlib
import itertools
import os
import json
import subprocess
from pathlib import Path

import numpy as np

# The patterns made of fractal noise all answer the same two questions, so they share one pair of
# dials. Both are 0..1 and both sit at 0.5, which is where the pattern was before it had any -- so
# an old layer looks the same and the dial has the same room either way. `_fractal` below turns
# them into what the noise actually takes.
FRACTAL_DIALS = {
    "scale": {"label": "scale", "default": 0.5,
              "note": "How big the masses are \u2014 a fine boil at the bottom of the dial, a "
                      "handful of great billows at the top"},
    "boil": {"label": "boil", "default": 0.5,
             "note": "How fast the field churns under the beat \u2014 at the bottom it holds "
                     "still and the hit alone opens it, at the top it rolls while it goes"},
}

# Each pattern is one idea of what a hit looks like. Data, so another one is an entry here and a
# branch in _picture -- the panel draws whatever this list says.
PATTERNS = [
    {"id": "burst", "label": "Burst", "base": "black",
     "note": "Static inside a ring that opens out of the middle on every hit and fades as it goes."},
    {"id": "grain", "label": "Grain", "base": "grey",
     "note": "Static over the whole frame, thickening on the hit and thinning between."},
    {"id": "bloom", "label": "Bloom", "base": "black",
     "note": "A soft disc swelling out of the middle — no texture, just light."},
    {"id": "flash", "label": "Flash", "base": "black",
     "note": "The whole frame, flat. The blend mode does the work: screen to bloom, multiply to stomp."},
    {"id": "rings", "label": "Rings", "base": "black",
     "note": "Rings travelling outward, one launched on each hit, brightest as it leaves."},
    {"id": "blocks", "label": "Blocks", "base": "black",
     "note": "A white square opening from nothing to the whole frame and closing again on every "
             "hit \u2014 hard edges, flat white, no glow."},
    {"id": "circle", "label": "Circle", "base": "black",
     "note": "A disc opening out of the middle on every hit until it is past the corners \u2014 "
             "hard edged and flat white, so it reads as a matte and not as light."},
    {"id": "grid", "label": "Grid", "base": "black",
     "note": "The frame cut into big squares; a handful light on every hit and fade in their own "
             "time, so no two hits land the same."},
    {"id": "slabs", "label": "Slabs", "base": "black",
     "note": "Big hard rectangles thrown across the frame on every hit, holding a moment and "
             "going out \u2014 no two hits in the same place."},
    {"id": "clouds", "label": "Clouds", "base": "black", "dials": FRACTAL_DIALS,
     "note": "Fractal noise, coarse and slowly boiling: whole billows open out of nothing on the "
             "hit and close again."},
    {"id": "smoke", "label": "Smoke", "base": "black", "dials": FRACTAL_DIALS,
     "note": "Smoke billowing out of nothing on every hit until it has taken the whole frame, and "
             "drawing back again \u2014 climbing as it goes, each layer at its own speed."},
    {"id": "wipe", "label": "Wipe", "base": "black",
     "note": "A band of light falling down the frame on every hit \u2014 a straight gradient, no "
             "texture: white at its middle, black either side of it."},
    {"id": "fall", "label": "Fall", "base": "black", "dials": FRACTAL_DIALS,
     "note": "Smoke's own field, carried down the frame by a front instead of sitting still \u2014 "
             "it gathers at the ceiling and leaves by the floor."},
    {"id": "veins", "label": "Veins", "base": "black", "dials": FRACTAL_DIALS,
     "note": "Ridged fractal noise: thick marbled filaments that swell wide on the hit and close "
             "to nothing between."},
]
BY_ID = {p["id"]: p for p in PATTERNS}

DRAW_MAX = 768          # the long side these are painted at before being scaled back up
DRAW_VERSION = 6        # bump when a draw changes shape, so an old preview is not handed back


def spec(kind: str) -> dict:
    return BY_ID.get(str(kind or "").lower()) or PATTERNS[0]


def dials(kind: str) -> dict:
    """Which dials this pattern has, if any. The panel draws whatever this says, as with the
    patterns themselves: a pattern that answers neither question is not asked."""
    return spec(kind).get("dials") or {}


def dial(kind: str, name: str, value=None) -> float:
    """One dial as 0..1: what was asked for, held inside the dial, or its default."""
    d = dials(kind).get(name) or {}
    if value is None or value == "":
        return float(d.get("default", 0.5))
    return min(1.0, max(0.0, float(value)))


SCALE_SPAN = 2.2        # how far the scale dial reaches either way, as a factor on the cell count
BOIL_SPAN = 2.0         # the top of the boil dial, as a factor on the pattern's own rate


def _fractal(scale: float, boil: float, cells: float, rate: float) -> tuple:
    """A pattern's own coarseness and clock, moved by its two dials.

    Both sit at 0.5, and the mapping is exact there -- 2.2 ** 0.0 and 2.0 * 0.5 are 1.0 to the last
    bit -- so a pattern with its dials untouched draws the frames it drew before it had any. That
    is what lets `clip` leave an untouched dial out of the cache key.

    Scale runs backwards through the cell count, because a cell is a mass: fewer cells across the
    frame is a bigger billow, so the dial reads the way the picture does and not the way the
    lattice does. Boil is the clock the field is sampled on, and at 0 it stops -- the field holds
    still and the beat alone opens it.
    """
    return cells * SCALE_SPAN ** (1.0 - 2.0 * scale), rate * BOIL_SPAN * boil


def catalogue() -> dict:
    """What the panel draws its picker from, and which draw it is looking at.

    The version goes out with it because the preview URLs are cached for an hour and read the same
    whatever the pattern now looks like: without something in the address that changes, a redrawn
    pattern is invisible until the cache lets go of it, restart or no restart.
    """
    return {"patterns": PATTERNS, "version": DRAW_VERSION}


# ---------------------------------------------------------------- the signal
def _follow(raw: np.ndarray, fps: float, attack: float, release: float) -> np.ndarray:
    """An envelope follower: snap up, ease down.

    Without it a hit is one frame long -- true to the sound and invisible on screen at 24 fps. The
    rise is kept fast because a slow rise reads as a fade rather than a hit.
    """
    a = float(np.exp(-1.0 / max(1e-3, attack * fps)))
    r = float(np.exp(-1.0 / max(1e-3, release * fps)))
    out = np.empty_like(raw)
    level = 0.0
    for i, v in enumerate(raw):
        k = a if v > level else r
        level = v + (level - v) * k
        out[i] = level
    return out


def _pcm(audio: Path, rate: int = 22050) -> np.ndarray:
    """The file as one channel of floats. ffmpeg does the decoding and the downmix."""
    done = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-map", "0:a:0",
                           "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
                          capture_output=True)
    if done.returncode or not done.stdout:
        err = done.stderr.decode("utf-8", "replace").strip()
        # A video is a fine thing to hand this -- its soundtrack is what gets read -- but a video
        # with no soundtrack is the common way to get here, and ffmpeg says so as "Stream map
        # '0:a:0' matches no streams", which tells you nothing about what to do next.
        if "matches no streams" in err or "does not contain any stream" in err:
            raise RuntimeError(f"there is no sound in {audio.name} — a beat needs a track with audio in it")
        lines = err.splitlines()
        raise RuntimeError(lines[-1] if lines else "no audio in that file")
    return np.frombuffer(done.stdout, dtype="<i2").astype(np.float32) / 32768.0


BANDS = {"all": (20.0, 11000.0), "low": (20.0, 180.0), "mid": (180.0, 2000.0), "high": (2000.0, 11000.0)}


def _from_audio(frames: int, fps: float, audio: Path, band: str, start: float, rate: int = 22050) -> np.ndarray:
    """How loud the sound is under each frame, in the chosen band.

    One window per frame, which is the only resolution that can matter here: the layer cannot change
    faster than the picture does. The level is scaled against its own loud end rather than full
    scale, because a quiet recording should still pulse.
    """
    pcm = _pcm(audio, rate)
    lo, hi = BANDS.get(band, BANDS["all"])
    span = max(1, int(round(rate / max(1.0, fps))))
    off = max(0, int(round(start * rate)))
    win = np.hanning(span).astype(np.float32)
    freqs = np.fft.rfftfreq(span, 1.0 / rate)
    keep = (freqs >= lo) & (freqs <= hi)
    out = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        a = off + i * span
        chunk = pcm[a:a + span]
        if chunk.size < span:
            chunk = np.pad(chunk, (0, span - chunk.size))
        if band == "all":
            out[i] = float(np.sqrt(np.mean(chunk * chunk)))
        else:
            mag = np.abs(np.fft.rfft(chunk * win))
            out[i] = float(np.sqrt(np.mean(mag[keep] ** 2)) / span * 2.0)
    top = float(np.percentile(out, 98.0))
    return out / top if top > 1e-6 else out


def _from_bpm(frames: int, fps: float, bpm: float, division: float, offset: float,
              drift: float = 0.0, rng=None) -> np.ndarray:
    """A hit every beat (or half, or third of one): 1 on the frame it lands, 0 between.

    The shape of the hit is left to the follower, so a count and a sound are smoothed by the same
    thing and a pattern cannot tell them apart.

    `drift` unlocks the grid. At 0 this is a metronome and every hit is identical, which is what
    makes even a field that boils on its own clock pulse like a machine. Turned up, each hit moves
    off the line by up to half a beat, lands with a different weight, and now and then does not
    land at all -- the count is still the count, it has just stopped being quantised. It is drawn
    from the layer's seed, so a take you like comes back.
    """
    step = 60.0 / max(1.0, bpm) / max(0.125, division)      # seconds between hits
    out = np.zeros(frames, dtype=np.float32)
    d = min(1.0, max(0.0, float(drift)))
    loose = rng if (d > 0.0 and rng is not None) else None
    t = float(offset) % step
    while t < frames / fps:
        at, lvl = t, 1.0
        if loose is not None:
            at = t + (float(loose.random()) - 0.5) * d * step      # half a beat either way at full
            lvl = 1.0 - 0.6 * d * float(loose.random())            # and not every one as hard
            if float(loose.random()) < 0.18 * d:                   # every so often, a hit missing
                lvl = 0.0
        i = int(round(at * fps))
        if lvl > 0.0 and 0 <= i < frames:
            out[i] = max(out[i], lvl)       # two that drifted onto one frame are one hit, not two
        t += step
    return out


def envelope(frames: int, fps: float, *, bpm: float = 0.0, division: float = 1.0,
             audio: str = "", band: str = "all", start: float = 0.0,
             attack: float = 0.02, release: float = 0.22, gain: float = 1.0,
             curve: float = 1.0, drift: float = 0.0, seed: int = 0) -> list[float]:
    """One number per frame, 0..1. A sound if there is one, otherwise the count."""
    frames, fps = max(1, int(frames)), max(1.0, float(fps))
    src = Path(str(audio or ""))
    if str(audio or "") and not src.is_file():
        # Falling back to a count here would hand back a pulse that looks right and is not the one
        # asked for: a sound that has gone missing is worth saying out loud.
        raise RuntimeError(f"no sound at {src.name or audio}")
    # A sound has no grid to come off, so drift is a count's dial and nothing else's.
    raw = (_from_audio(frames, fps, src, str(band or "all"), float(start)) if src.is_file()
           else _from_bpm(frames, fps, float(bpm or 120.0), float(division or 1.0), float(start),
                          float(drift or 0.0), np.random.default_rng(int(seed) or 7)))
    env = _follow(raw * float(gain or 1.0), fps, float(attack), float(release))
    env = np.clip(env, 0.0, 1.0) ** max(0.1, float(curve))
    return [round(float(v), 4) for v in env]


# ---------------------------------------------------------------- the picture
def _draw_size(width: int, height: int) -> tuple:
    k = max(1, -(-max(width, height) // DRAW_MAX))
    return max(16, (width // k) & ~1), max(16, (height // k) & ~1)


# ---------------------------------------------------------------- fractal noise
# Value noise written out here rather than pulled in: one random lattice per octave, smoothed on
# the way up to the frame, and crossfaded between two draws of itself so it boils rather than
# flickers. Deliberately coarse -- what these patterns want is thick cloud, not sand.
def _up(lat: np.ndarray, w: int, h: int, ox: float = 0.0, oy: float = 0.0) -> np.ndarray:
    """One lattice stretched over the frame, smoothstepped between its cells and wrapping round."""
    gh, gw = lat.shape
    fx = (np.arange(w, dtype=np.float32) + 0.5) * gw / w - 0.5 + ox
    fy = (np.arange(h, dtype=np.float32) + 0.5) * gh / h - 0.5 + oy
    x0, y0 = np.floor(fx), np.floor(fy)
    tx, ty = (fx - x0).astype(np.float32), (fy - y0).astype(np.float32)
    tx, ty = tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty)   # cells, not facets
    xi, yi = x0.astype(np.int64) % gw, y0.astype(np.int64) % gh
    xj, yj = (xi + 1) % gw, (yi + 1) % gh
    top = lat[yi][:, xi] * (1.0 - tx) + lat[yi][:, xj] * tx
    bot = lat[yj][:, xi] * (1.0 - tx) + lat[yj][:, xj] * tx
    return (top * (1.0 - ty)[:, None] + bot * ty[:, None]).astype(np.float32)


def _face(seed: int, o: int, i: int, gh: int, gw: int) -> np.ndarray:
    """One octave's lattice at one step of its own clock.

    Drawn from the seed and the step rather than carried forward from the frame before. That is the
    whole difference between a layer that has to be rendered and one that can be generated: a
    streamed lattice can only give you frame 300 after it has drawn the 299 before it, which is
    fine for a file written once and useless for a dial you are dragging. (engine/noise._lat is the
    same trick for the same reason.)
    """
    rng = np.random.default_rng((int(seed) & 0x7FFFFFFF, 0x5021, int(o), max(0, int(i))))
    return rng.random((gh, gw), dtype=np.float32)


def _fbm(w: int, h: int, z: float, seed: int, *, octaves: int = 3, cells: float = 3.0,
         drift: float = 0.0, ridged: bool = False, tall: float = 1.0) -> np.ndarray:
    """Octaves of value noise, each at half the amplitude and twice the frequency, as 0..1.

    `cells` is how many lattice cells fit across the frame at the first octave: three is a picture
    made of a handful of masses, which is the whole point of these. `drift` slides each octave
    upward at its own speed, so the layers part as they climb instead of sliding as one sheet.

    `tall` stretches the lattice vertically -- fewer cells down the frame than across it, so the
    masses come out taller than they are wide. An even field reads as foam however it is
    thresholded, because its masses are as wide as they are high.
    """
    out = np.zeros((h, w), dtype=np.float32)
    amp, tot, g = 1.0, 0.0, float(cells)
    for o in range(octaves):
        gw = max(2, int(round(g)))
        gh = max(2, int(round(g * h / max(1, w) / max(0.1, tall))))
        zo = z * (1.0 + 0.35 * o)                  # each octave on its own clock: they part as they boil
        i0 = int(np.floor(zo))
        f = float(zo - i0)
        f = f * f * (3.0 - 2.0 * f)                # cells, not facets
        lat = _face(seed, o, i0, gh, gw) * (1.0 - f) + _face(seed, o, i0 + 1, gh, gw) * f
        v = _up(lat, w, h, oy=-drift * z * (o + 1))
        out += amp * ((1.0 - np.abs(v * 2.0 - 1.0)) if ridged else v)
        tot += amp
        amp *= 0.5
        g *= 2.0
    return out / max(1e-6, tot)


def _cut(v: np.ndarray, cover: float, soft: float) -> np.ndarray:
    """The brightest `cover` of a field, with a soft edge: 0 is black whatever the field is doing.

    A cut at a fixed level would do, if the field held still. A lattice three cells across does
    not -- its own mean wanders from frame to frame, and a fixed level turns that wander into a
    flicker between hits that are meant to match. Asking for a share of the frame instead pins the
    picture to the beat. The quantile is read off every second pixel, which is far more of the
    frame than it needs to land in the right place.
    """
    if cover <= 0.002:
        return np.zeros_like(v)
    thr = float(np.quantile(v[::2, ::2], 1.0 - min(1.0, cover)))
    return np.clip((v - thr) / max(1e-3, soft) + 0.5, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- the bookkeeping, replayed
# Most of these patterns remember something between frames: where a front has travelled to, which
# cells are still lit, how far the band has fallen. That used to live in a `carry` list threaded
# through a loop, which meant frame 300 could only be drawn after the 299 before it -- fine for a
# file written once, useless for a dial you are dragging and a playhead you are scrubbing.
#
# Every one of those is a cheap recurrence starting from a known state at frame 0, so any frame can
# be reached by running it again. A few hundred float operations next to a frame of fractal noise
# is nothing, and it buys the whole difference between a layer that is rendered and one that is
# generated.
def _ks(env, amount: float, n: int) -> list:
    """The envelope through Swing, frame by frame. A short envelope loops, the way a short layer
    does: the length belongs to the render, not to whatever the beat was drawn for."""
    a = min(1.5, max(0.0, float(amount)))
    if not len(env):
        return [0.0] * n
    m = len(env)
    return [max(0.0, min(1.0, float(env[j % m]))) * a for j in range(n)]


def _fronts(ks, step: float, decay: float, gap: float) -> list:
    """Fronts travelling outward from the middle: one launched on a hit, the rest carrying on and
    fading. Burst uses them as a window on static, Rings draws them as light."""
    out: list = []
    for k in ks:
        out = [c for c in ((pos + step, lvl * decay) for pos, lvl in out)
               if c[0] < 1.6 and c[1] > 0.02]
        if k > 0.25 and (not out or out[-1][0] > gap):
            out.append((0.0, k))
    return out


def _disc(ks, soft: float) -> float:
    """Circle's one disc: how big it is now. A hit takes the frame back over rather than nesting a
    second disc inside the first, and the close is quicker than the open."""
    size, way, prev = -1.0, -1.0, 0.0
    for k in ks:
        if k > 0.25 and prev <= 0.25:           # the EDGE of a hit, not every frame of one
            size, way = 0.03, 1.0
        elif way > 0:
            size += 0.11                        # ~9 frames from nothing to past the corners
            if size > 1.0 + soft:               # the corners are covered: turn around here
                way = -1.0
        else:
            size -= 0.30                        # closing is nearly three times quicker
        prev = k
    return size


def _wipe_at(ks, half: float) -> float:
    """Where the middle of Wipe's band has fallen to. It travels on its own clock rather than the
    beat's, launched on each hit, so a fast count starts the fall again from the top."""
    pos, prev = 9.0, 0.0
    for k in ks:
        pos = -half if (k > 0.25 and prev <= 0.25) else pos + 0.16
        prev = k
    return pos


def _fall_at(ks) -> float:
    """The same travel, for the front Fall's noise is seen through."""
    pos, prev = 9.0, 0.0
    for k in ks:
        pos = -0.5 if (k > 0.25 and prev <= 0.25) else pos + 0.16
        prev = k
    return pos


def _cells(ks, n: int, rng) -> list:
    """Grid's cells: a hit lights half of them at random and they fade on their own clock."""
    lev = [0.0] * n
    prev = 0.0
    for k in ks:
        for j in range(n):
            lev[j] *= 0.84
        if k > 0.3 and prev <= 0.3:
            for j in rng.choice(n, size=max(2, n // 2), replace=False):
                lev[int(j)] = k
        prev = k
    return lev


def _slabs(ks, rng) -> list:
    """Slabs' rectangles: a few thrown at the frame on every hit, held and then gone."""
    out: list = []
    for k in ks:
        out = [c for c in ((x0, x1, y0, y1, lvl * 0.87) for x0, x1, y0, y1, lvl in out)
               if c[4] > 0.03]
        if k > 0.3 and (not out or out[-1][4] < k * 0.75):
            for _ in range(2 + int(rng.integers(0, 2))):
                px, py = float(rng.random()), float(rng.random())
                long_, thin = 0.3 + 0.55 * float(rng.random()), 0.12 + 0.3 * float(rng.random())
                dx, dy = (long_, thin) if rng.random() < 0.5 else (thin, long_)
                out.append((px * (1.0 - dx), px * (1.0 - dx) + dx,
                            py * (1.0 - dy), py * (1.0 - dy) + dy, k))
    return out


def _grit(seed: int, tag: int, i: int, h: int, w: int) -> np.ndarray:
    """Per-pixel random for one frame, from the seed and the frame number. Grain and Burst used to
    draw theirs off a stream shared with everything else in the clip; keyed like this, the frame
    you scrubbed to is the frame that gets drawn."""
    return np.random.default_rng((int(seed) & 0x7FFFFFFF, int(tag), max(0, int(i)))
                                 ).random((h, w), dtype=np.float32)


def settings(row) -> dict:
    """A pulse layer row as the generators read it. One place decides what a missing field means.

    The envelope comes WITH the row rather than being worked out here: it is one number per frame
    and it costs an audio analysis when the beat comes from a track, so it is computed once when
    the beat changes and carried on the layer from then on. Everything else here is the picture,
    which is what the dials move.
    """
    row = row if isinstance(row, dict) else {}
    kind = spec(row.get("pulse") or row.get("kind"))["id"]
    out = {"kind": kind,
           "env": [round(max(0.0, min(1.0, float(v))), 4) for v in (row.get("env") or [])],
           "amount": min(1.5, max(0.0, float(row.get("amount", 1.0) or 0.0))),
           # `pinvert` wins: on a LAYER row `invert` already means the matte's level-invert, and
           # the pulse's own flip has to have a name of its own. A settings dict has only `invert`.
           "invert": bool(row.get("pinvert") if row.get("pinvert") is not None else row.get("invert")),
           "seed": int(row.get("seed") or 0) & 0x7FFFFFFF}
    for n in dials(kind):
        out[n] = dial(kind, n, row.get(n))
    return out


# ---------------------------------------------------------------- the picture, frame by frame
def draw(kind: str, width: int, height: int, *, i: int = 0, env=(), amount: float = 1.0,
         invert: bool = False, scale: float = -1.0, boil: float = -1.0,
         seed: int = 0) -> np.ndarray:
    """Frame `i` as 0..1 greys, drawn from nothing but these arguments.

    Nothing is written to disk. The page draws this into a canvas so the dials move under your
    hand, and the render draws it inside the per-frame stack -- the same picture, from the same
    numbers, with no file between them.
    """
    p = spec(kind)
    kind = p["id"]
    w, h = max(2, int(width)), max(2, int(height))
    i = max(0, int(i))
    scale = dial(kind, "scale", None if scale < 0 else scale)
    boil = dial(kind, "boil", None if boil < 0 else boil)
    amount = min(1.5, max(0.0, float(amount)))
    ks = _ks(env, amount, i + 1)
    k = ks[-1]
    # float32 out of every branch, whatever it did inside: a Python float in a np.maximum quietly
    # promotes the whole frame to float64, and the compositors expect one dtype.
    g = np.asarray(_picture(kind, ks, k, w, h, i, amount, scale, boil, int(seed) & 0x7FFFFFFF),
                   dtype=np.float32)
    # Upside down: what was light at the hit is dark, and what was nothing at rest is everything.
    # Since a pattern drawn on black is its own alpha, inverting it inverts that too -- the layer is
    # fully there between hits and is punched out on them. The neutral blend turns over with it:
    # screen for the light one, multiply for this.
    return (1.0 - g).astype(np.float32) if invert else g


def _picture(kind, ks, k, w, h, i, amount, scale, boil, seed) -> np.ndarray:
    y, x = np.ogrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    half = max(1.0, float(np.hypot(cx, cy)))
    r = np.hypot(x - cx, y - cy) / half                  # 0 in the middle, 1 at the corners

    if kind == "grain":
        # Mid grey at rest: overlay and soft light leave the picture alone, and the static swings
        # either side of it rather than only brightening.
        return np.clip(0.5 + (_grit(seed, 1, i, h, w) - 0.5) * k, 0.0, 1.0)
    if kind == "flash":
        return np.full((h, w), k, dtype=np.float32)
    if kind == "bloom":
        # A disc whose edge is soft and whose middle is bright: light, not a shape.
        return np.clip((1.0 - r / max(0.05, k * 1.2)) * k, 0.0, 1.0).astype(np.float32)
    if kind == "burst":
        # The noise texture that expands: static, cut to a ring opening out of the middle. Each hit
        # launches a new front, and what is left of the last one keeps travelling.
        ring = np.zeros((h, w), dtype=np.float32)
        for pos, lvl in _fronts(ks, 0.035, 0.9, 0.12):
            ring = np.maximum(ring, np.clip(1.0 - np.abs(r - pos) / 0.28, 0.0, 1.0) * lvl)
        return (ring * _grit(seed, 2, i, h, w)).astype(np.float32)
    if kind == "blocks":
        # The square IS the beat: its size is the envelope, so it opens from nothing on the hit and
        # closes back to nothing as the hit lets go -- 0 to the whole frame and back, once a beat,
        # for as long as the clip runs. Nothing is carried between frames and nothing travels on a
        # clock of its own, which is what keeps it locked to the count however fast you set it.
        #
        # It does not dim, either. Fading a shape that is already changing size says the same thing
        # twice and leaves the moment it fills the frame a flat grey; here the white stays white
        # and the size does all the talking -- white, black, and the line between them.
        short = max(1.0, min(cx, cy))
        sq = np.maximum(np.abs(x - cx), np.abs(y - cy)) / short
        soft = 2.0 / short                          # a pixel of edge, so it is not a staircase
        full = max(cx, cy) / short                  # the size at which it is past every corner
        return np.clip((k * full - sq) / soft, 0.0, 1.0).astype(np.float32)
    if kind == "circle":
        # One disc, out of the middle, until it is past the corners -- `r` is measured to the
        # corner, so a size of 1.0 is a frame exactly covered. The edge is the one pixel it takes
        # not to be a staircase, and nothing else: white inside, black outside, which is what makes
        # it hold up as a matte where Bloom's falloff does not.
        soft = 2.0 / half
        at = _disc(ks, soft)
        if at <= 0.0:
            return np.zeros((h, w), dtype=np.float32)
        # Flat white, never the hit's own level. The size is what carries the beat here, so dimming
        # the disc as well says it twice and leaves the moment it covers the frame an 88% grey field
        # instead of a white one.
        return np.clip((at - r) / soft, 0.0, 1.0).astype(np.float32)
    if kind == "grid":
        # The frame cut into a few big cells. A hit lights half of them at random and they fade
        # on their own clock, so the pattern never repeats while the count does.
        cols = 4 if w >= h else 3
        rows = max(2, int(round(cols * h / max(1, w))))
        n = cols * rows
        lev = np.asarray(_cells(ks, n, np.random.default_rng(seed or 7)),
                         dtype=np.float32).reshape(rows, cols)
        out = lev[np.arange(h) * rows // h][:, np.arange(w) * cols // w]
        # a gutter between the cells, so what you see is squares and not one lit field
        fx, fy = (np.arange(w) * cols / w) % 1.0, (np.arange(h) * rows / h) % 1.0
        gap = ((fx > 0.03) & (fx < 0.97))[None, :] & ((fy > 0.03) & (fy < 0.97))[:, None]
        return (out * gap).astype(np.float32)
    if kind == "slabs":
        # Rectangles thrown at the frame: a few on every hit, somewhere else each time, held for a
        # moment and then gone. Hard edges and flat white, so what moves is the arrangement.
        fx = (x / max(1.0, w - 1.0)).astype(np.float32)
        fy = (y / max(1.0, h - 1.0)).astype(np.float32)
        out = np.zeros((h, w), dtype=np.float32)
        for x0, x1, y0, y1, lvl in _slabs(ks, np.random.default_rng(seed or 7)):
            band = ((fx >= x0) & (fx <= x1)) & ((fy >= y0) & (fy <= y1))
            out = np.maximum(out, band.astype(np.float32) * lvl)
        return out
    if kind == "wipe":
        # A band of light falling down the frame: linear and untextured, which is the whole of it.
        # Straight across, so every row is one value and what travels is where the white is -- a
        # point on the frame goes black, white as the band arrives, black again once it is past,
        # and white once more on the next hit.
        HALF = 0.45                                 # how deep the band is, as a share of the frame
        at = _wipe_at(ks, HALF)
        if at > 1.0 + HALF:
            return np.zeros((h, w), dtype=np.float32)
        y01 = (np.arange(h, dtype=np.float32) / max(1.0, h - 1.0))[:, None]   # 0 at the ceiling
        # A linear ramp off the middle of the band. Not a curve: a straight fall is what reads as a
        # wipe, where anything eased reads as a glow travelling.
        g = np.clip(1.0 - np.abs(y01 - at) / HALF, 0.0, 1.0)
        return np.repeat(g, w, axis=1).astype(np.float32)
    if kind == "fall":
        # Wipe's travel with Smoke's field inside it. The noise boils on its own clock as it does
        # there; what lets you see it is a band falling from the ceiling, launched on each hit.
        at = _fall_at(ks)
        if at > 1.6:
            return np.zeros((h, w), dtype=np.float32)
        # Negative drift: _fbm slides each octave by -drift*z, so Smoke climbs and this falls. The
        # texture has to move with the front or the band looks like a window dragged over a field
        # that is going the other way.
        cells, rate = _fractal(scale, boil, 3.0, 0.05)
        v = _fbm(w, h, (i + 1.0) * rate, seed, octaves=5, cells=cells, drift=-1.1)
        y01 = (np.arange(h, dtype=np.float32) / max(1.0, h - 1.0))[:, None]
        # Thicker behind the front than ahead of it, so it reads as weather arriving rather than a
        # ruled edge: the leading side is the short ramp, the trailing side the long one.
        lead = np.clip((at - y01) / 0.30, 0.0, 1.0)
        tail = np.clip(1.0 - (at - y01 - 0.45) / 0.60, 0.0, 1.0)
        band = lead * tail
        if float(band.max()) <= 0.0:
            # _cut takes a SHARE of a field, by quantile. Handed nothing it puts the threshold at 0
            # and every pixel comes back at exactly mid grey -- so the frames before the band has
            # arrived would be a flat grey field rather than black.
            return np.zeros((h, w), dtype=np.float32)
        # Cut the field, THEN window it. Cutting the masked field instead lets the black outside the
        # band own the quantile, and what little shows comes back as hard blobs.
        #
        # The cover is the amount, not the beat: letting the hit set it as Smoke does would fade the
        # band out from under its own travel, and the travel is the point here.
        return (_cut(v, 0.72 * min(1.5, max(0.05, amount)), 0.07) * band).astype(np.float32)
    if kind in ("clouds", "smoke", "veins"):
        # The three noise layers. One clock ticking a frame at a time, so the field boils at its
        # own pace and the beat only decides how much of it you are allowed to see: at rest the
        # threshold sits above the brightest part of the noise, which is how these stay black.
        t = i + 1.0
        if kind == "clouds":
            cells, rate = _fractal(scale, boil, 3.0, 0.055)
            return _cut(_fbm(w, h, t * rate, seed, octaves=5, cells=cells), 0.6 * k, 0.07)
        if kind == "smoke":
            # A pulse of it, filling the frame and drawing back. Clouds opens to 60% of the frame
            # and stops; this one takes all of it, so the hit ends on white rather than on a lot of
            # white with gaps -- which is what makes it read as one breath instead of a boil.
            #
            # The 1.2 is headroom. The envelope's own peak is not reliably 1.0 (a follower rounds
            # the top off a hit), so asking for exactly the envelope would leave a frame that never
            # quite closed over.
            #
            # Stretched taller than it is wide, and drifting up: an even field reads as foam.
            cells, rate = _fractal(scale, boil, 2.5, 0.045)
            v = _fbm(w, h, t * rate, seed, octaves=5, cells=cells, drift=1.2, tall=2.6)
            # A wide soft edge, where Clouds takes a narrow one. The threshold IS the silhouette
            # here, so a sharp cut gives torn paper and a wide one gives something with air in it.
            return _cut(v, min(1.0, 1.2 * k), 0.13)
        cells, rate = _fractal(scale, boil, 3.0, 0.06)
        v = _fbm(w, h, t * rate, seed, octaves=4, cells=cells, ridged=True)
        return _cut(v, 0.35 * k, 0.07)
    # rings: the same fronts, drawn as light rather than used as a window
    out = np.zeros((h, w), dtype=np.float32)
    for pos, lvl in _fronts(ks, 0.03, 0.93, 0.1):
        out = np.maximum(out, np.clip(1.0 - np.abs(r - pos) / 0.06, 0.0, 1.0) * lvl)
    return out


def clip(dest_dir: Path, *, kind: str = "burst", width: int = 768, height: int = 768,
         frames: int = 124, fps: float = 24.0, env: list | None = None, amount: float = 1.0,
         seed: int = 0, invert: bool = False, scale=None, boil=None, **sound) -> dict:
    """One pulse layer, written under a name that says what is in it.

    The envelope can be handed in -- the panel has already drawn it for you to look at -- or asked
    for here from the same arguments. Either way it is part of the name, so changing the beat makes
    a different file and changing nothing costs a stat().
    """
    p = spec(kind)
    width, height = max(16, int(width)) & ~1, max(16, int(height)) & ~1
    frames, fps = max(2, int(frames)), max(1.0, float(fps))
    amount = min(1.5, max(0.05, float(amount)))
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    signal = list(env) if env else envelope(frames, fps, seed=seed, **sound)
    # A short envelope loops rather than running out into silence, the way draw() reads it and the
    # way a short layer loops under a long one.
    signal = ([float(signal[j % len(signal)]) for j in range(frames)] if signal
              else [0.0] * frames)

    asked = {"scale": scale, "boil": boil}
    vals = {n: dial(p["id"], n, asked.get(n)) for n in dials(p["id"])}
    # Only a dial that has been moved goes into the name. A pattern at its defaults draws exactly
    # the frames it drew before it had dials (see _fractal), so every clip already on disk is still
    # the right answer to the question being asked -- and a cache is only worth having if asking
    # the same thing twice finds it.
    moved = "".join(f"|{n}{vals[n]:.3f}" for n in sorted(vals) if vals[n] != dial(p["id"], n))

    sig = hashlib.sha1(json.dumps([round(v, 3) for v in signal]).encode()).hexdigest()[:8]
    key = (f"{p['id']}|{width}x{height}|{frames}|{fps:.4f}|{amount:.3f}|{int(seed)}|{sig}"
           f"|{int(invert)}|{DRAW_VERSION}{moved}")
    out = dest_dir / f"pulse-{p['id']}-{hashlib.sha1(key.encode()).hexdigest()[:6]}.mp4"
    if not (out.is_file() and out.stat().st_size):
        dw, dh = _draw_size(width, height)
        # ffmpeg writes to a part file and the finished clip is moved into place, because the name
        # is also the cache: anything else asking for this one checks `out.is_file()` and a file
        # being written passes that check happily. The flatten then opened a clip that was half a
        # clip -- "Invalid data found when processing input", from a layer that was fine seconds
        # later. os.replace is atomic, so a reader sees either nothing or the whole thing.
        # One name per draw, not one per clip. Two requests for the same clip arrive whenever a
        # dial is nudged back to where it was a moment ago, and they used to share a part file:
        # two ffmpegs writing it, and whichever finished first hit os.replace while the other
        # still had the handle open -- which on Windows is a PermissionError, i.e. a 500 with
        # nothing in it. They get a file each now, and the loser throws its copy away.
        tmp = out.with_name(f"{out.name}.{os.getpid()}-{next(_PART)}.part.mp4")   # still an mp4: ffmpeg reads the container off the name
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{dw}x{dh}", "-r", f"{fps:.4f}", "-i", "-"]
        if (dw, dh) != (width, height):
            cmd += ["-vf", f"scale={width}:{height}:flags=bicubic"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "veryfast", str(tmp)]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i in range(frames):
                g = draw(p["id"], dw, dh, i=i, env=signal, amount=amount, invert=invert,
                         seed=int(seed), **vals)
                try:
                    pipe.stdin.write((np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes())
                except OSError:
                    break           # ffmpeg is gone; its stderr below says why, as a RuntimeError
        finally:
            pipe.stdin.close()
            err = pipe.stderr.read().decode("utf-8", "replace")
            pipe.wait()
        if pipe.returncode or not (tmp.is_file() and tmp.stat().st_size):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg could not write the pulse: {err.strip()[:200] or 'no output'}")
        try:
            os.replace(tmp, out)
        except OSError:
            # The name is the contents, so whatever is already there is this clip: keep it and
            # drop the copy. Only a failure with nothing in its place is worth raising.
            tmp.unlink(missing_ok=True)
            if not (out.is_file() and out.stat().st_size):
                raise
    return {"path": str(out), "name": out.name, "pattern": p["id"], "label": p["label"],
            "base": ("white" if invert else p["base"]), "invert": bool(invert),
            "width": width, "height": height, "frames": frames,
            "fps": round(fps, 3), "amount": round(amount, 3), "env": [round(v, 3) for v in signal],
            "dials": {n: round(v, 3) for n, v in vals.items()}}


_PART = itertools.count()   # one part-file name per draw; see clip() below


PREVIEW_W, PREVIEW_H, PREVIEW_FRAMES, PREVIEW_BPM = 256, 144, 48, 120.0


def preview(dest_dir: Path, kind: str = "burst", width: int = PREVIEW_W, height: int = PREVIEW_H) -> Path:
    """One pattern as a small loop at a steady count, for the picker. A pattern you cannot see
    before you pick it is a guess."""
    p = spec(kind)
    width = max(32, min(640, int(width))) & ~1
    height = max(32, min(640, int(height))) & ~1
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{p['id']}-{width}x{height}-v{DRAW_VERSION}.mp4"
    if out.is_file() and out.stat().st_size:
        return out
    made = clip(dest_dir, kind=p["id"], width=width, height=height, frames=PREVIEW_FRAMES,
                fps=24.0, amount=1.0, bpm=PREVIEW_BPM, division=1.0)
    Path(made["path"]).replace(out)
    return out
