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
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

# Each pattern is one idea of what a hit looks like. Data, so another one is an entry here and a
# branch in _frame -- the panel draws whatever this list says.
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
]
BY_ID = {p["id"]: p for p in PATTERNS}

DRAW_MAX = 768          # the long side these are painted at before being scaled back up
DRAW_VERSION = 1        # bump when _frame changes shape, so an old cache is not handed back


def spec(kind: str) -> dict:
    return BY_ID.get(str(kind or "").lower()) or PATTERNS[0]


def catalogue() -> dict:
    """What the panel draws its picker from."""
    return {"patterns": PATTERNS}


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
        err = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(err[-1] if err else "no audio in that file")
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


def _from_bpm(frames: int, fps: float, bpm: float, division: float, offset: float) -> np.ndarray:
    """A hit every beat (or half, or third of one): 1 on the frame it lands, 0 between.

    The shape of the hit is left to the follower, so a count and a sound are smoothed by the same
    thing and a pattern cannot tell them apart.
    """
    step = 60.0 / max(1.0, bpm) / max(0.125, division)      # seconds between hits
    out = np.zeros(frames, dtype=np.float32)
    t = float(offset) % step
    while t < frames / fps:
        out[min(frames - 1, int(round(t * fps)))] = 1.0
        t += step
    return out


def envelope(frames: int, fps: float, *, bpm: float = 0.0, division: float = 1.0,
             audio: str = "", band: str = "all", start: float = 0.0,
             attack: float = 0.02, release: float = 0.22, gain: float = 1.0,
             curve: float = 1.0) -> list[float]:
    """One number per frame, 0..1. A sound if there is one, otherwise the count."""
    frames, fps = max(1, int(frames)), max(1.0, float(fps))
    src = Path(str(audio or ""))
    if str(audio or "") and not src.is_file():
        # Falling back to a count here would hand back a pulse that looks right and is not the one
        # asked for: a sound that has gone missing is worth saying out loud.
        raise RuntimeError(f"no sound at {src.name or audio}")
    raw = (_from_audio(frames, fps, src, str(band or "all"), float(start)) if src.is_file()
           else _from_bpm(frames, fps, float(bpm or 120.0), float(division or 1.0), float(start)))
    env = _follow(raw * float(gain or 1.0), fps, float(attack), float(release))
    env = np.clip(env, 0.0, 1.0) ** max(0.1, float(curve))
    return [round(float(v), 4) for v in env]


# ---------------------------------------------------------------- the picture
def _draw_size(width: int, height: int) -> tuple:
    k = max(1, -(-max(width, height) // DRAW_MAX))
    return max(16, (width // k) & ~1), max(16, (height // k) & ~1)


def _frame(kind: str, e: float, w: int, h: int, amount: float, rng, carry: list) -> np.ndarray:
    """One frame as 0..1 greys, at the drawing size."""
    y, x = np.ogrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    half = max(1.0, float(np.hypot(cx, cy)))
    r = np.hypot(x - cx, y - cy) / half                  # 0 in the middle, 1 at the corners
    k = max(0.0, min(1.0, e)) * amount

    if kind == "grain":
        # Mid grey at rest: overlay and soft light leave the picture alone, and the static swings
        # either side of it rather than only brightening.
        return np.clip(0.5 + (rng.random((h, w), dtype=np.float32) - 0.5) * k, 0.0, 1.0)
    if kind == "flash":
        return np.full((h, w), k, dtype=np.float32)
    if kind == "bloom":
        # A disc whose edge is soft and whose middle is bright: light, not a shape.
        return np.clip((1.0 - r / max(0.05, k * 1.2)) * k, 0.0, 1.0).astype(np.float32)
    if kind == "burst":
        # The noise texture that expands: static, cut to a ring opening out of the middle. Each hit
        # launches a new front, and what is left of the last one keeps travelling.
        carry[:] = [c for c in ((pos + 0.035, lvl * 0.9) for pos, lvl in carry) if c[0] < 1.6 and c[1] > 0.02]
        if k > 0.25 and (not carry or carry[-1][0] > 0.12):
            carry.append((0.0, k))
        ring = np.zeros((h, w), dtype=np.float32)
        for pos, lvl in carry:
            ring = np.maximum(ring, np.clip(1.0 - np.abs(r - pos) / 0.28, 0.0, 1.0) * lvl)
        return (ring * rng.random((h, w), dtype=np.float32)).astype(np.float32)
    # rings: the same fronts, drawn as light rather than used as a window
    carry[:] = [c for c in ((pos + 0.03, lvl * 0.93) for pos, lvl in carry) if c[0] < 1.6 and c[1] > 0.02]
    if k > 0.25 and (not carry or carry[-1][0] > 0.1):
        carry.append((0.0, k))
    out = np.zeros((h, w), dtype=np.float32)
    for pos, lvl in carry:
        out = np.maximum(out, np.clip(1.0 - np.abs(r - pos) / 0.06, 0.0, 1.0) * lvl)
    return out


def clip(dest_dir: Path, *, kind: str = "burst", width: int = 768, height: int = 768,
         frames: int = 124, fps: float = 24.0, env: list | None = None, amount: float = 1.0,
         seed: int = 0, invert: bool = False, **sound) -> dict:
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
    signal = list(env) if env else envelope(frames, fps, **sound)
    signal = [float(v) for v in signal[:frames]] + [0.0] * max(0, frames - len(signal))

    sig = hashlib.sha1(json.dumps([round(v, 3) for v in signal]).encode()).hexdigest()[:8]
    key = f"{p['id']}|{width}x{height}|{frames}|{fps:.4f}|{amount:.3f}|{int(seed)}|{sig}|{int(invert)}|{DRAW_VERSION}"
    out = dest_dir / f"pulse-{p['id']}-{hashlib.sha1(key.encode()).hexdigest()[:6]}.mp4"
    if not (out.is_file() and out.stat().st_size):
        dw, dh = _draw_size(width, height)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{dw}x{dh}", "-r", f"{fps:.4f}", "-i", "-"]
        if (dw, dh) != (width, height):
            cmd += ["-vf", f"scale={width}:{height}:flags=bicubic"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "veryfast", str(out)]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        rng = np.random.default_rng(int(seed) or 7)
        carry: list = []
        try:
            for i in range(frames):
                g = _frame(p["id"], signal[i], dw, dh, amount, rng, carry)
                # Upside down: what was light at the hit is dark, and what was nothing at rest is
                # everything. Since a pattern drawn on black is its own alpha, inverting it inverts
                # that too -- the layer is fully there between hits and is punched out on them. The
                # neutral blend turns over with it: screen for the light one, multiply for this.
                if invert:
                    g = 1.0 - g
                pipe.stdin.write((np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes())
        finally:
            pipe.stdin.close()
            err = pipe.stderr.read().decode("utf-8", "replace")
            pipe.wait()
        if pipe.returncode or not out.is_file():
            out.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg could not write the pulse: {err.strip()[:200] or 'no output'}")
    return {"path": str(out), "name": out.name, "pattern": p["id"], "label": p["label"],
            "base": ("white" if invert else p["base"]), "invert": bool(invert),
            "width": width, "height": height, "frames": frames,
            "fps": round(fps, 3), "amount": round(amount, 3), "env": [round(v, 3) for v in signal]}


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
