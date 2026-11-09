"""Authored motion: the same path JSON as engine/tracks.py, drawn from a formula instead of measured.

Tracks measured off a video turn out to be a motion *signature*, not a structural match — the paths land
where the source clip's features were, which in another picture is usually nothing, and the model still
moves. Once that is what they are for, a video is a roundabout way to ask for "drift left" or "turn
slowly": the clip is only there to be averaged into a few dozen curves.

So this file draws the curves directly. Every kind here is one analytic displacement over normalised
time, evaluated on a spread of seed points and handed to tracks.to_json unchanged, which keeps the one
format fact that matters: a bare list of tracks, one {x, y} per output frame, every track the same
length, in the pixels of the frame about to be generated.

Two things are deliberate. The motion is **eased** at both ends (smoothstep), because a field that
starts at full speed on frame one reads as a jump cut to anything trained on real footage. And the whole
cloud of paths is **fitted into the frame** afterwards rather than clamped to it: a clamped point stops
dead at the edge and stops being motion at all, while scaling the set about the centre keeps the shape
of the field and only costs a little reach.
"""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import numpy as np

from engine import tracks

MARGIN = 2          # px kept clear of the edge, so nothing sits exactly on the border
FIT = 0.98          # the cloud is fitted to this much of the frame: a hair of air around it

# What the panel offers. Each entry says which dials it reads, so the page builds its own controls from
# this list and a new field needs an entry here and no page edit (the same bargain as workflows.json).
KINDS = [
    {"id": "pan", "label": "Pan", "angle": True,
     "note": "Everything drifts one way, together. Reads as a camera move.",
     "amount": {"min": 0.05, "max": 1.0, "step": 0.05, "default": 0.45}},
    {"id": "zoom", "label": "Push / pull", "angle": False,
     "note": "Out from the centre, or in towards it. Negative pulls back.",
     "amount": {"min": -1.0, "max": 1.0, "step": 0.05, "default": 0.45}},
    {"id": "orbit", "label": "Orbit", "angle": False,
     "note": "The whole frame turns about its centre. Negative turns the other way.",
     "amount": {"min": -1.0, "max": 1.0, "step": 0.05, "default": 0.4}},
    {"id": "swirl", "label": "Swirl", "angle": False,
     "note": "A turn strongest at the centre, dying off outwards — a vortex.",
     "amount": {"min": -1.0, "max": 1.0, "step": 0.05, "default": 0.5}},
    {"id": "wave", "label": "Wave", "angle": True,
     "note": "Travel one way with a sway across it, each path a little behind the last.",
     "amount": {"min": 0.05, "max": 1.0, "step": 0.05, "default": 0.5}},
    {"id": "drift", "label": "Drift", "angle": False, "seed": True,
     "note": "No common direction: every path wanders its own smooth way. The seed changes it.",
     "amount": {"min": 0.05, "max": 1.0, "step": 0.05, "default": 0.5}},
]
BY_ID = {k["id"]: k for k in KINDS}


def spec(kind: str) -> dict:
    return BY_ID.get(kind) or KINDS[0]


# ------------------------------------------------------------------ the seeds
def seeds(points: int, width: int, height: int) -> np.ndarray:
    """`points` starting positions, spread over the frame.

    A drawn field is read as a pattern, so the seeds want to be a clean lattice: whenever the count
    factors into a pair of sides close to the frame's own aspect (8, 16, 24 and 48 all do), that is the
    grid, and every cell is used. Otherwise it falls back to the nearest square-ish grid and picks from
    it farthest-first, which spreads the points but leaves a gap where a row came up short."""
    points = max(1, int(points))
    aspect = max(0.2, width / max(1, height))
    pair = min(((abs(math.log((c / (points // c)) / aspect)), c) for c in range(1, points + 1)
                if points % c == 0), default=(9.9, 1))
    if pair[0] < 0.65:                                   # a lattice within ~2x of the frame shape
        cols, rows = pair[1], points // pair[1]
    else:
        cols = max(1, int(round(math.sqrt(points * aspect))))
        rows = max(1, math.ceil(points / cols))
    xs = (np.arange(cols) + 0.5) / cols * width
    ys = (np.arange(rows) + 0.5) / rows * height
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2).astype(np.float64)
    if len(grid) <= points:
        return grid
    centre = np.array([width / 2.0, height / 2.0])
    taken = [int(np.argmin(((grid - centre) ** 2).sum(1)))]
    far = ((grid - grid[taken[0]]) ** 2).sum(1)
    while len(taken) < points:
        i = int(np.argmax(far))
        taken.append(i)
        far = np.minimum(far, ((grid - grid[i]) ** 2).sum(1))
    return grid[taken]


def _ease(length: int) -> np.ndarray:
    """Normalised time, smoothstepped: still at both ends, fastest in the middle."""
    t = np.linspace(0.0, 1.0, max(2, int(length)))
    return t * t * (3.0 - 2.0 * t)


def _rotate(pts: np.ndarray, centre: np.ndarray, ang) -> np.ndarray:
    """pts (T,N,2) turned about `centre` by an angle broadcastable to (T,N) — per frame, per point
    or both. Anything with a trailing axis of its own broadcasts into a third dimension instead."""
    d = pts - centre
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([centre[0] + d[..., 0] * c - d[..., 1] * s,
                     centre[1] + d[..., 0] * s + d[..., 1] * c], -1)


# ------------------------------------------------------------------ the fields
def paths(kind: str, *, points: int, length: int, width: int, height: int,
          amount: float = 0.45, angle: float = 0.0, seed: int = 0) -> np.ndarray:
    """(length, N, 2) in pixels: where every path is on every frame of what is about to be rendered."""
    kind = spec(kind)["id"]
    p0 = seeds(points, width, height)                       # (N,2)
    length = max(2, int(length))
    e = _ease(length)[:, None]                              # (T,1)
    n = len(p0)
    c = np.array([width / 2.0, height / 2.0])
    s = float(min(width, height))                           # one yardstick, so a 16:9 orbit stays round
    a = math.radians(float(angle))
    dirv = np.array([math.cos(a), -math.sin(a)])            # screen y points down
    amt = float(amount)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    t = np.linspace(0.0, 1.0, length)[:, None]              # (T,1)

    if kind == "pan":
        out = p0[None] + (e * amt * s * 0.9)[..., None] * dirv
    elif kind == "zoom":
        out = c + (p0[None] - c) * (1.0 + e * amt * 0.8)[..., None]
    elif kind == "orbit":
        out = _rotate(p0[None], c, e * math.radians(100.0 * amt))
    elif kind == "swirl":
        r = np.linalg.norm(p0 - c, axis=1)
        fall = np.clip(1.0 - (r / (0.55 * s)) ** 2, 0.0, 1.0)            # (N,)
        out = _rotate(p0[None], c, e * (math.radians(200.0 * amt) * fall)[None])
    elif kind == "wave":
        along = p0 @ dirv                                   # how far along the travel each seed starts
        lag = (along - along.min()) / max(1e-6, float(np.ptp(along))) * 2.0 * math.pi
        perp = np.array([-dirv[1], dirv[0]])
        sway = np.sin(2.0 * math.pi * 1.5 * t - lag[None]) - np.sin(-lag[None])
        out = (p0[None] + (e * amt * s * 0.45)[..., None] * dirv
               + (sway * amt * s * 0.18)[..., None] * perp)
    else:                                                   # drift
        out = np.repeat(p0[None].astype(np.float64), length, 0)
        for freq in (0.7, 1.7):                 # two slow sines: smooth, and never reads as a loop
            ph = rng.uniform(0, 2 * math.pi, (1, n, 2))
            amp = rng.uniform(0.5, 1.0, (1, n, 2)) * amt * s * 0.3 / freq
            out = out + amp * (np.sin(2 * math.pi * freq * t[..., None] + ph) - np.sin(ph))
    return fit(np.asarray(out, dtype=np.float64), width, height)


def fit(pos: np.ndarray, width: int, height: int) -> np.ndarray:
    """The whole cloud centred and, if it hangs outside, scaled about the frame centre until it fits.

    Not clamped: a clamped path stops at the edge, and the model is then handed a point that has stopped
    moving, which is the one thing a motion field must not say by accident."""
    flat = pos.reshape(-1, 2)
    lo, hi = flat.min(0), flat.max(0)
    pos = pos + (np.array([width, height]) / 2.0 - (lo + hi) / 2.0)
    span = np.maximum(hi - lo, 1e-6)
    room = np.array([width - 2 * MARGIN, height - 2 * MARGIN], dtype=np.float64) * FIT
    k = float(min(1.0, room[0] / span[0], room[1] / span[1]))
    if k < 1.0:
        c = np.array([width / 2.0, height / 2.0])
        pos = (pos - c) * k + c
    return pos


# ------------------------------------------------------------------ what comes out
def make(dest_dir: Path, *, kind: str = "pan", points: int = 24, length: int = 81,
         width: int = 720, height: int = 720, amount: float = 0.45, angle: float = 0.0,
         seed: int = 0) -> dict:
    """One field written as <kind>-<id>.json next to the measured tracks, plus the points themselves, so
    the page can animate them without reading the file back or rendering an mp4 nobody needs."""
    k = spec(kind)
    length, width, height = max(2, int(length)), int(width), int(height)
    pos = paths(k["id"], points=points, length=length, width=width, height=height,
                amount=amount, angle=angle, seed=seed)
    data = tracks.to_json(pos, list(range(pos.shape[1])), (width, height), (width, height))
    assert all(len(p) == length for p in data), "the node needs every track the same length"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{k['id']}-{uuid.uuid4().hex[:6]}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    return {"path": str(out), "name": out.name, "field": k["id"], "label": k["label"],
            "paths": len(data), "frames": length, "width": width, "height": height,
            "amount": round(float(amount), 4), "angle": round(float(angle), 2), "seed": int(seed),
            "source": "", "tracks": data}
