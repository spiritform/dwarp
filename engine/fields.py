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
    {"id": "draw", "label": "Draw", "angle": False, "draw": True,
     "note": "",                 # the canvas above says it better than a line of prose can
     "amount": {"min": 0.1, "max": 1.0, "step": 0.05, "default": 1.0}},
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


# ------------------------------------------------------------------ lines you drew yourself
MAX_STROKES, MAX_POINTS = 64, 512
MIN_TRAVEL = 0.012        # of the short side: shorter than this was a tap, not a line


def clean_strokes(strokes) -> list[np.ndarray]:
    """The page's normalised polylines, checked and turned into arrays of (x, y) in 0..1.

    Anything that arrives here was drawn by hand, so it is bounded on the way in rather than trusted:
    there are limits on both counts, and every number has to be a finite 0..1. A stroke of a **single
    point** is allowed and means an anchor — see `drawn`."""
    if not isinstance(strokes, (list, tuple)) or not strokes:
        raise ValueError("draw a line first")
    if len(strokes) > MAX_STROKES:
        raise ValueError(f"{len(strokes)} lines is more than the {MAX_STROKES} this takes")
    out = []
    for s in strokes:
        p = np.asarray(s, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] != 2:
            raise ValueError("a line is a list of [x, y] points")
        if p.shape[0] > MAX_POINTS:                      # thin it rather than refuse a long drag
            p = p[np.linspace(0, p.shape[0] - 1, MAX_POINTS).round().astype(int)]
        if not np.isfinite(p).all() or p.min() < -0.001 or p.max() > 1.001:
            raise ValueError("a drawn point is outside the frame")
        out.append(np.clip(p, 0.0, 1.0))
    return out


def drawn(strokes: list[np.ndarray], *, length: int, width: int, height: int,
          amount: float = 1.0) -> np.ndarray:
    """(length, N, 2): one path per stroke drawn, travelled over the frames of the render.

    Three choices worth naming. Each line is resampled by **arc length**, so how fast it was dragged
    never becomes how fast the picture moves — the shape is the instruction, the speed is not; and
    the same smoothstep easing every other field uses carries a path from its start to `amount` of
    the way along, so a drawn field starts and stops as gently as a formula one.

    And a stroke that goes nowhere — a click, or a drag under `MIN_TRAVEL` — is an **anchor**: a path
    that holds its point for every frame, saying *this stays put*. Everywhere else in this file a
    motionless path is the one thing a field must not say by accident; drawn by hand it is the only
    way to say "that moves, and the rest of the picture does not", which is what separates moving an
    object from moving the camera. A field of nothing but anchors is still refused, since that asks
    for no motion at all."""
    e = _ease(length)                                           # (T,)
    scale = np.array([float(width), float(height)])
    short = float(min(width, height))
    out, moving = [], 0
    for s in strokes:
        p = s * scale                                           # (P,2) in pixels
        step = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.zeros(0)
        run = np.concatenate([[0.0], np.cumsum(step)])
        if run[-1] < MIN_TRAVEL * short:                        # an anchor: where it is, for every frame
            out.append(np.repeat(p[:1], length, 0))
            continue
        moving += 1
        u = e * float(amount) * run[-1]                         # (T,) distance along the line
        out.append(np.stack([np.interp(u, run, p[:, 0]), np.interp(u, run, p[:, 1])], -1))
    if not moving:
        raise ValueError("nothing here moves — draw a line, not only anchors")
    pos = np.stack(out, 1)
    # Held inside the frame rather than fitted into it like a formula field: these were drawn in the
    # frame and only ever travel along themselves, so this is a guard, not a transform — moving what
    # was drawn would make the picture disagree with the hand that drew it.
    return np.clip(pos, MARGIN, np.array([width - MARGIN, height - MARGIN], dtype=np.float64))


# ------------------------------------------------------------------ what comes out
def make(dest_dir: Path, *, kind: str = "pan", points: int = 24, length: int = 81,
         width: int = 720, height: int = 720, amount: float = 0.45, angle: float = 0.0,
         seed: int = 0, strokes=None) -> dict:
    """One field written as <kind>-<id>.json next to the measured tracks, plus the points themselves, so
    the page can animate them without reading the file back or rendering an mp4 nobody needs."""
    k = spec(kind)
    length, width, height = max(2, int(length)), int(width), int(height)
    # Held to the range this kind declares, the one the panel's slider already obeys: a caller asking
    # a drawn line to travel -1 would get a path that stands still, which is the shape of "no motion".
    lim = k.get("amount") or {}
    amount = min(float(lim.get("max", 1.0)), max(float(lim.get("min", -1.0)), float(amount)))
    pos = (drawn(clean_strokes(strokes), length=length, width=width, height=height, amount=amount)
           if k.get("draw") else
           paths(k["id"], points=points, length=length, width=width, height=height,
                 amount=amount, angle=angle, seed=seed))
    data = tracks.to_json(pos, list(range(pos.shape[1])), (width, height), (width, height))
    assert all(len(p) == length for p in data), "the node needs every track the same length"
    # An anchor is a track like any other to the node; the page only wants the count, to say so.
    anchors = sum(1 for p in data if p[0] == p[-1] and p[0] == p[length // 2])
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{k['id']}-{uuid.uuid4().hex[:6]}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    return {"path": str(out), "name": out.name, "field": k["id"], "label": k["label"],
            "paths": len(data), "anchors": anchors, "frames": length, "width": width, "height": height,
            "amount": round(float(amount), 4), "angle": round(float(angle), 2), "seed": int(seed),
            "source": "", "tracks": data}
