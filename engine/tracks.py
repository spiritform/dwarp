"""Motion tracks: a video in, point trajectories out, for driving a motion-control model.

Models like Wan-Move take a list of hand-drawn paths — one {x, y} per output frame — and move the
picture along them. Drawing those by hand is what the FL Path Animator node is for. DWARP already has
the real thing: a warp render runs RAFT-large between every pair of source frames, and knows, per pixel,
where it went. This file turns that into the same JSON, so the motion of footage you shot (or of a warp
render you made) can drive a model that has never seen it.

The method is plain flow chaining. Seed a dense grid on the first frame; each step, sample the flow at
each point and add it; drop a point when the forward/backward round trip stops agreeing (the occlusion
test from engine/warp.py, applied at a point rather than over the whole frame). Then keep the N tracks
that survived and moved the most, spread out so they do not all pile onto one shoulder.

Flow chaining drifts: over eighty frames a point walks off whatever feature it started on. That is why
there is a preview mp4 next to the JSON — the dots are drawn on your own frames, so you can see the
drift before spending anything on a render. Good enough is the goal here, not a tracker.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

TRACK_LONG = 512               # what tracking runs at; the coordinates are scaled to the output after
GRID = 28                      # candidate seeds per side: 784 points tracked, the best few kept
TRAIL = 12                     # frames of tail drawn in the preview


class TrackError(RuntimeError):
    """Anything that stops a track, phrased for the page."""


# ------------------------------------------------------------------ the video
def probe(src: str) -> tuple[int, int, float, float]:
    """(width, height, fps, duration). ffprobe, same as the rest of DWARP."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
         "-show_entries", "stream=width,height,avg_frame_rate:format=duration", src],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise TrackError(f"could not read {Path(src).name}")
    data = json.loads(out.stdout or "{}")
    st = (data.get("streams") or [{}])[0]
    num, _, den = (st.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    dur = float((data.get("format") or {}).get("duration") or 0.0)
    return int(st.get("width") or 0), int(st.get("height") or 0), fps or 24.0, dur


def read_frames(src: str, trim: list | None = None, long: int = TRACK_LONG, cap: int = 160):
    """The trimmed span at tracking size, at the clip's own rate, up to `cap` frames.

    Tracking runs on the frames that exist rather than on the output length: asking ffmpeg for eighty
    frames of a two-second clip would hand back duplicates, and RAFT between two identical frames
    measures nothing. The paths are stretched to the output length afterwards instead, which also
    handles the other direction — a long clip tracked at a sane cost."""
    w, h, fps, dur = probe(src)
    if not w or not h:
        raise TrackError(f"no video stream in {Path(src).name}")
    a = max(0.0, float(trim[0]) if trim else 0.0)
    span = (float(trim[1]) - a) if trim and len(trim) > 1 else max(0.0, dur - a)
    span = max(0.1, span if span > 0 else dur or 1.0)
    k = min(1.0, long / max(w, h))
    tw, th = max(16, int(round(w * k)) // 2 * 2), max(16, int(round(h * k)) // 2 * 2)
    rate = min(fps, cap / span)                           # at most `cap` frames out of the span
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-i", src, "-t", f"{span:.3f}",
           "-vf", f"fps={rate:.6f},scale={tw}:{th}", "-frames:v", str(cap),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True)
    step = tw * th * 3
    if out.returncode != 0 or len(out.stdout) < step * 2:
        raise TrackError("could not decode enough frames to track (needs at least two)")
    n = len(out.stdout) // step
    frames = np.frombuffer(out.stdout[:n * step], dtype=np.uint8).reshape(n, th, tw, 3)
    return frames, (w, h), (tw, th), rate


# ------------------------------------------------------------------ the tracking
def _sample(flow, pts):
    """The flow vector at each point. flow 1x2xHxW, pts Nx2 in pixels -> Nx2."""
    import torch.nn.functional as F
    _, _, h, w = flow.shape
    g = pts.clone()
    g[:, 0] = g[:, 0] / max(1, w - 1) * 2 - 1
    g[:, 1] = g[:, 1] / max(1, h - 1) * 2 - 1
    got = F.grid_sample(flow, g.view(1, 1, -1, 2), mode="bilinear",
                        padding_mode="border", align_corners=True)
    return got[0, :, 0].transpose(0, 1)


def chase(frames, device=None, grid: int = GRID, log: Callable[[str], None] = print,
          cancelled=lambda: False, on=lambda done, total: None):
    """Every grid point walked through the whole clip. Returns (positions, alive, moved).

    positions: frames x N x 2, in tracking pixels. alive: N, how many frames each point survived
    before the round-trip test gave up on it. moved: N, how far it travelled while it was alive."""
    import torch

    from engine.warp import Flow, to_tensor
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow = Flow(dev)
    n, th, tw = len(frames), frames.shape[1], frames.shape[2]

    ys, xs = np.meshgrid(np.linspace(th * .06, th * .94, grid),
                         np.linspace(tw * .06, tw * .94, grid), indexing="ij")
    pos = torch.tensor(np.stack([xs.ravel(), ys.ravel()], 1), dtype=torch.float32, device=dev)
    live = torch.ones(len(pos), dtype=torch.bool, device=dev)
    alive = torch.zeros(len(pos), dtype=torch.float32, device=dev)
    moved = torch.zeros(len(pos), dtype=torch.float32, device=dev)
    out = [pos.clone()]

    prev = to_tensor(frames[0], dev)
    for i in range(1, n):
        if cancelled():
            break
        cur = to_tensor(frames[i], dev)
        ff, fb = flow(prev, cur), flow(cur, prev)      # both ways: the round trip is the occlusion test
        step = _sample(ff, pos)
        nxt = pos + step
        back = _sample(fb, nxt)
        err = ((step + back) ** 2).sum(1)
        mag = (step ** 2).sum(1) + (back ** 2).sum(1)
        inside = ((nxt[:, 0] >= 0) & (nxt[:, 0] <= tw - 1)
                  & (nxt[:, 1] >= 0) & (nxt[:, 1] <= th - 1))
        live &= (err <= 0.01 * mag + 0.5) & inside     # Sundaram's threshold, at a point
        alive += live.float()
        moved += live.float() * step.norm(dim=1)
        pos = torch.where(live.unsqueeze(1), nxt, pos)  # a dead point stops where it died
        out.append(pos.clone())
        prev = cur
        on(i + 1, n)
        if i % 16 == 0:
            log(f"  tracked {i + 1}/{n} frames · {int(live.sum())} points still held")
    return (torch.stack(out).cpu().numpy(), alive.cpu().numpy(), moved.cpu().numpy())


def pick(positions, alive, moved, want: int, size, sep: float = 0.0) -> list[int]:
    """The `want` best tracks: ones that lasted, then ones that moved, then spread out.

    Without the spacing rule every pick lands on the same moving shoulder and the model gets one path
    drawn eight times."""
    tw, th = size
    n = len(positions) - 1
    sep = sep or max(8.0, min(tw, th) / math.sqrt(max(1, want)) * 0.55)
    order = sorted(range(len(alive)), key=lambda i: (0 if alive[i] >= n * 0.9 else 1, -float(moved[i])))
    taken: list[int] = []
    for i in order:
        if len(taken) >= want:
            break
        if moved[i] < 1.0:                              # a point that never went anywhere is not a path
            continue
        p = positions[0][i]
        if all(float(np.hypot(*(p - positions[0][j]))) >= sep for j in taken):
            taken.append(i)
    if not taken:                                       # a perfectly still clip: give back a few anyway
        taken = order[:want]
    return taken


# ------------------------------------------------------------------ what comes out
def resample(positions, length: int):
    """The tracked frames stretched (or squeezed) to the length the model will generate, straight
    linear interpolation along time. One point per output frame is what the path nodes read."""
    n = len(positions)
    if n == length:
        return positions
    t = np.linspace(0, n - 1, max(2, int(length)))
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    f = (t - lo)[:, None, None]
    return positions[lo] * (1 - f) + positions[hi] * f


def to_json(positions, chosen, track_size, out_size) -> list:
    """The shape the path nodes read: a list of tracks, each a list of one {x, y} per output frame, in
    the pixels of the clip that is about to be generated. Every track is the same length, which the
    node requires — a point that died early was frozen rather than dropped, so they all run to the end.

    Not the {"paths": [...]} wrapper a path editor shows in its own widget: that is the editor's
    document, and what it hands the node is this."""
    (tw, th), (ow, oh) = track_size, out_size
    kx, ky = ow / max(1, tw), oh / max(1, th)
    # clamped to the frame: a path node is given pixel coordinates and has no reason to expect one
    # outside the picture, and interpolation at the ends can push a point a hair past the edge
    clamp = lambda v, hi: round(min(max(v, 0.0), float(hi)), 4)          # noqa: E731
    return [[{"x": clamp(float(p[i][0]) * kx, ow), "y": clamp(float(p[i][1]) * ky, oh)}
             for p in positions] for i in chosen]


def preview(frames, positions, chosen, dest: Path, fps: float) -> Path:
    """The tracks drawn on your own frames. This is the point of the whole thing: flow chaining drifts,
    and the only honest way to know whether it drifted too far is to watch it."""
    from PIL import Image, ImageDraw
    colours = [(255, 106, 61), (255, 179, 107), (120, 200, 255), (160, 255, 160),
               (255, 140, 200), (255, 240, 120), (180, 160, 255), (120, 255, 230)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1], frames.shape[2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dest)]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for t in range(len(positions)):
        img = Image.fromarray(frames[t].copy())
        d = ImageDraw.Draw(img)
        for k, i in enumerate(chosen):
            c = colours[k % len(colours)]
            tail = [(float(positions[s][i][0]), float(positions[s][i][1]))
                    for s in range(max(0, t - TRAIL), t + 1)]
            if len(tail) > 1:
                d.line(tail, fill=c, width=2, joint="curve")
            x, y = tail[-1]
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c, outline=(0, 0, 0))
        pipe.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
    pipe.stdin.close()
    if pipe.wait() != 0:
        raise TrackError("could not write the preview: " + pipe.stderr.read().decode()[:200])
    return dest


def make(src: str, dest_dir: Path, *, points: int = 24, length: int = 81,
         width: int = 720, height: int = 720, trim: list | None = None, fps: float = 16.0,
         log: Callable[[str], None] = print, cancelled=lambda: False,
         on=lambda done, total: None) -> dict:
    """Track one video and write <name>.json and <name>.mp4 into dest_dir."""
    t0 = time.time()
    frames, src_size, track_size, rate = read_frames(src, trim)
    log(f"tracking {Path(src).name} · {len(frames)} frames at {track_size[0]}x{track_size[1]}"
        f" (from {src_size[0]}x{src_size[1]}) · out: {length} frames at {width}x{height}")
    positions, alive, moved = chase(frames, log=log, cancelled=cancelled, on=on)
    if cancelled():
        raise TrackError("cancelled")
    chosen = pick(positions, alive, moved, points, track_size)
    held = sum(1 for i in chosen if alive[i] >= (len(positions) - 1) * 0.9)
    log(f"kept {len(chosen)} paths of {len(alive)} · {held} held all the way")

    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(src).stem[:40]}-{uuid.uuid4().hex[:6]}"
    data = to_json(resample(positions, length), chosen, track_size, (width, height))
    assert all(len(t) == length for t in data), "the node needs every track the same length"
    out = dest_dir / f"{stem}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    shown = preview(frames, positions, chosen, dest_dir / f"{stem}.mp4", rate)
    seconds = round(len(frames) / max(0.01, rate), 2)
    # Everything that was expensive lives in `positions`: the flow, the round trips, the GPU. Length,
    # output size and even how many paths to keep are decided afterwards, out of that same array — so
    # it is kept beside the JSON and refit() can answer those without touching the card again.
    keep = dest_dir / f"{stem}.chase.npz"
    np.savez_compressed(
        keep, positions=positions.astype(np.float32), alive=alive, moved=moved,
        chosen=np.asarray(chosen, dtype=np.int32),
        meta=json.dumps({"track_size": list(track_size), "source": src, "trim": list(trim or []),
                         "rate": rate, "seconds": seconds, "stem": stem}))
    log(f"wrote {out.name} and {shown.name} in {time.time() - t0:.0f}s"
        f" · kept the chase ({keep.stat().st_size / 2**20:.1f} MB) so Length, size and Paths can be"
        f" refit without the GPU")
    return {"path": str(out), "name": out.name, "preview": str(shown), "paths": len(chosen),
            "frames": length, "width": width, "height": height, "held": held, "source": src,
            "chase": str(keep), "points": int(points),
            # how long the tracked span actually ran: what "auto" on the panel matches
            "seconds": seconds}


def refit(chase: str | Path, *, points: int, length: int, width: int, height: int,
          log: Callable[[str], None] = print) -> dict:
    """The same measurement, fitted to a different Length, output box or number of paths.

    The chase is where the cost was; resampling it in time and scaling it into the output box is
    arithmetic. So changing the Length of a render, or its shape, should not mean waiting on RAFT
    again — it rewrites the JSON out of the saved array and keeps the preview you already checked.

    Asking for a different number of paths does change which tracks are chosen, and then the preview
    would be showing dots that are no longer the ones going up; that case redraws it from the clip,
    which is ffmpeg and PIL on the CPU, no card involved."""
    t0 = time.time()
    chase = Path(chase)
    try:
        z = np.load(chase, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
    except Exception as exc:                            # a half-written or hand-edited sidecar
        raise TrackError(f"could not read the saved measurement: {exc}") from None
    positions, alive, moved = z["positions"], z["alive"], z["moved"]
    was = [int(i) for i in z["chosen"]]
    track_size = tuple(meta["track_size"])
    length, width, height = max(2, int(length)), int(width), int(height)

    chosen = pick(positions, alive, moved, max(1, int(points)), track_size)
    data = to_json(resample(positions, length), chosen, track_size, (width, height))
    assert all(len(t) == length for t in data), "the node needs every track the same length"
    stem = f"{meta.get('stem', chase.stem)}-{uuid.uuid4().hex[:4]}"
    out = chase.parent / f"{stem}.json"
    out.write_text(json.dumps(data), encoding="utf-8")

    shown = chase.parent / f"{meta.get('stem', chase.stem)}.mp4"
    if chosen != was:                                   # different tracks: the old preview is a lie
        try:
            frames, _, _, rate = read_frames(meta["source"], meta.get("trim") or None)
            shown = preview(frames, positions, chosen, chase.parent / f"{stem}.mp4", rate)
            np.savez_compressed(chase, positions=positions, alive=alive, moved=moved,
                                chosen=np.asarray(chosen, dtype=np.int32),
                                meta=json.dumps({**meta, "stem": stem}))
        except TrackError as exc:                       # the clip has moved or gone
            raise TrackError(f"the paths can be refit, but the preview cannot be redrawn: {exc}") from None
    held = sum(1 for i in chosen if alive[i] >= (len(positions) - 1) * 0.9)
    log(f"refit {out.name}: {len(chosen)} paths · {length} frames · {width}x{height}"
        f" in {time.time() - t0:.1f}s, no GPU")
    return {"path": str(out), "name": out.name, "preview": str(shown), "paths": len(chosen),
            "frames": length, "width": width, "height": height, "held": held,
            "source": meta.get("source", ""), "chase": str(chase), "points": int(points),
            "seconds": meta.get("seconds", 0)}
