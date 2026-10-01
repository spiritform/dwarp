"""The Shape library: black-and-white masks to try with QR Code Monster (and Shape in general).

Presets are drawn here, not downloaded: no licences, nothing in the repo but this code. They're written
once into shapes/presets the first time the library is opened. The animated ones loop seamlessly (the last
frame leads back into the first), so they pair with the Shape card's loop button. The user's own masks live
in shapes/yours.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE / "shapes"
PRESETS = ROOT / "presets"
YOURS = ROOT / "yours"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

SIZE, VSIZE, FRAMES, FPS = 768, 512, 96, 24     # stills / videos: 4 s at 24 fps


def _grid(n: int):
    a = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(a, -a)
    return x, y, np.hypot(x, y), np.arctan2(y, x)


def _bw(v: np.ndarray, soft: float = 0.08) -> np.ndarray:
    """A signal in -1..1 to black / white with a soft edge (QR Code Monster likes a little softness)."""
    return np.clip(0.5 + v / (2 * soft), 0, 1)


# ---------------------------------------------------------------- stills: f(x, y, r, theta) -> 0..1
def spiral(x, y, r, t, arms=1, turns=3.5, t0=0.0):     # the classic "spiral illusion" swirl
    return _bw(np.sin(arms * (t + t0) + 2 * np.pi * turns * r), 0.25)


STILLS = {
    "spiral": lambda x, y, r, t: spiral(x, y, r, t),
    "spiral, 3 arms": lambda x, y, r, t: spiral(x, y, r, t, arms=3, turns=2),
    "spiral, fine": lambda x, y, r, t: spiral(x, y, r, t, arms=2, turns=7),
    "rings": lambda x, y, r, t: _bw(np.sin(2 * np.pi * 5 * r), 0.3),
    "rings, wide": lambda x, y, r, t: _bw(np.sin(2 * np.pi * 2 * r + np.pi / 2), 0.3),       # a disc and two broad bands
    "sunburst": lambda x, y, r, t: _bw(np.sin(12 * t), 0.25),
    "checker": lambda x, y, r, t: _bw(np.sin(np.pi * 4 * x) * np.sin(np.pi * 4 * y), 0.15),
    "stripes": lambda x, y, r, t: _bw(np.sin(2 * np.pi * 5 * (x + y) / 2 ** 0.5), 0.3),
    "waves": lambda x, y, r, t: _bw(np.sin(2 * np.pi * 4 * y + 1.6 * np.sin(2 * np.pi * 1.5 * x)), 0.3),
    "circle": lambda x, y, r, t: _bw(0.55 - r, 0.04),
    "glow": lambda x, y, r, t: np.clip(1 - r / 1.1, 0, 1) ** 1.6,
    "gradient": lambda x, y, r, t: (x + 1) / 2,
    "halftone": lambda x, y, r, t: _bw((0.5 - np.hypot((x * 8 + 8) % 2 - 1, (y * 8 + 8) % 2 - 1)) + 0.45 * (1 - r), 0.06),
    "star": lambda x, y, r, t: _bw(0.22 + 0.38 * np.abs((5 * (t - np.pi / 2) / (2 * np.pi)) % 1 * 2 - 1) ** 2.2 - r, 0.02),
    "heart": lambda x, y, r, t: _bw(-((x * 1.25) ** 2 + (y * 1.25 + 0.15 - np.abs(x * 1.25) ** (2 / 3) * 0.75) ** 2 - 0.55), 0.05),
}


def _blocks(seed: int, n: int = 21) -> np.ndarray:
    """QR-like random squares (n x n cells), upscaled to the frame later."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, n)) > 0.5).astype(np.float32)


def _cells(g: np.ndarray, size: int) -> np.ndarray:
    k = -(-size // g.shape[0])
    return np.kron(g, np.ones((k, k), np.float32))[:size, :size]


# ---------------------------------------------------------------- videos: f(x, y, r, theta, p) with p = 0..1 over one loop
VIDEOS = {
    "spiral, turning": lambda x, y, r, t, p: spiral(x, y, r, t, t0=2 * np.pi * p),
    "spiral, 3 arms turning": lambda x, y, r, t, p: spiral(x, y, r, t, arms=3, turns=2, t0=2 * np.pi * p / 3),
    "rings, outward": lambda x, y, r, t, p: _bw(np.sin(2 * np.pi * (5 * r - p)), 0.3),
    "rings wide, outward": lambda x, y, r, t, p: _bw(np.sin(2 * np.pi * (2 * r - p) + np.pi / 2), 0.3),
    "sunburst, turning": lambda x, y, r, t, p: _bw(np.sin(12 * t + 2 * np.pi * p), 0.25),
    "stripes, sliding": lambda x, y, r, t, p: _bw(np.sin(2 * np.pi * (5 * (x + y) / 2 ** 0.5 - p)), 0.3),
    "waves, flowing": lambda x, y, r, t, p: _bw(np.sin(2 * np.pi * 4 * y + 1.6 * np.sin(2 * np.pi * (1.5 * x - p))), 0.3),
    "circle, pulsing": lambda x, y, r, t, p: _bw(0.45 + 0.15 * np.sin(2 * np.pi * p) - r, 0.04),
    "tunnel": lambda x, y, r, t, p: _bw(np.sin(3 * t + 2 * np.pi * (4 * np.log(r + 0.05) / np.log(20) * 3 + p)), 0.3),
    "blocks, shuffling": None,                  # drawn below: a new QR-like grid every 8 frames, 12 grids per loop
}


def _png(path: Path, a: np.ndarray):
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGB").save(path)


def _mp4(path: Path, frames):
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{VSIZE}x{VSIZE}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path)],
                          stdin=subprocess.PIPE)
    for a in frames:
        ff.stdin.write((np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {path.name}")


def _file(name: str, ext: str) -> Path:
    return PRESETS / (name.replace(", ", "-").replace(" ", "-") + ext)


def make_presets() -> None:
    PRESETS.mkdir(parents=True, exist_ok=True)
    YOURS.mkdir(parents=True, exist_ok=True)
    g = _grid(SIZE)
    for name, f in STILLS.items():
        p = _file(name, ".png")
        if not p.exists():
            _png(p, f(*g))
    p = _file("blocks", ".png")
    if not p.exists():
        _png(p, _cells(_blocks(7), SIZE))
    g = _grid(VSIZE)
    for name, f in VIDEOS.items():
        p = _file(name, ".mp4")
        if p.exists():
            continue
        if f is None:
            _mp4(p, (_cells(_blocks(100 + (i // 8)), VSIZE) for i in range(FRAMES)))
        else:
            _mp4(p, (f(*g, i / FRAMES) for i in range(FRAMES)))


_MAKER: threading.Thread | None = None


def presets_ready() -> bool:
    """True once every preset is on disk; otherwise starts drawing them (in the background)."""
    global _MAKER
    want = [_file(n, ".png") for n in STILLS] + [_file("blocks", ".png")] + [_file(n, ".mp4") for n in VIDEOS]
    if all(p.exists() for p in want):
        return True
    if _MAKER is None or not _MAKER.is_alive():
        _MAKER = threading.Thread(target=make_presets, daemon=True)
        _MAKER.start()
    return False


def _entry(p: Path) -> dict:
    return {"path": str(p), "name": p.stem.replace("-", " "), "video": p.suffix.lower() in VIDEO_EXTS}


def listing() -> dict:
    ready = presets_ready()
    YOURS.mkdir(parents=True, exist_ok=True)
    pres = sorted((p for p in PRESETS.glob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS), key=lambda p: (p.suffix != ".png", p.stem)) if PRESETS.is_dir() else []
    yours = sorted((p for p in YOURS.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS), key=lambda p: -p.stat().st_mtime)
    return {"ready": ready, "presets": [_entry(p) for p in pres], "yours": [_entry(p) for p in yours], "folder": str(YOURS)}


if __name__ == "__main__":                       # python shapelib.py: draw them all now
    make_presets()
    print("\n".join(str(p) for p in sorted(PRESETS.iterdir())))
