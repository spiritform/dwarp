"""The Shape library: black-and-white masks to try with QR Code Monster (and Shape in general).

Presets are drawn here, not downloaded: no licences, nothing in the repo but this code. They're written
once into shapes/presets the first time the library is opened. The animated ones loop seamlessly (the last
frame leads back into the first), so they pair with the Shape card's loop button. The user's own masks live
in shapes/yours.

Every preset is a recipe rather than a picture, so the one on the card can be redrawn to taste: a
loop has a **speed** and most patterns have a **density** -- their own "how much of it fits across
the frame" number, which is the turns of a spiral, the count of rings, the spokes of a sunburst.
Those redraws land in shapes/drawn, named after what was asked for, and the gallery keeps showing
the preset at its defaults. Speed is the length of the loop and not the rate it is played at: a
faster loop is a shorter one, which is the only way it can still meet itself at the end.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image

# The lattice sampler the pulse layers draw their noise with. Imported rather than copied: two
# versions of "smoothstep a lattice over a frame" would quietly drift apart, and this is the same
# thing being asked for. What differs here is the time axis -- see _fractal.
from engine.pulse import _up as _lattice_over_frame

HERE = Path(__file__).resolve().parent
ROOT = HERE / "shapes"
PRESETS = ROOT / "presets"
YOURS = ROOT / "yours"
DRAWN = ROOT / "drawn"                          # presets redrawn at your own speed / density
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

SIZE, VSIZE, FRAMES, FPS = 768, 512, 96, 24     # stills / videos: 4 s at 24 fps
DRAWN_KEEP = 60                                 # how many redraws to keep before the oldest go


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


def iris(r, p):
    """A circle opening from nothing until it fills the frame in white, then a black one doing the same
    over it: all black at the start of the loop, all white half way, all black again at the end. 1.5 is
    past the corner of the square the presets are drawn in, so it really does fill."""
    disc = _bw(1.5 * (p % 0.5) * 2 - r, 0.04)
    return disc if p < 0.5 else 1 - disc


def _blocks(seed: int, n: int = 21) -> np.ndarray:
    """QR-like random squares (n x n cells), upscaled to the frame later."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, n)) > 0.5).astype(np.float32)


def _cells(g: np.ndarray, size: int) -> np.ndarray:
    k = -(-size // g.shape[0])
    return np.kron(g, np.ones((k, k), np.float32))[:size, :size]


# ---------------------------------------------------------------- fractal loops
# Noise that wraps in time as well as in space. The pulse layer's version streams -- it holds the
# two faces it is between and draws a new one as it passes them -- which is right for something
# paced by a beat and wrong for a library loop, because a loop has to meet itself. So each octave
# is a ring of SLABS faces instead, sampled round and round: at p = 1 every octave is back on the
# face it started from, and the last frame leads into the first by construction rather than by
# being faded into it.
#
# The whole ring is redrawn from the same seed on every frame rather than kept somewhere. It is a
# few dozen numbers per octave, and it means a frame of one of these depends on nothing but the
# arguments -- which is what lets the frames be generated lazily and written straight to ffmpeg.
SLABS = 4


def _fractal(x, y, r, t, p, cells=3.0, *, seed=11, octaves=4, ridged=False, tall=1.0,
             rise=0.0, cover=0.5, soft=0.06) -> np.ndarray:
    """Octaves of looping value noise, cut to the brightest `cover` of the frame.

    `rise` drifts the field upward over the loop -- a positive one climbs -- in whole lattice cells
    so that it wraps: the sampler takes its offset modulo the lattice, so a whole number of cells
    per loop comes back to where it began and anything else jumps at the seam. Measured in cells
    and not in pixels, which is the same thing said twice: the lattice height cancels, so the field
    moves `rise * p` of the frame however coarse it is.
    """
    h, w = x.shape
    rng = np.random.default_rng(seed)
    out, amp, tot, g = np.zeros((h, w), np.float32), 1.0, 0.0, float(cells)
    for o in range(octaves):
        gw = max(2, int(round(g)))
        gh = max(2, int(round(g / max(0.1, tall))))
        ring = rng.random((SLABS, gh, gw), dtype=np.float32)
        # Finer octaves go round the ring more times, so the layers part as they move instead of
        # sliding as one sheet. A whole number of times, or they would not all land together.
        z = (p * (o + 1) % 1.0) * SLABS
        i = int(z) % SLABS
        f = z - int(z)
        f = f * f * (3.0 - 2.0 * f)
        lat = ring[i] * (1.0 - f) + ring[(i + 1) % SLABS] * f
        # The sampler reads the lattice at fy + oy, so a bigger offset fetches from further DOWN
        # the lattice -- the top of the frame shows what was below it, and the picture climbs.
        # Measured, not reasoned about: shapes_dials_test rolls one against the other.
        v = _lattice_over_frame(lat, w, h, oy=rise * p * gh * (o + 1))
        out += amp * ((1.0 - np.abs(v * 2.0 - 1.0)) if ridged else v)
        tot += amp
        amp *= 0.5
        g *= 2.0
    v = out / max(1e-6, tot)
    # A share of the frame rather than a fixed level: the mean of a lattice three cells across
    # wanders as it moves, and a fixed level turns that wander into the mask breathing. Asking for
    # half the frame keeps a cloud a cloud all the way round the loop.
    thr = float(np.quantile(v[::2, ::2], 1.0 - min(1.0, max(0.0, cover))))
    return np.clip((v - thr) / max(1e-3, soft) + 0.5, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- the presets
# One table: the name, whether it moves, how a frame of it is drawn, and which of its own numbers
# is worth a dial. `fn` takes the grid, the loop position p (0..1 over one loop, 0 for a still) and
# the pattern's density. `density` is None for the shapes that have nothing to count -- a gradient
# is a gradient, and offering a number for it would be the panel pretending.
#
# A loop's speed dial is the same for all of them, because it does the same thing to all of them.
SPEED = {"label": "speed", "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.25,
         "note": "How fast the loop runs — which is how long it is: at 2 it is half the "
                 "frames and goes round twice as often, and it still meets itself at the end"}


def _den(label, default, lo, hi, note, step=0.5):
    """One density dial. The step comes from here rather than from the panel working it out: a
    spiral's turns want halves and a disc's size wants twentieths, and the pattern knows which."""
    return {"label": label, "default": float(default), "min": float(lo), "max": float(hi),
            "step": float(step), "note": note}


TURNS = "How many times it winds from the middle to the edge — a slow sweep at the bottom of the dial, a tight whirlpool at the top"
RINGS = "How many rings fit between the middle and the edge"
SPOKES = "How many spokes come out of the middle"
BANDS = "How many bands fit across the frame"
CELLS = "How many squares fit across the frame"
MASSES = "How big the masses are — a fine froth at the bottom of the dial, a few great billows at the top"

SPEC = [
    # ---- stills
    {"name": "spiral", "video": False, "density": _den("turns", 3.5, 0.5, 12, TURNS),
     "fn": lambda x, y, r, t, p, d: spiral(x, y, r, t, turns=d)},
    {"name": "spiral, 3 arms", "video": False, "density": _den("turns", 2, 0.5, 12, TURNS),
     "fn": lambda x, y, r, t, p, d: spiral(x, y, r, t, arms=3, turns=d)},
    {"name": "spiral, fine", "video": False, "density": _den("turns", 7, 1, 20, TURNS),
     "fn": lambda x, y, r, t, p, d: spiral(x, y, r, t, arms=2, turns=d)},
    {"name": "rings", "video": False, "density": _den("rings", 5, 1, 20, RINGS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(2 * np.pi * d * r), 0.3)},
    {"name": "rings, wide", "video": False, "density": _den("rings", 2, 0.5, 12, RINGS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(2 * np.pi * d * r + np.pi / 2), 0.3)},
    {"name": "sunburst", "video": False, "density": _den("spokes", 12, 2, 48, SPOKES),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(d * t), 0.25)},
    {"name": "checker", "video": False, "density": _den("squares", 4, 1, 24, CELLS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(np.pi * d * x) * np.sin(np.pi * d * y), 0.15)},
    {"name": "stripes", "video": False, "density": _den("bands", 5, 1, 24, BANDS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(2 * np.pi * d * (x + y) / 2 ** 0.5), 0.3)},
    {"name": "waves", "video": False, "density": _den("bands", 4, 1, 16, BANDS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(2 * np.pi * d * y + 1.6 * np.sin(2 * np.pi * 1.5 * x)), 0.3)},
    {"name": "circle", "video": False,
     "density": _den("size", 0.55, 0.1, 1.4, "How much of the frame the disc fills — at the top of the dial it is past the corners", 0.05),
     "fn": lambda x, y, r, t, p, d: _bw(d - r, 0.04)},
    {"name": "glow", "video": False, "density": None,
     "fn": lambda x, y, r, t, p, d: np.clip(1 - r / 1.1, 0, 1) ** 1.6},
    {"name": "gradient", "video": False, "density": None,
     "fn": lambda x, y, r, t, p, d: (x + 1) / 2},
    {"name": "halftone", "video": False, "density": _den("dots", 8, 2, 32, "How many dots fit across the frame"),
     "fn": lambda x, y, r, t, p, d: _bw((0.5 - np.hypot((x * d + d) % 2 - 1, (y * d + d) % 2 - 1)) + 0.45 * (1 - r), 0.06)},
    {"name": "star", "video": False, "density": _den("points", 5, 3, 16, "How many points it has"),
     "fn": lambda x, y, r, t, p, d: _bw(0.22 + 0.38 * np.abs((d * (t - np.pi / 2) / (2 * np.pi)) % 1 * 2 - 1) ** 2.2 - r, 0.02)},
    {"name": "heart", "video": False, "density": None,
     "fn": lambda x, y, r, t, p, d: _bw(-((x * 1.25) ** 2 + (y * 1.25 + 0.15 - np.abs(x * 1.25) ** (2 / 3) * 0.75) ** 2 - 0.55), 0.05)},
    {"name": "blocks", "video": False, "blocks": True,
     "density": _den("squares", 21, 3, 48, CELLS), "fn": None},
    # ---- loops
    {"name": "spiral, turning", "video": True, "density": _den("turns", 3.5, 0.5, 12, TURNS),
     "fn": lambda x, y, r, t, p, d: spiral(x, y, r, t, turns=d, t0=2 * np.pi * p)},
    {"name": "spiral, 3 arms turning", "video": True, "density": _den("turns", 2, 0.5, 12, TURNS),
     "fn": lambda x, y, r, t, p, d: spiral(x, y, r, t, arms=3, turns=d, t0=2 * np.pi * p / 3)},
    {"name": "rings, outward", "video": True, "density": _den("rings", 5, 1, 20, RINGS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(2 * np.pi * (d * r - p)), 0.3)},
    {"name": "rings wide, outward", "video": True, "density": _den("rings", 2, 0.5, 12, RINGS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(2 * np.pi * (d * r - p) + np.pi / 2), 0.3)},
    {"name": "sunburst, turning", "video": True, "density": _den("spokes", 12, 2, 48, SPOKES),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(d * t + 2 * np.pi * p), 0.25)},
    {"name": "stripes, sliding", "video": True, "density": _den("bands", 5, 1, 24, BANDS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(2 * np.pi * (d * (x + y) / 2 ** 0.5 - p)), 0.3)},
    {"name": "waves, flowing", "video": True, "density": _den("bands", 4, 1, 16, BANDS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(2 * np.pi * d * y + 1.6 * np.sin(2 * np.pi * (1.5 * x - p))), 0.3)},
    {"name": "circle, expanding", "video": True, "density": None,
     "fn": lambda x, y, r, t, p, d: iris(r, p)},
    {"name": "tunnel", "video": True, "density": _den("rings", 12, 2, 36, RINGS),
     "fn": lambda x, y, r, t, p, d: _bw(np.sin(3 * t + 2 * np.pi * (d * np.log(r + 0.05) / np.log(20) + p)), 0.3)},
    {"name": "blocks, shuffling", "video": True, "blocks": True,
     "density": _den("squares", 21, 3, 48, CELLS), "fn": None},
    # ---- fractal loops: the same fields the pulse layers are made of, wrapped in time
    {"name": "clouds, drifting", "video": True, "density": _den("masses", 3, 1, 10, MASSES),
     "fn": lambda x, y, r, t, p, d: _fractal(x, y, r, t, p, d, seed=11, octaves=4)},
    {"name": "smoke, rising", "video": True, "density": _den("masses", 2.5, 1, 10, MASSES),
     "fn": lambda x, y, r, t, p, d: _fractal(x, y, r, t, p, d, seed=23, octaves=4, tall=2.6,
                                             rise=1.0, cover=0.45, soft=0.12)},
    {"name": "veins, flowing", "video": True, "density": _den("masses", 3, 1, 10, MASSES),
     "fn": lambda x, y, r, t, p, d: _fractal(x, y, r, t, p, d, seed=37, octaves=3, ridged=True,
                                             cover=0.35)},
]
BY_NAME = {s["name"]: s for s in SPEC}


def _stem(name: str) -> str:
    return name.replace(", ", "-").replace(" ", "-")


def label(spec: dict) -> str:
    """What the gallery calls it: the file's own name, commas and all taken out by the stem. The
    card has read these names since before they were a table, and a preset renaming itself would
    change what every saved run says it used."""
    return _stem(spec["name"]).replace("-", " ")


# A preset's drawing, by file name, so a listing finds the recipe behind a card exactly rather than
# by unpicking the name it was saved under.
BY_FILE = {_stem(s["name"]) + (".mp4" if s["video"] else ".png"): s for s in SPEC}

# The two tables this module used to be, kept because the drawing loop reads them and because they
# say plainly what the library is: fifteen stills and a dozen loops.
STILLS = {s["name"]: s for s in SPEC if not s["video"]}
VIDEOS = {s["name"]: s for s in SPEC if s["video"]}


def dials(spec: dict) -> dict:
    """Which dials this preset has: a loop's speed, a pattern's own density, or neither."""
    out = {}
    if spec.get("video"):
        out["speed"] = SPEED
    if spec.get("density"):
        out["density"] = spec["density"]
    return out


def dial(spec: dict, name: str, value=None) -> float:
    """One dial's value for this preset: what was asked for, held inside its range, or its default."""
    d = dials(spec).get(name)
    if d is None:
        return 1.0
    if value is None or value == "":
        return float(d["default"])
    return min(float(d["max"]), max(float(d["min"]), float(value)))


def _png(path: Path, a: np.ndarray):
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGB").save(path)


def _mp4(path: Path, frames, size: int = VSIZE, fps: float = FPS):
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{size}x{size}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path)],
                          stdin=subprocess.PIPE)
    for a in frames:
        ff.stdin.write((np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {path.name}")


def _file(name: str, ext: str) -> Path:
    return PRESETS / (_stem(name) + ext)


def _paint(spec: dict, size: int, frames: int, density: float):
    """The frames of one preset, lazily: one for a still, a whole loop for a video."""
    g = _grid(size)
    if spec.get("blocks"):
        n = max(3, int(round(density)))
        if not spec["video"]:
            yield _cells(_blocks(7, n), size)
            return
        # Twelve grids in a loop however long the loop is, so the shuffle keeps its rhythm when
        # the speed dial takes frames away.
        per = max(1, frames // 12)
        for i in range(frames):
            yield _cells(_blocks(100 + (i // per), n), size)
        return
    for i in range(frames):
        yield spec["fn"](*g, i / frames, density)


def wanted() -> list[Path]:
    return [_file(n, ".png") for n in STILLS] + [_file(n, ".mp4") for n in VIDEOS]


def make_presets() -> None:
    PRESETS.mkdir(parents=True, exist_ok=True)
    YOURS.mkdir(parents=True, exist_ok=True)
    keep = {p.name for p in wanted()}            # a renamed or dropped preset: the old drawing goes, or the
    for f in PRESETS.glob("*"):                  # library would keep listing it (it lists the folder).
        if f.is_file() and f.name not in keep:   # Only ever drawings this file made; your own live in shapes/yours.
            f.unlink(missing_ok=True)
    for spec in SPEC:
        ext = ".mp4" if spec["video"] else ".png"
        p = _file(spec["name"], ext)
        if p.exists():
            continue
        d = dial(spec, "density")
        if spec["video"]:
            _mp4(p, _paint(spec, VSIZE, FRAMES, d), VSIZE)
        else:
            _png(p, next(_paint(spec, SIZE, 1, d)))


_MAKER: threading.Thread | None = None


def presets_ready() -> bool:
    """True once every preset is on disk; otherwise starts drawing them (in the background)."""
    global _MAKER
    want = wanted()
    if all(p.exists() for p in want) and not any(f.is_file() and f.name not in {p.name for p in want} for f in PRESETS.glob("*")):
        return True
    if _MAKER is None or not _MAKER.is_alive():
        _MAKER = threading.Thread(target=make_presets, daemon=True)
        _MAKER.start()
    return False


# ---------------------------------------------------------------- drawing one to taste
def _num(v: float) -> str:
    """A dial's value in a file name: short, and the same string for the same number."""
    return f"{v:.3f}".rstrip("0").rstrip(".").replace("-", "m")


def _sweep() -> None:
    """Keep the newest DRAWN_KEEP redraws. They are cheap to make again and nobody browses them:
    the gallery shows the presets, and a redraw is reached by moving the dial that made it."""
    files = sorted((f for f in DRAWN.glob("*") if f.is_file()), key=lambda f: -f.stat().st_mtime)
    for f in files[DRAWN_KEEP:]:
        f.unlink(missing_ok=True)


def draw(preset: str, speed=None, density=None) -> dict:
    """One preset at a speed and a density of your own, as a file.

    At its defaults this hands back the preset the gallery is already showing -- same path, nothing
    drawn -- so the dials cost nothing until they are moved. Anything else is named after what was
    asked for and kept, which makes going back to a setting you liked a stat() rather than a draw.
    """
    spec = BY_NAME.get(str(preset or ""))
    if spec is None:
        raise ValueError(f"no preset called {preset!r}")
    sp, den = dial(spec, "speed", speed), dial(spec, "density", density)
    ext = ".mp4" if spec["video"] else ".png"
    frames = max(2, int(round(FRAMES / sp))) if spec["video"] else 1
    plain = _file(spec["name"], ext)
    if sp == dial(spec, "speed") and den == dial(spec, "density") and plain.is_file():
        out = plain
    else:
        DRAWN.mkdir(parents=True, exist_ok=True)
        bits = [_stem(spec["name"])]
        if spec.get("density"):
            bits.append(f"d{_num(den)}")
        if spec["video"]:
            bits.append(f"s{_num(sp)}")
        out = DRAWN / ("-".join(bits) + ext)
        if not (out.is_file() and out.stat().st_size):
            size = VSIZE if spec["video"] else SIZE
            # Written under a part name and moved into place: the name is also the cache, and a
            # file being written passes `is_file()` happily -- the same way a half-written pulse
            # clip once reached the flatten.
            tmp = out.with_name(out.name + ".part" + ext)
            if spec["video"]:
                _mp4(tmp, _paint(spec, size, frames, den), size)
            else:
                _png(tmp, next(_paint(spec, size, 1, den)))
            tmp.replace(out)
            _sweep()
    return {"path": str(out), "name": label(spec), "video": spec["video"], "preset": spec["name"],
            "frames": frames if spec["video"] else 1, "fps": float(FPS),
            "speed": round(sp, 3), "density": round(den, 3), "dials": dials(spec)}


def _entry(p: Path) -> dict:
    out = {"path": str(p), "name": p.stem.replace("-", " "), "video": p.suffix.lower() in VIDEO_EXTS}
    # The gallery card carries the recipe behind it, where there is one, so picking it brings its
    # dials along. A file of your own is a file: it has no recipe and no dials.
    spec = BY_FILE.get(p.name) if p.parent == PRESETS else None
    if spec is not None:
        out.update(preset=spec["name"], dials=dials(spec),
                   frames=FRAMES if spec["video"] else 1, fps=float(FPS))
    return out


def listing() -> dict:
    ready = presets_ready()
    YOURS.mkdir(parents=True, exist_ok=True)
    pres = sorted((p for p in PRESETS.glob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS), key=lambda p: (p.suffix != ".png", p.stem)) if PRESETS.is_dir() else []
    yours = sorted((p for p in YOURS.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS), key=lambda p: -p.stat().st_mtime)
    return {"ready": ready, "presets": [_entry(p) for p in pres], "yours": [_entry(p) for p in yours], "folder": str(YOURS)}


if __name__ == "__main__":                       # python shapelib.py: draw them all now
    make_presets()
    print("\n".join(str(p) for p in sorted(PRESETS.iterdir())))
