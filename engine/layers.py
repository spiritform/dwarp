"""A stack of clips flattened into one, before anything is uploaded.

The problem this solves: a hosted video model takes a reference and a sentence. If you want two
videos arranged inside one shot -- this one here, that one there, this one over the top of that one
-- the only lever a sentence gives you is to *describe* the arrangement and hope. DWARP's colour
masks were an attempt at that, handing the model a picture of a mask and explaining the code in
English, and it is four inferences deep with nothing enforcing any of them.

So the arrangement stops being a claim and becomes pixels. Every source is a layer with a blend mode
and an opacity; a black and white layer can be set as a **luma matte** for the layer below it, its
brightness becoming that layer's alpha. The stack flattens to a single clip and *that* is the
reference. The model's job drops from "obey this legend" to "resolve this into a coherent shot",
which is the thing it is actually good at.

Nothing here is model-specific, so it works for any hosted graph, not just H3.
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
from pathlib import Path

# ffmpeg's blend filter names, in the order the panel offers them. These are the Photoshop set; the
# names are ffmpeg's own, so a mode added here needs no translation table.
# Only the modes a browser canvas and ffmpeg both have, and have the same way: the panel previews
# the stack live by compositing one frame itself, so a mode it cannot show is a mode that would make
# the preview lie. ffmpeg's vividlight, pinlight, phoenix and the rest are left out for that reason.
MODES = ["normal", "multiply", "screen", "overlay", "softlight", "hardlight", "difference",
         "exclusion", "lighten", "darken", "addition"]

DEFAULTS = {"mode": "normal", "opacity": 1.0, "blur": 0.0, "matte": False,
            "black": 0.0, "mid": 0.5, "white": 1.0, "invert": False}

# Blur is kept as a share of the frame rather than a number of pixels, so a stack judged at preview
# size comes out the same when it is flattened four times larger.
BLUR_SPAN = 0.04

# Rec. 709 luma, spelled out rather than left to format=gray. ffmpeg picks its matrix from the
# input's colour tags -- BT.601 for an untagged SD clip -- while the panel's live preview computes
# 0.2126/0.7152/0.0722 in a canvas. On a drawn black and white matte the two agree because 0 and 255
# are 0 and 255 either way; on a colour clip used as a matte they would not, and the preview would
# quietly stop predicting the render. So both ends are pinned to the same coefficients.
LUMA709 = ("colorchannelmixer="
           "rr=0.2126:rg=0.7152:rb=0.0722:"
           "gr=0.2126:gg=0.7152:gb=0.0722:"
           "br=0.2126:bg=0.7152:bb=0.0722")


def _blur(blur: float, width: int, height: int) -> str:
    sigma = max(0.0, float(blur)) * min(width, height) * BLUR_SPAN
    return f",gblur=sigma={sigma:.3f}" if sigma >= 0.3 else ""


def _num(v, lo, hi, fallback):
    try:
        return min(hi, max(lo, float(v)))
    except (TypeError, ValueError):
        return fallback


def _levels_lut(black: float, mid: float, white: float, invert: bool) -> str:
    """A 3-point levels curve as a lut expression, so an arbitrary grey clip can be shaped into a
    matte: black and white set what counts as fully out and fully in, mid bends everything between.

    A lut builds 256 entries once and reads them per pixel, which is why this is not a geq.
    """
    b, w = black * 255.0, white * 255.0
    if abs(w - b) < 1e-6:                      # degenerate: a hard threshold at that point
        body = f"if(gt(val,{b:.4f}),255,0)"
    else:
        gamma = math.log(0.5) / math.log(min(0.99, max(0.01, mid)))
        body = f"255*pow(clip((val-{b:.4f})/({w - b:.4f}),0,1),{gamma:.6f})"
    return f"255-({body})" if invert else body


def _source(i: int, width: int, height: int, fps: float, blur: float = 0.0) -> str:
    """One input, cropped to fill the frame and put on the render's clock.

    Cover rather than contain: a layer is a picture laid over another picture, and letterbox bars
    would be content as far as the model is concerned.
    """
    return (f"[{i}:v]fps={fps:.6f},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1{_blur(blur, width, height)},format=gbrp")


def clean(layers) -> list[dict]:
    """The stack as the engine will read it: topmost first, every field present and in range."""
    out = []
    for raw in layers or []:
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        mode = str(raw.get("mode") or "normal").lower().replace(" ", "").replace("-", "")
        out.append({
            "path": str(raw["path"]),
            "mode": mode if mode in MODES else "normal",
            "opacity": _num(raw.get("opacity"), 0.0, 1.0, 1.0),
            "blur": _num(raw.get("blur"), 0.0, 1.0, 0.0),
            "matte": bool(raw.get("matte")),
            "black": _num(raw.get("black"), 0.0, 1.0, 0.0),
            "mid": _num(raw.get("mid"), 0.01, 0.99, 0.5),
            "white": _num(raw.get("white"), 0.0, 1.0, 1.0),
            "invert": bool(raw.get("invert")),
        })
    return out


def pair_index(stack) -> list[tuple]:
    """(drawn layer's index, its matte's index or None) bottom-up, over an already-clean stack.

    A matte layer is never drawn. It belongs to the layer directly below it in the panel, which is
    the After Effects rule and the reason a matte can sit in the stack and be looked at rather than
    hiding inside a slot. Indices rather than the dicts themselves, because the per-frame compositor
    keeps a bundle of extracted frames per position and dict identity does not survive a clean().
    """
    pairs = []
    for i, L in enumerate(stack):
        if L["matte"]:
            continue
        pairs.append((i, i - 1 if i > 0 and stack[i - 1]["matte"] else None))
    pairs.reverse()                            # the last entry in the panel is the base
    # A matte on the base has nothing underneath to reveal, so it is not a matte, it is a hole. It
    # also used to leave an unconsumed filter output and take the whole flatten down with it.
    if pairs:
        pairs[0] = (pairs[0][0], None)
    return pairs


def plan(layers) -> list[tuple]:
    """(drawn layer, its matte or None) bottom-up -- the order they get composited in."""
    stack = clean(layers)
    return [(stack[a], stack[m] if m is not None else None) for a, m in pair_index(stack)]


def flatten(layers, dest: Path, *, width: int, height: int, frames: int, fps: float = 24.0) -> Path:
    """Composite the stack into one mp4 at the render's size and length.

    Every layer is looped if it is short and cut if it is long, because the output's length belongs
    to the render, not to whichever clip happened to be dropped in.
    """
    pairs = plan(layers)
    if not pairs:
        raise ValueError("a plate needs at least one layer that is not a matte")
    width, height = max(16, int(width)), max(16, int(height))
    frames, fps = max(1, int(frames)), max(1.0, float(fps))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    idx: dict[int, int] = {}                   # id(layer dict) -> ffmpeg input index

    def put(layer) -> int:
        key = id(layer)
        if key not in idx:
            idx[key] = len(idx)
            inputs.extend(["-stream_loop", "-1", "-i", layer["path"]])
        return idx[key]

    for drawn, matte in pairs:
        put(drawn)
        if matte:
            put(matte)

    steps: list[str] = []
    for n, (drawn, matte) in enumerate(pairs):
        # The base has nothing underneath it, so its opacity cannot reveal anything -- it scales the
        # channels instead, which is a fade to black. Every layer above it rides on blend's opacity.
        fade = (f",colorchannelmixer=rr={drawn['opacity']:.4f}:gg={drawn['opacity']:.4f}"
                f":bb={drawn['opacity']:.4f}") if n == 0 and drawn["opacity"] < 1 else ""
        steps.append(f"{_source(put(drawn), width, height, fps, drawn['blur'])}{fade}[s{n}]")
        if matte:
            lut = _levels_lut(matte["black"], matte["mid"], matte["white"], matte["invert"])
            # blurred after the levels, so a hard pattern can be given a soft edge without the
            # levels then clipping the softness straight back off
            steps.append(f"[{put(matte)}:v]fps={fps:.6f},"
                         f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                         f"crop={width}:{height},setsar=1,{LUMA709},format=gray,"
                         f"lutyuv=y='{lut}'{_blur(matte['blur'], width, height)},format=gbrp[m{n}]")

    chain = "[s0]"
    for n, (drawn, matte) in enumerate(pairs[1:], start=1):
        op = drawn["opacity"]
        # The matte decides *where*, the blend decides *how*, and opacity rides on the blend so a
        # matted layer at 50% is half-strength inside the matte rather than a half-faded matte.
        if matte:
            # What is underneath is needed twice -- once blended, once as what shows outside the
            # matte -- and a filter output can only be consumed once, so it is split first.
            steps.append(f"{chain}split[u{n}a][u{n}b]")
            steps.append(f"[u{n}a][s{n}]blend=all_mode={drawn['mode']}:all_opacity={op:.4f}[b{n}]")
            steps.append(f"[u{n}b][b{n}][m{n}]maskedmerge[c{n}]")
            chain = f"[c{n}]"
        else:
            steps.append(f"{chain}[s{n}]blend=all_mode={drawn['mode']}:all_opacity={op:.4f}[b{n}]")
            chain = f"[b{n}]"

    steps.append(f"{chain}format=yuv420p[out]")
    cmd = (["ffmpeg", "-v", "error", "-y"] + inputs
           + ["-filter_complex", ";".join(steps), "-map", "[out]",
              "-frames:v", str(frames), "-r", f"{fps:.6f}",
              "-c:v", "libx264", "-crf", "17", "-preset", "veryfast", "-pix_fmt", "yuv420p",
              str(dest)])
    done = subprocess.run(cmd, capture_output=True)
    if done.returncode or not dest.is_file():
        # ffmpeg says what is actually wrong on its last line; the first is the filtergraph's own
        # "[fc#0 @ 0x...]" prefix, which is the part that used to be all anyone saw.
        lines = [l.strip() for l in done.stderr.decode("utf-8", "replace").splitlines() if l.strip()]
        why = lines[-1] if lines else "no output"
        raise RuntimeError(why[:300])
    return dest


_probes: dict = {}          # path -> (mtime, result): counting frames decodes the whole file


def probe(path) -> tuple:
    """(width, height, frames, fps) of a clip, so a plate can be cut to the shape of the thing it
    replaces rather than to a number typed somewhere else.

    Cached on the file's mtime, because counting frames means decoding the whole thing -- a second
    for a 1440x1440 clip -- and the preview asks this on every slider move.
    """
    try:
        stamp = os.stat(path).st_mtime
    except OSError:
        stamp = 0.0
    hit = _probes.get(str(path))
    if hit and hit[0] == stamp:
        return hit[1]
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
         "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    vals = [v.strip() for v in out.stdout.splitlines() if v.strip()]
    if len(vals) < 4:
        raise RuntimeError(f"could not read {Path(path).name}")
    w, h, rate, n = int(vals[0]), int(vals[1]), vals[2], vals[3]
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1)
    got = (w, h, max(1, int(n)), fps if fps > 0 else 24.0)
    _probes[str(path)] = (stamp, got)
    return got


def flatten_like_base(layers, dest: Path) -> Path:
    """Flatten a stack to the size, length and rate of its own base layer.

    The modes that paint frame by frame hand `job.video` to the rest of the engine -- trim, range,
    fps, every size decision downstream. So a plate that stands in for the clip has to be the same
    shape as the clip, and the base layer is the clip. The cloud path does not use this: it already
    knows the render's own size and length and passes them in.
    """
    pairs = plan(layers)
    if not pairs:
        raise ValueError("a plate needs at least one layer that is not a matte")
    base = pairs[0][0]                          # bottom-up, so the first pair is the base
    w, h, frames, fps = probe(base["path"])
    return flatten(layers, dest, width=w, height=h, frames=frames, fps=fps)


def _key(layers, width, height, frames, fps) -> str:
    stack = clean(layers)
    bits = [f"{width}x{height}|{frames}|{fps:.4f}"]
    for L in stack:
        try:
            stamp = os.stat(L["path"]).st_mtime
        except OSError:
            stamp = 0.0
        bits.append("|".join(str(L[k]) for k in sorted(L)) + f"|{stamp:.0f}")
    return hashlib.sha1("\n".join(bits).encode("utf-8")).hexdigest()[:10]


def plate(dest_dir: Path, layers, *, width: int = 0, height: int = 0, frames: int = 0,
          fps: float = 0.0) -> dict:
    """The flattened stack, named after what is in it, so the same stack is composited once.

    Everything about the output is in the name -- every layer's path, mode, opacity, levels and
    mtime, plus the size asked for -- so moving one slider makes a new plate and moving it back
    finds the old one still sitting there.
    """
    stack = clean(layers)
    pairs = plan(stack)
    if not pairs:
        raise ValueError("a plate needs at least one layer that is not a matte")
    if not (width and height and frames and fps):
        bw, bh, bn, bf = probe(pairs[0][0]["path"])
        width, height, frames, fps = width or bw, height or bh, frames or bn, fps or bf
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"plate-{_key(stack, width, height, frames, fps)}.mp4"
    if not (out.is_file() and out.stat().st_size):
        flatten(stack, out, width=width, height=height, frames=frames, fps=fps)
    return {"path": str(out), "name": out.name, "width": width, "height": height,
            "frames": frames, "fps": round(fps, 3), "layers": len(pairs)}


# ------------------------------------------------------------------ live: composited per frame
# The flatten above makes one clip and is what the cloud uploads. The painting modes do not upload
# anything, so they can do better: keep every layer as frames and composite the one being painted,
# at the moment it is painted. That is what lets a slider move while a render runs -- drag opacity
# at frame 40 and frames 41 onward are composited differently, the same way Live camera works.
#
# The blend formulas are the separable ones from the compositing spec, which is also what a browser
# canvas computes -- so what the panel previews is what this paints. ffmpeg agrees with them on
# every mode here except soft light, where its own definition differs by a hair.
def _blend(a, b, mode: str):
    """Bottom `a` and top `b` as float arrays in 0..1."""
    import numpy as np
    if mode == "normal":
        return b
    if mode == "multiply":
        return a * b
    if mode == "screen":
        return a + b - a * b
    if mode == "overlay":
        return np.where(a <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == "hardlight":
        return np.where(b <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == "softlight":
        d = np.where(a <= 0.25, ((16 * a - 12) * a + 4) * a, np.sqrt(np.maximum(a, 0)))
        return np.where(b <= 0.5, a - (1 - 2 * b) * a * (1 - a), a + (2 * b - 1) * (d - a))
    if mode == "difference":
        return np.abs(a - b)
    if mode == "exclusion":
        return a + b - 2 * a * b
    if mode == "lighten":
        return np.maximum(a, b)
    if mode == "darken":
        return np.minimum(a, b)
    if mode == "addition":
        return a + b
    return b


class FrameStack:
    """Every layer as frames at render size, and the composite of any one of them on demand.

    Each layer goes through the engine's own frame extraction at the render's size and every-nth.
    The base is the clip, so it is read over the trimmed range. Everything laid over it is read
    whole, from its own first frame, and looked up by the absolute frame number so a layer shorter
    than the clip loops -- the way the preview loops it, and the way the flatten does.

    Reading a layer over the clip's range was what the trim meant before, and it could not work: a
    still has one frame, so trimming to 84 selected nothing out of it and ffmpeg failed the render.
    """

    def __init__(self, job, out: Path, extract, log=lambda s: None):
        from copy import copy
        self.stack = clean(getattr(job, "layers", None) or [])
        self.pairs = pair_index(self.stack)
        if not self.pairs:
            raise ValueError("a plate needs at least one layer that is not a matte")
        self.dir = Path(out) / "src"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames: dict[int, list] = {}
        base_k = self.pairs[0][0]
        self.off: dict[int, int] = {}
        for k, L in enumerate(self.stack):
            one = copy(job)
            one.video = L["path"]
            # Fill the frame and keep the shape, as the flatten does. The clip itself is left alone:
            # the render size comes from it, so there is nothing to crop.
            if k != base_k:
                one.fit = "fill"
                one.frame_start, one.frame_end = 0, -1   # read whole; the trim is the clip's, not its
            self.off[k] = 0 if k == base_k else max(0, job.frame_start)
            self.frames[k] = extract(one, Path(out) / "layers" / f"{k}")
            log(f"  {Path(L['path']).name} \u00b7 {len(self.frames[k])} frames"
                + (" \u00b7 matte" if L["matte"] else f" \u00b7 {L['mode']} {L['opacity'] * 100:.0f}%"))
        base = self.frames[self.pairs[0][0]]
        self.paths = [self.dir / f"{i:06d}.jpg" for i in range(len(base))]
        self.done: dict[int, str] = {}

    def _frame(self, k: int, i: int):
        from PIL import Image
        got = self.frames[k]
        return Image.open(got[(self.off[k] + i) % len(got)]).convert("RGB")

    def write(self, i: int, stack=None) -> Path:
        """Composite frame `i` with the settings as they stand, unless that is already on disk."""
        import numpy as np
        from PIL import Image, ImageFilter
        live = clean(stack) if stack else self.stack
        # A live edit moves settings; it cannot add or remove a layer, because the frames for one
        # were never extracted. A stack of a different shape is simply not this stack.
        if len(live) != len(self.stack):
            live = self.stack
        # A live edit changes settings, never which files are in the stack, so the frames already
        # extracted still line up. The signature is every setting but the paths.
        want = repr([{k: v for k, v in L.items() if k != "path"} for L in live])
        if self.done.get(i) == want and self.paths[i].is_file():
            return self.paths[i]
        soft = lambda im, b, w, h: im.filter(ImageFilter.GaussianBlur(b * min(w, h) * BLUR_SPAN)) if b > 0 else im
        pairs = self.pairs                     # positions never change, only the settings do
        base_L = live[pairs[0][0]]
        img = self._frame(pairs[0][0], i)
        w, h = img.size
        acc = np.asarray(soft(img, base_L["blur"], w, h), dtype=np.float32) / 255.0
        acc *= base_L["opacity"]                     # nothing under the base: it fades to black
        for n in range(1, len(pairs)):
            a, mk = pairs[n]
            L, M = live[a], (live[mk] if mk is not None else None)
            top = np.asarray(soft(self._frame(a, i), L["blur"], w, h), np.float32) / 255.0
            mixed = np.clip(_blend(acc, top, L["mode"]), 0.0, 1.0)
            mixed = acc + (mixed - acc) * L["opacity"]
            if M:
                mi = self._frame(mk, i)
                m = np.asarray(soft(mi, 0.0, w, h).convert("RGB"), np.float32) / 255.0
                lum = 0.2126 * m[..., 0] + 0.7152 * m[..., 1] + 0.0722 * m[..., 2]
                b, wt = M["black"], M["white"]
                lum = np.where(abs(wt - b) < 1e-6, (lum > b).astype(np.float32),
                               np.clip((lum - b) / (wt - b if abs(wt - b) > 1e-6 else 1), 0, 1))
                lum = lum ** (math.log(0.5) / math.log(min(0.99, max(0.01, M["mid"]))))
                if M["invert"]:
                    lum = 1.0 - lum
                if M["blur"] > 0:                    # feather after the levels, as the flatten does
                    lum = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(
                        ImageFilter.GaussianBlur(M["blur"] * min(w, h) * BLUR_SPAN)), np.float32) / 255.0
                mixed = acc + (mixed - acc) * lum[..., None]
            acc = mixed
        Image.fromarray((np.clip(acc, 0, 1) * 255).astype("uint8")).save(self.paths[i], quality=95)
        self.done[i] = want
        return self.paths[i]


# ------------------------------------------------------------------ a generated layer
def noise_clip(dest_dir: Path, *, width: int, height: int, frames: int, fps: float = 24.0,
               grain: float = 1.0, seed: int = 0) -> dict:
    """Monochrome static, as a layer you can stack like any other clip.

    Injecting noise is a different lever from raising Denoise. Denoise asks the model to invent
    more; noise changes the pixels it is looking at -- and with the warp carrying paint forward, the
    grain becomes structure the next frame elaborates. Its softness is the layer's own blur dial, so
    there is one kind here rather than six: fine static, blurred into clouds if you want clouds.

    Named after what is in it, like everything else in this file, so the same ask is made once.
    """
    width, height = max(16, int(width)) & ~1, max(16, int(height)) & ~1
    frames, fps = max(1, int(frames)), max(1.0, float(fps))
    grain = min(1.0, max(0.05, float(grain)))
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    key = f"noise|{width}x{height}|{frames}|{fps:.4f}|{grain:.3f}|{int(seed)}"
    out = dest_dir / f"noise-{hashlib.sha1(key.encode()).hexdigest()[:6]}.mp4"
    if not (out.is_file() and out.stat().st_size):
        # the noise filter rather than geq: a per-pixel expression is minutes at 1440 square
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "lavfi", "-i", f"color=c=gray:s={width}x{height}:r={fps:.4f}",
               "-vf", f"noise=alls={int(round(grain * 100))}:allf=t+u,format=yuv420p",
               "-frames:v", str(frames),
               "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", str(out)]
        done = subprocess.run(cmd, capture_output=True)
        if done.returncode or not out.is_file():
            lines = [l.strip() for l in done.stderr.decode("utf-8", "replace").splitlines() if l.strip()]
            raise RuntimeError(lines[-1] if lines else "ffmpeg wrote no noise")
    return {"path": str(out), "name": out.name, "width": width, "height": height,
            "frames": frames, "fps": round(fps, 3), "grain": round(grain, 3)}
