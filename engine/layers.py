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


def plan(layers) -> list[tuple]:
    """(drawn layer, its matte or None) bottom-up -- the order they get composited in.

    A matte layer is never drawn. It belongs to the layer directly below it in the panel, which is
    the After Effects rule and the reason the matte can sit in the stack and be looked at rather
    than hiding inside a slot.
    """
    stack = clean(layers)
    pairs = []
    for i, L in enumerate(stack):
        if L["matte"]:
            continue
        above = stack[i - 1] if i > 0 else None
        pairs.append((L, above if above and above["matte"] else None))
    pairs.reverse()                            # the last entry in the panel is the base
    # A matte on the base has nothing underneath to reveal, so it is not a matte, it is a hole. It
    # also used to leave an unconsumed filter output and take the whole flatten down with it.
    if pairs:
        pairs[0] = (pairs[0][0], None)
    return pairs


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
                         f"crop={width}:{height},setsar=1,format=gray,"
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
