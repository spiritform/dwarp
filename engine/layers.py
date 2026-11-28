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
# size comes out the same when it is flattened four times larger. BLUR_SPAN is the sigma at the top
# of the dial as a share of the short side: at 100 on a 768 render, about 92 px.
#
# The dial is not linear. A feather on a matte lives in the bottom of it and wants the fine travel
# it has always had; the top is asked for when a layer should stop being a shape at all -- a pulse
# turned into light rather than a square. So the old linear term is kept exactly as it was and a
# cubic rides on top of it: below half the dial nothing anyone already set moves much, and 100 is
# three times the blur it used to be.
BLUR_FINE = 0.04        # the linear part: what the whole dial used to be
BLUR_WIDE = 0.08        # the cubic part, which only arrives near the top
BLUR_SPAN = BLUR_FINE + BLUR_WIDE       # sigma at 100, as a share of the short side


def blur_sigma(blur: float, width: int, height: int) -> float:
    """The Gaussian sigma this dial means at this size. One definition: ffmpeg's gblur, PIL's
    GaussianBlur and the page's canvas filter all take a standard deviation, and all three have to
    agree or the preview stops predicting the render."""
    b = min(1.0, max(0.0, float(blur)))
    return (BLUR_FINE * b + BLUR_WIDE * b ** 3) * min(width, height)

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
    sigma = blur_sigma(blur, width, height)
    # steps: gblur's IIR approximation shows its seams at a large sigma, and the top of this dial
    # is now large. Three passes cost little and keep a wide blur round.
    return f",gblur=sigma={sigma:.3f}:steps={3 if sigma > 20 else 1}" if sigma >= 0.3 else ""


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
    """The stack as the engine will read it: topmost first, every field present and in range.

    A layer is usually a file, and then `path` is what it is. A GENERATED layer is not: it has
    nothing on disk and never did -- noise is a kind, a grain and a size; a pulse is a pattern, an
    envelope and a seed -- and whichever compositor gets to it draws it. Those rows carry `noise`
    or `pulse` instead of a path, and everything else about them (blend, opacity, blur, whether it
    is a matte) is the same as any other layer's.
    """
    out = []
    for raw in layers or []:
        if not isinstance(raw, dict):
            continue
        gen, beat = noise_settings(raw), pulse_settings(raw)
        if not gen and not beat and not raw.get("path"):
            continue
        mode = str(raw.get("mode") or "normal").lower().replace(" ", "").replace("-", "")
        out.append({
            "noise": gen,
            "pulse": beat,
            "path": "" if (gen or beat) else str(raw["path"]),
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


def label(L) -> str:
    """What to call a layer in a log: its file, or what a generated one is made of."""
    if L.get("noise"):
        return f"noise · {L['noise']['kind']}"
    if L.get("pulse"):
        return f"pulse · {L['pulse']['kind']}"
    return Path(L["path"]).name


def generated(L) -> bool:
    """Is this row drawn rather than read? One question, asked in a dozen places."""
    return bool(L.get("noise") or L.get("pulse"))


def draw_gen(L, width: int, height: int, i: int):
    """Frame `i` of a generated layer, as 0..1 greys (or h x w x 3 if it has a colour).

    The one place that knows which generator a row names. Both of them take the same three things
    -- a size and a frame number -- because neither has a size or a length of its own: a generated
    layer is drawn at the size of whatever it is going over, at the moment it is composited.
    """
    if L.get("noise"):
        from engine import noise
        return noise.draw(width=width, height=height, i=i, **L["noise"])
    from engine import pulse
    return pulse.draw(width=width, height=height, i=i, **L["pulse"])


def noise_settings(raw) -> dict | None:
    """The generated-layer settings on a row, or None if the row is a file like any other."""
    if not isinstance(raw, dict) or not raw.get("noise"):
        return None
    from engine import noise
    # An older save has `noise: true` beside an mp4 that was drawn for it. The settings are all
    # still on the row, so it is read as a generated layer and the stale file is simply not opened.
    return noise.settings(raw if raw.get("noise") is True else raw["noise"])


def pulse_settings(raw) -> dict | None:
    """The same, for a pulse layer: a pattern, an envelope, a seed and its own dials.

    A pulse used to be an mp4 drawn before you could see any of it, so an older save has `pulse`
    as the pattern's NAME beside the file that was drawn for it. The settings are all still on the
    row, so it is read as a generated layer and the stale file is simply not opened -- the same
    thing that happened to noise when it stopped being a file.
    """
    if not isinstance(raw, dict) or not raw.get("pulse"):
        return None
    from engine import pulse
    return pulse.settings(raw["pulse"] if isinstance(raw.get("pulse"), dict) else raw)


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
    # A matte on the base used to be thrown away here, on the grounds that there is nothing under
    # the base to reveal. There is: black. It is the same black the base's own opacity fades into
    # -- every compositor here fills with it before the base goes down -- so a matte on the base
    # punches the source out against it, which is the only way to matte the source itself. Every
    # one of the three compositors has to consume it, or the flatten loses an output and falls over.
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
            if layer["noise"]:
                # Generated as it is consumed rather than drawn to a file first: an infinite source,
                # which is fine because `-frames:v` below is what says how long the output is.
                from engine import noise
                inputs.extend(["-f", "lavfi", "-i",
                               noise.lavfi(width=width, height=height, fps=fps, **layer["noise"])])
            elif layer["pulse"]:
                # A pulse has no lavfi and cannot have one: there is no ffmpeg filter that draws a
                # ring travelling out of the middle of the frame on a beat. So here, and ONLY here,
                # it is drawn to a clip first and read back like any other input.
                #
                # This is the flatten -- it runs when you render or upload, once, for a stack that
                # has a clip at the bottom. Nothing a dial does comes through here: the preview is
                # drawn in the page and the painting modes draw it per frame below, neither of
                # which touches a file. Moving the whole flatten onto the numpy compositor would
                # retire this too; until then one clip at render time is the honest cost.
                from engine import pulse
                made = pulse.clip(dest.parent, width=width, height=height, frames=frames, fps=fps,
                                  **layer["pulse"])
                inputs.extend(["-stream_loop", "-1", "-i", made["path"]])
            else:
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
    if pairs[0][1]:
        # The base against black, as a shape. The black is made from the base itself rather than
        # from a new input, so it arrives at the same size, rate and format with nothing to line up.
        steps.append("[s0]split[k0a][k0b]")
        steps.append("[k0a]colorchannelmixer=rr=0:gg=0:bb=0[blk0]")
        steps.append("[blk0][k0b][m0]maskedmerge[s0m]")
        chain = "[s0m]"
    for n, (drawn, matte) in enumerate(pairs[1:], start=1):
        op = drawn["opacity"]
        # Which way round ffmpeg's `blend` wants its two inputs is NOT the same for every mode, and
        # it is not what the documentation leads you to expect. Measured against the numpy
        # compositor over all 256x256 pairs of values, at opacity 1:
        #
        #   normal                      wrong as [under][over], exact as [over][under]
        #   overlay, hardlight          exact as [under][over], wrong the other way round
        #   softlight                   [under][over] (ffmpeg's own formula, off by ~29/255)
        #   multiply, screen, lighten,  symmetric: identical either way
        #   darken, difference,
        #   exclusion, addition
        #
        # So only `normal` is swapped, and its opacity goes with it -- ffmpeg reads that number as
        # how much of its FIRST input survives. Handed [under][over] it was running backwards: a
        # normal layer at 100% disappeared entirely and the same layer at 0% replaced everything
        # under it. The per-frame numpy compositor was always right, which is why a render looked
        # correct while the plate behind it -- the Source tab, the cloud upload, and the picture
        # Image -> Video starts from -- quietly dropped every opaque layer.
        over_first = drawn["mode"] == "normal"
        def mix(under: str, src: str = f"[s{n}]") -> str:
            pair = f"{src}{under}" if over_first else f"{under}{src}"
            return f"{pair}blend=all_mode={drawn['mode']}:all_opacity={op:.4f}[b{n}]"
        # The matte decides *where*, the blend decides *how*, and opacity rides on the blend so a
        # matted layer at 50% is half-strength inside the matte rather than a half-faded matte.
        if matte:
            # What is underneath is needed twice -- once blended, once as what shows outside the
            # matte -- and a filter output can only be consumed once, so it is split first.
            steps.append(f"{chain}split[u{n}a][u{n}b]")
            steps.append(mix(f"[u{n}a]"))
            steps.append(f"[u{n}b][b{n}][m{n}]maskedmerge[c{n}]")
            chain = f"[c{n}]"
        else:
            steps.append(mix(chain))
            chain = f"[b{n}]"

    # A picture asks for the same graph and a different tail: Image -> Video starts from one frame,
    # so its plate is one frame, written as a picture rather than as a clip of length 1.
    one = dest.suffix.lower() == ".png"
    steps.append(f"{chain}format={'rgb24' if one else 'yuv420p'}[out]")
    cmd = (["ffmpeg", "-v", "error", "-y"] + inputs
           + ["-filter_complex", ";".join(steps), "-map", "[out]"]
           + (["-frames:v", "1", "-update", "1", str(dest)] if one else
              ["-frames:v", str(frames), "-r", f"{fps:.6f}",
               "-c:v", "libx264", "-crf", "17", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               str(dest)]))
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
    if generated(base):
        # Generated layers have no size of their own -- they are drawn at the size of what they are
        # going over. With nothing underneath there is nothing to take that from.
        raise ValueError("the bottom of the stack has to be a clip or a picture, not a generated layer")
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
        if generated(pairs[0][0]):
            raise ValueError("the bottom of the stack has to be a clip or a picture, not a generated layer")
        bw, bh, bn, bf = probe(pairs[0][0]["path"])
        width, height, frames, fps = width or bw, height or bh, frames or bn, fps or bf
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"plate-{_key(stack, width, height, frames, fps)}.mp4"
    if not (out.is_file() and out.stat().st_size):
        flatten(stack, out, width=width, height=height, frames=frames, fps=fps)
    return {"path": str(out), "name": out.name, "width": width, "height": height,
            "frames": frames, "fps": round(fps, 3), "layers": len(pairs)}


def plate_still(dest_dir: Path, layers, *, width: int = 0, height: int = 0) -> dict:
    """The flattened stack as one picture.

    Image -> Video renders out of a single frame, so there is nothing for a plate to be a clip of:
    the stack is flattened to the picture the render starts from. A layer that is a video gives its
    first frame, which is the only frame of it that could have been used.
    """
    stack = clean(layers)
    pairs = plan(stack)
    if not pairs:
        raise ValueError("a plate needs at least one layer that is not a matte")
    if not (width and height):
        if generated(pairs[0][0]):
            raise ValueError("the bottom of the stack has to be a clip or a picture, not a generated layer")
        bw, bh, _, _ = probe(pairs[0][0]["path"])
        width, height = width or bw, height or bh
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"plate-{_key(stack, width, height, 1, 1.0)}.png"
    if not (out.is_file() and out.stat().st_size):
        flatten(stack, out, width=width, height=height, frames=1, fps=1.0)
    return {"path": str(out), "name": out.name, "width": width, "height": height,
            "frames": 1, "fps": 1.0, "layers": len(pairs), "image": True}


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

    def __init__(self, job, out: Path, extract, log=lambda s: None, over: bool = False,
                 layers=None):
        from copy import copy
        self.stack = clean(layers if layers is not None else (getattr(job, "layers", None) or []))
        self.pairs = pair_index(self.stack)
        if not self.pairs:
            raise ValueError("a plate needs at least one layer that is not a matte")
        self.dir = Path(out) / "src"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames: dict[int, list] = {}
        # Text -> Video has no clip at the bottom of the stack and nothing on disk to put there:
        # the frame it is laid over is the one being painted, handed in a frame at a time. So every
        # layer here is an overlay -- none of them is the base -- and none of them is trimmed,
        # because there is no trim to be an offset into.
        self.over = bool(over)
        base_k = None if self.over else self.pairs[0][0]
        self.off: dict[int, int] = {}
        # A picture given a length is the one base that cannot be trimmed: it has a single frame, so
        # any frame_start selects nothing out of it and ffmpeg fails the render. On that base the
        # number means where the overlays start -- which is how Frame renders the frame you are
        # looking at, the picture being the same in every one of them.
        still_base = int(getattr(job, "source_frames", 0) or 0) > 0
        for k, L in enumerate(self.stack):
            # A generated layer has no frames to extract: it is drawn at the size of whatever it is
            # going over, at the moment that frame is composited. Nothing to decode, nothing on
            # disk, and a dial moved mid-render lands on the next frame like every other dial.
            if generated(L):
                self.frames[k], self.off[k] = None, 0
                log(f"  {label(L)} · generated"
                    + (" · matte" if L["matte"] else f" · {L['mode']} {L['opacity'] * 100:.0f}%"))
                continue
            one = copy(job)
            one.video = L["path"]
            # Fill the frame and keep the shape, as the flatten does. The clip itself is left alone:
            # the render size comes from it, so there is nothing to crop.
            if k != base_k:
                one.fit = "fill"
                one.frame_start, one.frame_end = 0, -1   # read whole; the trim is the clip's, not its
            elif still_base:
                one.frame_start, one.frame_end = 0, -1   # the one frame there is
            self.off[k] = 0 if self.over or k == base_k else max(0, job.frame_start)
            self.frames[k] = extract(one, Path(out) / "layers" / f"{k}")
            log(f"  {Path(L['path']).name} \u00b7 {len(self.frames[k])} frames"
                + (" \u00b7 matte" if L["matte"] else f" \u00b7 {L['mode']} {L['opacity'] * 100:.0f}%"))
        # How long the stack runs. The base usually says -- it is the clip -- but a picture at the
        # bottom is one frame and still has a length, the one that was asked for. Every layer is
        # looked up by frame number and loops, so a one-frame base simply repeats under the rest.
        # With nothing at the bottom the length is not the stack's to decide: the render says how
        # many frames there are, and nothing here is ever written to disk.
        base = [] if self.over else (self.frames[self.pairs[0][0]] or [])
        n = int(getattr(job, "source_frames", 0) or 0) if len(base) == 1 else 0
        self.paths = [] if self.over else [self.dir / f"{i:06d}.jpg" for i in range(max(n, len(base)))]
        self.done: dict[int, str] = {}

    def _frame(self, k, i: int, size=None, row=None):
        import numpy as np
        from PIL import Image
        got = None if k is None else self.frames[k]
        if got is None:                        # generated: drawn at the size of what it is over
            w, h = size or (512, 512)
            g = np.asarray(draw_gen(row if row is not None else self.stack[k], w, h, i))
            px = (np.clip(g, 0.0, 1.0) * 255.0).astype("uint8")
            # A generator may answer in colour -- a solid does -- and then it is already RGB.
            return (Image.fromarray(px, "RGB") if px.ndim == 3
                    else Image.fromarray(px, "L").convert("RGB"))
        return Image.open(got[(self.off[k] + i) % len(got)]).convert("RGB")

    def _live(self, stack=None) -> tuple:
        """The stack to composite with, and where each of its layers’ frames come from.

        A live edit usually just moves a dial. It can also ADD or DROP a generated layer -- noise, a
        solid -- because there is nothing on disk behind one: it is drawn at the moment it is
        composited, so a layer that did not exist when the render started can still be drawn now.
        That is the whole point of generating them, and a stack you cannot add to while you are
        watching it is a stack you have to stop the render to change.

        A layer that is a FILE cannot arrive this way. Its frames were extracted up front, at this
        render’s size and every-nth, and one that turned up since has none -- so a live stack is
        taken only if every file in it is a file we already pulled frames for. Otherwise the stack
        we started with stands, which is what used to happen to every edit that changed the count.

        Returns (stack, src), where src[j] is the position layer j’s frames were extracted at, or
        None for a generated one.
        """
        keep = (self.stack, list(range(len(self.stack))))
        if not stack:
            return keep
        live = clean(stack)
        pairs = pair_index(live)
        if not pairs or (not self.over and generated(live[pairs[0][0]])):
            # Nothing to draw, or a generated layer has become the base. Out here the base is where
            # the size and the length come from, and a generated layer has neither.
            return keep
        have: dict = {}
        for k, L in enumerate(self.stack):
            if not generated(L):
                have.setdefault(L["path"], []).append(k)
        src, used = [], {}
        for L in live:
            if generated(L):
                src.append(None)
                continue
            pool, at = have.get(L["path"]) or [], used.get(L["path"], 0)
            if at >= len(pool):
                return keep                    # a file nobody extracted: this is not this stack
            src.append(pool[at])
            used[L["path"]] = at + 1
        return live, src

    def write(self, i: int, stack=None) -> Path:
        """Composite frame `i` with the settings as they stand, unless that is already on disk."""
        import numpy as np
        from PIL import Image
        live, src = self._live(stack)
        # The frames already extracted still line up, because a live edit can only move settings or
        # bring a generated layer that has no frames at all. The signature is every setting but the
        # paths, plus where each layer reads from, so one arriving redraws what is already on disk.
        want = repr([{k: v for k, v in L.items() if k != "path"} for L in live]) + repr(src)
        if self.done.get(i) == want and self.paths[i].is_file():
            return self.paths[i]
        acc = self._compose(i, live, src)
        Image.fromarray((np.clip(acc, 0, 1) * 255).astype("uint8")).save(self.paths[i], quality=95)
        self.done[i] = want
        return self.paths[i]

    def lay(self, i: int, under, stack=None):
        """The stack over a frame that was never on disk, as an RGB uint8 array.

        Text -> Video paints out of its own previous frame, so the thing the stack belongs on top of
        only exists for the moment it is being painted. Nothing is cached: that frame is different
        every time, and a dial moved mid-render lands on the next one like every other dial.
        """
        import numpy as np
        live, src = self._live(stack)
        acc = self._compose(i, live, src, under)
        return (np.clip(acc, 0, 1) * 255).astype("uint8")

    def _compose(self, i: int, live: list, src: list, under=None):
        """The composite as floats in 0..1. `under` is the frame the stack goes over when the stack
        has no base of its own; without one the bottom layer of the stack is that frame.

        The pairing is read off the live stack rather than off the one we started with: a generated
        layer can be added or dropped while this runs, and where a matte sits moves when it is.
        """
        import numpy as np
        from PIL import Image, ImageFilter
        soft = lambda im, b, w, h: im.filter(ImageFilter.GaussianBlur(blur_sigma(b, w, h))) if b > 0 else im
        pairs = pair_index(live)
        if under is None:
            base_L = live[pairs[0][0]]
            img = self._frame(src[pairs[0][0]], i)
            w, h = img.size
        else:
            arr = np.asarray(under)
            under = arr.astype(np.float32) / (255.0 if arr.dtype == np.uint8 else 1.0)
            h, w = under.shape[0], under.shape[1]
        # The matte's alpha: luma, levels, invert, then the feather -- in that order, so a hard
        # pattern can be given a soft edge without the levels clipping the softness back off.
        def alpha(mk, M):
            m = np.asarray(soft(self._frame(src[mk], i, (w, h), M), 0.0, w, h).convert("RGB"),
                           np.float32) / 255.0
            lum = 0.2126 * m[..., 0] + 0.7152 * m[..., 1] + 0.0722 * m[..., 2]
            b, wt = M["black"], M["white"]
            lum = np.where(abs(wt - b) < 1e-6, (lum > b).astype(np.float32),
                           np.clip((lum - b) / (wt - b if abs(wt - b) > 1e-6 else 1), 0, 1))
            lum = lum ** (math.log(0.5) / math.log(min(0.99, max(0.01, M["mid"]))))
            if M["invert"]:
                lum = 1.0 - lum
            if M["blur"] > 0:
                lum = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(
                    ImageFilter.GaussianBlur(blur_sigma(M["blur"], w, h))), np.float32) / 255.0
            return lum[..., None]

        if under is None:
            acc = np.asarray(soft(img, base_L["blur"], w, h), dtype=np.float32) / 255.0
            acc *= base_L["opacity"]                 # nothing under the base: it fades to black
            if pairs[0][1] is not None:              # a matte on the base: the same black, as a shape
                acc *= alpha(pairs[0][1], live[pairs[0][1]])
            first = 1
        else:
            # The frame handed in is the bottom. Every layer of the stack is over it, the one at the
            # bottom of the panel included -- so that one keeps its blend mode and its matte here,
            # where as a base it would have had neither.
            acc, first = under, 0
        for n in range(first, len(pairs)):
            a, mk = pairs[n]
            L, M = live[a], (live[mk] if mk is not None else None)
            top = np.asarray(soft(self._frame(src[a], i, (w, h), L), L["blur"], w, h),
                             np.float32) / 255.0
            mixed = np.clip(_blend(acc, top, L["mode"]), 0.0, 1.0)
            mixed = acc + (mixed - acc) * L["opacity"]
            if M:
                mixed = acc + (mixed - acc) * alpha(mk, M)
            acc = mixed
        return acc


# ------------------------------------------------------------------ a generated layer
def noise_clip(dest_dir: Path, **kw) -> dict:
    """Kept so anything that still asks this module for noise gets it. The generator moved to
    engine/noise.py when there was more than one kind of it to draw."""
    from engine import noise
    return noise.clip(dest_dir, **kw)
