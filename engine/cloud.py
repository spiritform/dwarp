"""Comfy Cloud: DWARP's one mode that renders somewhere else.

MiniMax-H3 is open weights, but the parts that matter are a 61 GB transformer and a 32 B conditioner, and
the small quantised ones on the Hub are pruned in a way diffusers refuses. Running it here would mean ~20 GB
of RAM before the card is even touched. So this mode doesn't run it here: DWARP sends the job to Comfy Cloud
and brings back the mp4. The only thing it needs is an API key (Settings -> Comfy Cloud), and the only thing
it costs is Comfy credits.

What runs is whichever workflow is picked on the panel. `comfy/workflows.json` lists them: each entry is a
ComfyUI graph in API format plus a short map of where DWARP's values go — which node takes the prompt, which
takes the seed, which carries the size, the length and the reference slots. Nothing in this file knows what
any particular graph is made of, so adding a workflow is a JSON file and no Python.

The first one is Pika H3 Flex, a Pika adapter on MiniMax-H3 that lands a finished 1344x768 clip in six
sampling steps. Its graph is Comfy's own `video_pika_h3_flex_ref2va` with the resolution and duration helper
nodes taken out — this file computes both and writes them straight onto the model node.

Reference to video: up to nine pictures go in, and whatever is in them — a face, a costume, a painted DWARP
frame's whole look — carries into a new clip that the prompt describes. Refer to them in the prompt as
<Picture 1>, <Picture 2> ... in the order they were added; that tagging is how H3 was trained to read them.

Unlike every other DWARP mode this one can generate sound as well, so the mp4 it writes is kept as it arrives
(frames are extracted alongside it only so the stage can scrub). Nothing here is reassembled, which would
throw the audio away.
"""
from __future__ import annotations

import json
import mimetypes
import secrets
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from engine import layers

HERE = Path(__file__).resolve().parent
COMFY = HERE / "comfy"
REGISTRY = COMFY / "workflows.json"
CLOUD = "https://cloud.comfy.org"

TIMEOUT = 45                   # per request; a render takes minutes, but each poll is small
UPLOAD = 900                   # a whole clip goes up in one POST, so it gets its own patience
RETRIES = 3                    # a reset connection is weather; only /api/prompt is sent just once
POLL = 3.0


class CloudError(RuntimeError):
    """Anything Comfy Cloud said no to, phrased for the page's log."""


# ------------------------------------------------------------------ the workflows
def kind_of(w: dict) -> str:
    """render (makes a clip, lives in the panel) or upscale (takes a finished mp4, lives in Enhance)."""
    return (w.get("kind") or "render").strip().lower()


def workflows(kind: str = "") -> list[dict]:
    """Every installed workflow, in panel order, or just the one kind. A manifest entry whose graph is
    missing is skipped rather than offered and then failed on."""
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CloudError(f"could not read {REGISTRY.name}: {e}") from None
    found = [w for w in (data.get("workflows") or [])
             if w.get("id") and w.get("graph") and (COMFY / w["graph"]).is_file()]
    return [w for w in found if kind_of(w) == kind] if kind else found


def workflow(wid: str = "", kind: str = "render") -> dict:
    """The one asked for, or the first installed of that kind. An id that is no longer there falls back
    rather than failing the job: the graph is what matters and the panel can be out of date."""
    found = workflows(kind)
    if not found:
        raise CloudError(f"no Comfy Cloud {kind} workflows are installed (engine/comfy/workflows.json)")
    return next((w for w in found if w["id"] == wid), found[0])


def track_list(text: str) -> list:
    """Motion as the path nodes read it: a list of tracks, each a list of one {x, y} per frame.

    A path editor's own document wraps the same points as {"paths": [{"points": [...]}]}, and DWARP
    wrote that shape for a while before the node turned out to want the bare list. Both come in here,
    so a tracks file saved earlier still renders."""
    try:
        data = json.loads(text or "[]")
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [p.get("points") or [] for p in (data.get("paths") or [])]
    if not isinstance(data, list):
        return []
    if data and isinstance(data[0], dict) and "x" in data[0]:
        data = [data]                              # a single track, handed over unwrapped
    return [t for t in data if isinstance(t, list) and t]


def fit_tracks(got: list, frames: int) -> list:
    """The same paths, re-spaced to one point per frame of the render that is about to run.

    A path node takes exactly one point per frame, and Length is a thing the panel changes freely —
    so motion measured or drawn at one Length used to be refused at another, which is a refusal over
    arithmetic anyone would expect us to just do. Each path is walked at even steps of its own
    normalised time: the shape it describes is untouched, it simply takes the new number of frames to
    describe it. Over fewer frames the same journey happens quicker, which is what asking for a
    shorter clip of the same motion means.
    """
    frames = max(2, int(frames))
    out = []
    for path in got:
        have = len(path)
        if have == frames:
            out.append(path)
            continue
        if have == 1:                                     # an anchor: the one place it holds
            out.append([dict(path[0]) for _ in range(frames)])
            continue
        step = (have - 1) / (frames - 1)
        fitted = []
        for i in range(frames):
            u = i * step
            a = min(have - 2, int(u))
            t = u - a
            p, q = path[a], path[a + 1]
            fitted.append({"x": round(p["x"] + (q["x"] - p["x"]) * t, 4),
                           "y": round(p["y"] + (q["y"] - p["y"]) * t, 4)})
        out.append(fitted)
    return out


def graph_of(w: dict) -> dict:
    return json.loads((COMFY / w["graph"]).read_text(encoding="utf-8"))


def caps(w: dict) -> dict:
    """What the panel should offer for this workflow, read off its manifest: no refs key, no References
    card; no length key, no Length chips. The page asks for this, so a new graph needs no page edit."""
    length = w.get("length") or {}
    return {"id": w["id"], "label": w.get("label") or w["id"], "model": w.get("model", ""),
            "short": w.get("short") or w["id"],
            "note": w.get("note", ""), "refs": int((w.get("refs") or {}).get("max", 0)),
            # a workflow whose pictures mean particular things says what they are, instead of
            # letting the panel number them Picture 1, Picture 2 and leave the rest to guesswork
            "ref_labels": [str(x) for x in ((w.get("refs") or {}).get("labels") or [])],
            "tracks": bool(w.get("tracks")), "strength": w.get("strength") or None,
            "speed": w.get("speed") or None,
            "prompt_hint": w.get("prompt_hint") or None,
            "brief": bool(w.get("brief")),
            "megapixels": float((w.get("size") or {}).get("megapixels", 1.03)),
            "multiple": int((w.get("size") or {}).get("multiple", 32)),
            "video": bool(w.get("video")), "audio_in": bool((w.get("video") or {}).get("audio")),
            "video_max": int((w.get("video") or {}).get("max", 1)),
            "sound": bool(w.get("sound")), "seconds": length.get("seconds") or [],
            "fps": float(length.get("fps", 24)), "snap": int(length.get("snap", 1)),
            "min_frames": int(length.get("min_frames", 1)),
            "max_seconds": float(length.get("max_seconds", 0))}


def upscale_caps(w: dict) -> dict:
    """What Enhance should offer for one cloud upscaler. The controls go out as the panel draws them —
    label, kind, range, options — with the node and input left behind, since only the graph needs those."""
    keep = ("id", "label", "kind", "min", "max", "step", "default", "options", "note")
    return {"id": w["id"], "label": w.get("label") or w["id"], "short": w.get("short") or w["id"],
            "model": w.get("model", ""), "note": w.get("note", ""), "cost": w.get("cost", ""),
            "limits": w.get("limits", ""),
            "controls": [{k: c[k] for k in keep if k in c} for c in (w.get("controls") or [])]}


def control_values(w: dict, params: dict | None = None) -> dict:
    """The panel's numbers, clamped to what the manifest says each control takes. Anything missing or
    out of range falls back to the declared default, so a stale page can't put nonsense on a graph."""
    params, out = params or {}, {}
    for c in w.get("controls") or []:
        kind, default = (c.get("kind") or "number"), c.get("default")
        got = params.get(c["id"], default)
        if kind == "text":
            got = str(got if got is not None else "")
        elif kind == "choice":
            opts = list(c.get("options") or [])
            got = got if got in opts else (default if default in opts else (opts[0] if opts else got))
        elif kind == "toggle":
            got = bool(got)
        else:
            try:
                got = float(got)
            except (TypeError, ValueError):
                got = float(default or 0)
            got = min(float(c.get("max", got)), max(float(c.get("min", got)), got))
            if got == int(got) and float(c.get("step", 1)) == int(float(c.get("step", 1))):
                got = int(got)
        out[c["id"]] = got
    return out


# ------------------------------------------------------------------ the account
def api_key(config: dict | None = None) -> str:
    """config.json's `comfy_api_key` (gitignored, so it can't be committed), else the environment."""
    import os
    if config is None:
        f = HERE.parent / "config.json"
        config = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    return (config.get("comfy_api_key") or os.environ.get("COMFY_API_KEY") or "").strip()


def base_url(config: dict | None = None) -> str:
    if config is None:
        f = HERE.parent / "config.json"
        config = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    return (config.get("comfy_base") or CLOUD).rstrip("/")


# ------------------------------------------------------------------ http (stdlib: no new dependency)
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """/api/view answers 302 to a signed storage URL. Following it would carry the API key to another
    host, so the redirect is read and the file fetched separately, with no headers at all."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Redirect(newurl)


class _Redirect(urllib.error.HTTPError):
    def __init__(self, url: str):
        self.url_to = url
        super().__init__(url, 302, "redirect", {}, None)  # type: ignore[arg-type]


_opener = urllib.request.build_opener(_NoRedirect)


def _call(key: str, base: str, path: str, data: bytes | None = None, ctype: str = "", method: str = "",
          timeout: int = 0, tries: int = RETRIES, log: Callable[[str], None] = lambda s: None):
    """One call to Comfy Cloud, sent again while the failure is the network rather than an answer.

    A dropped or reset connection says nothing about the request, so it goes again; a 4xx is an
    answer and is raised at once. How many tries a request may have is the caller's to decide,
    because not all of them are safe to repeat."""
    last = None
    for attempt in range(1, max(1, tries) + 1):
        req = urllib.request.Request(f"{base}{path}", data=data,        # a Request is single-use
                                     method=method or ("POST" if data else "GET"))
        req.add_header("X-API-Key", key)
        req.add_header("User-Agent", "DWARP")
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            with _opener.open(req, timeout=timeout or TIMEOUT) as r:
                body = r.read()
            return json.loads(body) if body[:1] in (b"{", b"[") else body
        except _Redirect:
            raise
        except urllib.error.HTTPError as e:
            if e.code in (500, 502, 503, 504) and attempt < tries:      # their end, not the request
                last = f"Comfy Cloud {e.code}"
                log(f"  {last} on {path} — trying again ({attempt} of {tries - 1})")
                time.sleep(2 * attempt)
                continue
            detail = (e.read() or b"")[:400].decode("utf-8", "replace").strip()
            raise CloudError({
                401: "Comfy Cloud rejected the API key — check it in Settings.",
                402: "Comfy Cloud: out of credits.",
                429: "Comfy Cloud: this account has no active subscription (a render needs one).",
            }.get(e.code, f"Comfy Cloud {e.code} on {path}{': ' + detail if detail else ''}")) from e
        except (urllib.error.URLError, OSError) as e:
            last = getattr(e, "reason", None) or e
            if attempt < tries:
                log(f"  the connection dropped — trying again ({attempt} of {tries - 1})")
                time.sleep(2 * attempt)
                continue
            raise CloudError(f"couldn't reach Comfy Cloud ({last})") from e
    raise CloudError(f"couldn't reach Comfy Cloud ({last})")            # unreachable, kept honest


def _multipart(fields: dict[str, str], name: str, file: Path) -> tuple[bytes, str]:
    boundary = "----dwarp" + secrets.token_hex(12)
    out = bytearray()
    for k, v in fields.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    mime = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
    out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{file.name}\"\r\nContent-Type: {mime}\r\n\r\n").encode()
    out += file.read_bytes() + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


# ------------------------------------------------------------------ the pieces of a render
def upload(key: str, base: str, path: Path) -> str:
    """A picture into the account's asset store. Cloud is content-addressed, so sending the same frame
    twice is free; the name it answers with is what the graph's LoadImage asks for."""
    body, ctype = _multipart({"type": "input", "overwrite": "true"}, "image", path)
    r = _call(key, base, "/api/upload/image", body, ctype, timeout=UPLOAD)
    if not isinstance(r, dict) or not r.get("name"):
        raise CloudError(f"upload of {path.name} gave back no file name")
    return r["name"]


def _asset(r) -> dict:
    """The asset out of an /api/assets answer, whether it came back bare or wrapped."""
    if not isinstance(r, dict):
        return {}
    inner = r.get("asset")
    return inner if isinstance(inner, dict) else r


def upload_video(key: str, base: str, path: Path,
                 log: Callable[[str], None] = lambda s: None) -> str:
    """A clip into the asset store. /api/upload/image is for pictures only, so a video goes through
    /api/assets.

    LoadVideo names its files by content, not by what the file was called here, so the answer's `hash`
    is the name to put in the graph — it already carries the extension ("b5def...157a.mp4"). Sending
    `display_name` instead gets "the input file ... doesn't exist" back from the node. `asset_hash`
    (`blake3:<hex>`, no extension) is the spelling in the API docs; both are accepted here."""
    body, ctype = _multipart({"tags": "input", "name": path.name,
                              "mime_type": "video/mp4"}, "file", path)
    asset = _asset(_call(key, base, "/api/assets", body, ctype, timeout=UPLOAD, log=log))
    if not asset:
        raise CloudError(f"upload of {path.name} gave back no asset")
    name = asset.get("hash") or ""
    if not name and asset.get("asset_hash"):
        name = asset["asset_hash"].split(":")[-1] + (path.suffix or ".mp4")
    if not name:
        raise CloudError("Comfy Cloud stored the clip but gave back no content hash, so the graph "
                         f"can't name it: {json.dumps(asset)[:300]}")
    return name


def seconds_of(path: Path) -> float:
    """How long an mp4 actually is. The trim is what was *asked* for -- a video context has no trim,
    so the ask is H3's whole 15 s ceiling -- and saying that in the log reads as though a 5 s mask
    were being stretched to fifteen. What went up is what the log should say."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
        return max(0.0, float(out.stdout.strip()))
    except (ValueError, OSError):
        return 0.0


def has_audio(path: Path) -> bool:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                              "stream=codec_type", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=30)
        return "audio" in out.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def trim_span(trim: list, limit: float = 15.0) -> tuple[float, float, bool]:
    """Where the reference starts and how long it runs, plus whether H3's own ceiling cut it short."""
    a = max(0.0, float(trim[0]) if trim else 0.0)
    want = (float(trim[1]) - a) if trim and len(trim) > 1 else limit
    span = max(0.5, min(limit, want))
    return a, span, want > limit + 0.05


def clip_segment(src: str, trim: list, dest: Path, limit: float = 15.0) -> Path:
    """The trimmed range as its own small mp4, since the whole source may be minutes long and only the
    part under the trim is the reference. H3 looks at 15 s at most, so nothing longer is worth sending."""
    a, span, _ = trim_span(trim, limit)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:g}", "-t", f"{span:g}", "-i", str(src),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest)],
                   check=True)
    return dest


def frame_count(seconds: float, length: dict | None = None) -> int:
    """Seconds as the model's own frame count. H3, for one, takes 17n + 5 frames at 24 fps and nothing
    shorter than 124 (~5.2 s), so asking it for five seconds gets 5.2."""
    length = length or {}
    fps, snap = float(length.get("fps", 24)), max(1, int(length.get("snap", 1)))
    floor = max(1, int(length.get("min_frames", 1)))
    n = max(floor, round(max(0.0, seconds) * fps))
    return n + (floor % snap - n % snap) % snap


def fit_size(width: int, height: int, size: dict | None = None) -> tuple[int, int]:
    """The asked-for shape at the model's working area, on the multiple it needs. H3's 1.03 MP is what
    puts 16:9 on exactly the 1344x768 Pika Flex is tuned for."""
    size = size or {}
    megapixels, multiple = float(size.get("megapixels", 1.03)), max(1, int(size.get("multiple", 32)))
    w, h = max(1, int(width)), max(1, int(height))
    k = (megapixels * 1_000_000 / (w * h)) ** 0.5
    fit = lambda v: max(multiple, int(round(v * k / multiple)) * multiple)        # noqa: E731
    return fit(w), fit(h)


def build(w: dict, graph: dict, prompt: str, refs: list[str], width: int, height: int,
          frames: int, seed: int, videos: list | None = None,
          tracks: str = "", strength: float | None = None, fast: bool = False) -> dict:
    """The shipped graph with this job's values on it, placed where the manifest says they go.

    The reference pictures replace whatever the graph carries: one LoadImage per picture, wired into the
    auto-growing slots (ref_image_0, ref_image_1 ...). Clips join them as <Video 1..n> the same way,
    through LoadVideo -> GetVideoComponents, which gives the frames and — if the slot exists — the
    sound. `videos` is [(stored name, carry its sound)] in the order the prompt will name them."""
    videos = list(videos or [])
    g = {k: {**v, "inputs": dict(v["inputs"])} for k, v in graph.items()}
    at = lambda spec: g[str(spec[0])]["inputs"]                                  # noqa: E731
    if w.get("prompt"):
        at(w["prompt"])[w["prompt"][1]] = prompt
    if w.get("seed"):
        at(w["seed"])[w["seed"][1]] = int(seed) % (2 ** 63)

    main = g[str(w["node"])]["inputs"]

    def put(where, value, fallback):
        """A manifest slot is either an input name on the model node, or [node, input] anywhere."""
        if isinstance(where, (list, tuple)):
            g[str(where[0])]["inputs"][where[1]] = value
        else:
            main[where or fallback] = value

    size = w.get("size") or {}
    put(size.get("width"), int(width), "width")
    put(size.get("height"), int(height), "height")
    if w.get("length"):
        unit = (w["length"].get("unit") or "frames")
        main[w["length"].get("input", "length")] = (
            round(frames / float(w["length"].get("fps", 24)), 2) if unit == "seconds" else int(frames))

    for node_id, node in list(g.items()):              # whatever the graph shipped as its example
        if node.get("class_type") in (w.get("drop") or []):
            del g[node_id]

    spec = w.get("refs") or {}
    named = [str(x) for x in (spec.get("slots") or [])]
    if named:
        # Fixed inputs rather than an auto-growing list: the first and last frame of FL2VA are two
        # named sockets, not slot 0 and slot 1 of the same list. A socket with no picture for it has
        # to be removed outright, because the loader it pointed at is dropped with the examples.
        for i, name in enumerate(refs[:len(named)]):
            node_id = str(900 + i)
            g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name},
                          "_meta": {"title": named[i].replace("_", " ")}}
            main[named[i]] = [node_id, 0]
        for empty in named[len(refs):]:
            main.pop(empty, None)
    elif spec.get("slot"):                             # auto-growing reference slots: one loader each
        for slot in [k for k in main if k.startswith(spec.get("clear", "ref_images."))]:
            del main[slot]
        for i, name in enumerate(refs[:int(spec.get("max", 0))]):
            node_id = str(900 + i)
            g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name},
                          "_meta": {"title": f"Picture {i + 1}"}}
            main[spec["slot"].replace("{i}", str(i))] = [node_id, 0]
    elif spec.get("node") and refs:                    # one picture, onto a loader the graph already has
        g[str(spec["node"])]["inputs"][spec.get("input", "image")] = refs[0]

    sspec = w.get("strength") or {}
    if sspec and strength is not None:                 # how hard the conditioning pulls the picture
        main[sspec.get("input", "strength")] = float(strength)

    pspec = w.get("speed") or {}
    if pspec:
        # One boolean in the graph that swaps model, steps and CFG together (WanMove's "Enable Turbo
        # LoRA" switch feeds all three). The manifest says which value is the fast one, so a graph
        # wired the other way round needs no code here.
        yes = bool(pspec.get("fast", True))
        g[str(pspec["node"])]["inputs"][pspec.get("input", "value")] = yes if fast else not yes

    tspec = w.get("tracks") or {}
    if tspec:                                          # measured motion, as the path JSON it reads
        at(tspec["coords"])[tspec["coords"][1]] = tracks or "[]"

    vspec = w.get("video") or {}
    if videos and vspec:
        # One loader pair per video, into the same kind of auto-growing list the pictures use. A slot
        # name with no {i} in it is the old single-video shape, and keeps working.
        many = "{i}" in str(vspec.get("slot", ""))
        if many:
            for slot in [k for k in main if k.startswith(vspec.get("clear", "ref_videos."))]:
                del main[slot]
            for slot in [k for k in main if k.startswith(vspec.get("audio_clear", "ref_video_audios."))]:
                del main[slot]
        for i, (stored, with_sound) in enumerate(videos if many else videos[:1]):
            load, comps = str(910 + i * 2), str(911 + i * 2)
            g[load] = {"class_type": "LoadVideo", "inputs": {"file": stored},
                       "_meta": {"title": f"Video {i + 1}"}}
            g[comps] = {"class_type": "GetVideoComponents", "inputs": {"video": [load, 0]},
                        "_meta": {"title": f"Video {i + 1} frames"}}
            main[vspec["slot"].replace("{i}", str(i))] = [comps, 0]        # IMAGE: the frames
            if with_sound and vspec.get("audio"):
                main[vspec["audio"].replace("{i}", str(i))] = [comps, 1]   # AUDIO: what it sounded like
    return g


def submit(key: str, base: str, graph: dict) -> str:
    """Queue a graph. The key goes in the body as well as the header, because an API node needs it
    there: ComfyUI fills a node's hidden `api_key_comfy_org` input from `extra_data` (execution.py),
    and a partner node with nothing in it refuses with "Unauthorized: Please login first to use this
    node". The header alone authenticates the request, not the nodes inside it — which never came up
    while every graph here ran a real model on the GPU."""
    body = {"prompt": graph, "extra_data": {"api_key_comfy_org": key}}
    # The one call that must not be repeated: a second /api/prompt is a second job and a second bill.
    r = _call(key, base, "/api/prompt", json.dumps(body).encode(), "application/json", tries=1)
    if not isinstance(r, dict):
        raise CloudError("Comfy Cloud gave an unexpected answer to the job")
    if r.get("node_errors"):
        raise CloudError(f"the graph was rejected: {json.dumps(r['node_errors'])[:300]}")
    if not r.get("prompt_id"):
        raise CloudError(f"no job id came back: {json.dumps(r)[:300]}")
    return r["prompt_id"]


def wait(key: str, base: str, prompt_id: str, *, tick: Callable[[str], None] = lambda s: None,
         cancelled: Callable[[], bool] = lambda: False, log: Callable[[str], None] = print,
         what: str = "render", doing: str = "") -> dict | None:
    """Poll one cloud job to the end. Returns its detail, or None if it was cancelled here.

    Comfy's REST API reports no percentage, so a bar can't move and sits at 0 until the clip arrives.
    Elapsed time is the only honest signal there is, so it goes out every poll and into the log every
    half minute — otherwise a four-minute render looks like a hung one."""
    t0, seen, beat = time.time(), "", 0.0
    while True:
        if cancelled():
            try:
                _call(key, base, "/api/queue", json.dumps({"delete": [prompt_id]}).encode(),
                      "application/json")
                log("cancelled on Comfy Cloud")
            except CloudError as e:
                log(f"couldn't cancel the cloud job: {e}")
            return None
        detail = _call(key, base, f"/api/jobs/{prompt_id}")
        state = (detail.get("status") or "") if isinstance(detail, dict) else ""
        mins, secs = divmod(round(time.time() - t0), 60)
        label = {"pending": "waiting for a GPU",
                 "in_progress": doing or f"{what}ing"}.get(state, state or "working")
        tick(f"{label} on Comfy Cloud · {mins}m {secs:02d}s")
        waited = time.time() - t0
        if state != seen or waited - beat >= 30:
            log(f"{label} ({mins}m {secs:02d}s)")
            seen, beat = state, waited
        if state == "completed":
            return detail
        if state in ("failed", "error", "cancelled"):
            why = (detail.get("error_message") or
                   json.dumps(detail.get("execution_error") or {})[:300] or state)
            raise CloudError(f"the cloud {what} {state}: {why}")
        time.sleep(POLL)


def outputs(detail: dict, saver: str = "", skip: str = "") -> list[dict]:
    """What the finished job wrote.

    The saving node is asked first and on its own. Scanning every node instead gets the clip we sent
    up as well — LoadVideo reports the file it read, under the same shape as a real output, and being
    a video it wins any "prefer a video" sort. That is how a render came back as its own source."""
    nodes = detail.get("outputs") or {}

    def files(node) -> list[dict]:
        return [f for items in (node or {}).values() if isinstance(items, list)
                for f in items if isinstance(f, dict) and f.get("filename")
                and Path(f["filename"]).name != skip]

    if saver in nodes:
        found = files(nodes[saver])
        if found:
            return found
    found = [f for node in nodes.values() for f in files(node)]       # an older shape, or a moved saver
    video = [f for f in found if Path(f["filename"]).suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")]
    return video or found


def fetch(key: str, base: str, f: dict, dest: Path,
          log: Callable[[str], None] = lambda s: None) -> Path:
    """Two steps on purpose: read where the file lives, then fetch it with no API key attached."""
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                "type": f.get("type", "output")})
    try:
        body = _call(key, base, f"/api/view?{q}", timeout=UPLOAD, log=log)
        dest.write_bytes(body if isinstance(body, bytes) else json.dumps(body).encode())
        return dest
    except _Redirect as r:
        for attempt in range(1, RETRIES + 1):
            try:
                with urllib.request.urlopen(r.url_to, timeout=UPLOAD) as got:  # signed URL, no headers
                    dest.write_bytes(got.read())
                return dest
            except (urllib.error.URLError, OSError) as e:
                if attempt == RETRIES:
                    raise CloudError(f"the file was ready but the download failed ({e})") from e
                log(f"  the download dropped — trying again ({attempt} of {RETRIES - 1})")
                time.sleep(2 * attempt)
        return dest


def mute(video: Path) -> Path:
    """The audio track out of the mp4, kept as a remux so the picture is not touched or re-encoded.
    H3 makes picture and sound in one pass and there is no way to ask it for only the picture, so
    this costs nothing at render time and saves nothing either — it is about what the file holds."""
    quiet = video.with_suffix(".silent.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-c", "copy", "-an", str(quiet)],
                   check=True)
    video.unlink(missing_ok=True)
    quiet.rename(video)
    return video


def extract(video: Path, frames_dir: Path) -> list[Path]:
    """Stills beside the mp4, so the stage can scrub the clip like any other run. The mp4 itself is left
    exactly as it arrived — rebuilding it from these would drop H3's sound."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-start_number", "0",
                    str(frames_dir / "%06d.png")], check=True)
    return sorted(frames_dir.glob("*.png"))


# ------------------------------------------------------------------ the mode
def measure(path: Path) -> tuple[int, int, int]:
    """width, height and frame count of an mp4, straight from ffprobe. A graph that enlarges has to be
    told the shape it is enlarging, or it invents one \u2014 which is how a square clip comes back cropped."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True)
    try:
        st = (json.loads(out.stdout).get("streams") or [{}])[0]
        return int(st["width"]), int(st["height"]), int(st.get("nb_read_frames") or 0)
    except (ValueError, KeyError, IndexError) as e:
        raise CloudError(f"couldn't read {path.name}: {out.stderr.strip()[:200] or e}") from None


def upscale_size(spec: dict, width: int, height: int, scale: float) -> tuple[int, int]:
    """The frame to ask for: the source times the scale, kept to the graph's multiple and under its
    megapixel ceiling. Scaling both sides by the same number is what keeps the shape the run was made in."""
    mp = float(spec.get("max_megapixels") or 0)
    scale = max(1.0, float(scale))
    if mp and width * height * scale * scale > mp * 1_000_000:
        scale = max(1.0, (mp * 1_000_000 / (width * height)) ** 0.5)
    mult = max(1, int(spec.get("multiple", 8)))
    fit = lambda v: max(mult, int(round(v * scale / mult)) * mult)
    return fit(width), fit(height)


def upscale_cloud(wid: str, src: Path, dest: Path, params: dict | None = None, *,
                  progress: Callable[[str], None] = lambda note: None,
                  cancelled: Callable[[], bool] = lambda: False,
                  log: Callable[[str], None] = print) -> Path:
    """Send a finished run's mp4 to a cloud upscaler and write what comes back to `dest`.

    Nothing here touches the card, so this can run while a render has it — which is the point: an
    upscale that costs you nothing locally should not have to wait its turn behind one that does.
    """
    key, base = api_key(), base_url()
    if not key:
        raise CloudError("no Comfy Cloud API key — add one in Settings. "
                         "Keys come from platform.comfy.org/profile/api-keys.")
    src = Path(src)
    if not src.is_file():
        raise CloudError(f"no video at {src}")

    w = workflow(wid, kind="upscale")
    name = w.get("label") or w["id"]
    values = control_values(w, params)
    said = " \u00b7 ".join(f"{c.get('label') or c['id']} {values[c['id']]}"
                          for c in (w.get("controls") or []) if c["id"] in values)
    log(f"{name} on Comfy Cloud \u00b7 {src.name} ({src.stat().st_size / 2**20:.1f} MB)"
        f"{f' \u00b7 {said}' if said else ''}")
    if w.get("limits"):
        log(f"  {w['limits']}")

    progress("sending the clip")
    stored = upload_video(key, base, src, log=log)
    log(f"  sent as {stored}")
    if cancelled():
        raise CloudError("cancelled")

    graph = graph_of(w)
    node, field = w["input"]
    graph[str(node)]["inputs"][field] = stored
    if w.get("seed"):        # fresh every run, so asking again is another roll and not the cloud's cache
        graph[str(w["seed"][0])]["inputs"][w["seed"][1]] = secrets.randbelow(2 ** 31)
    for c in w.get("controls") or []:
        if c.get("node"):                 # a control with no node feeds the size below, not the graph
            graph[str(c["node"])]["inputs"][c["input"]] = values[c["id"]]

    size, frames_at = w.get("size") or {}, w.get("frames")
    if size or frames_at:
        sw, sh, count = measure(src)
        if size:
            tw, th = upscale_size(size, sw, sh, values.get("scale", 1))
            graph[str(size["width"][0])]["inputs"][size["width"][1]] = tw
            graph[str(size["height"][0])]["inputs"][size["height"][1]] = th
            log(f"  {sw}x{sh} -> {tw}x{th}")
        if frames_at and count:
            graph[str(frames_at[0])]["inputs"][frames_at[1]] = count
            log(f"  {count} frames")

    prompt_id = submit(key, base, graph)
    log(f"queued on Comfy Cloud ({prompt_id})")
    detail = wait(key, base, prompt_id, tick=progress, cancelled=cancelled, log=log,
                  what="upscale", doing="upscaling")
    if detail is None:
        raise CloudError("cancelled")

    files = outputs(detail, w.get("saver", ""), skip=stored)   # never hand back the clip we sent up
    if not files:
        raise CloudError("the cloud upscale finished but wrote no file")
    progress("downloading the clip")
    log(f"downloading {files[0]['filename']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = fetch(key, base, files[0], dest, log=log)
    log(f"downloaded {dest.name} ({got.stat().st_size / 2**20:.1f} MB)")
    return got


def render_cloud(job, progress: Callable = lambda *a: None, cancelled: Callable[[], bool] = lambda: False,
                 log: Callable[[str], None] = print) -> list[Path]:
    """Called by engine.warp.render for the Comfy family. Returns [] so the caller doesn't reassemble the
    mp4 (that would lose any audio); the file is already written."""
    key, base = api_key(), base_url()
    if not key:
        raise CloudError("no Comfy Cloud API key — add one in Settings. "
                         "Keys come from platform.comfy.org/profile/api-keys.")

    w = workflow(getattr(job, "comfy_workflow", "") or "")
    name = w.get("label") or w["id"]
    tracks = ""
    if w.get("tracks") and (getattr(job, "comfy_tracks", "") or ""):
        tracks = Path(job.comfy_tracks).read_text(encoding="utf-8")
    length, limit = w.get("length") or {}, float((w.get("length") or {}).get("max_seconds", 15.1))
    max_refs = int((w.get("refs") or {}).get("max", 0))

    out = Path(job.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    refs = [Path(p) for p in (getattr(job, "h3_refs", None) or []) if p and Path(p).is_file()]
    vspec = w.get("video") or {}
    max_vids = int(vspec.get("max", 1))
    # <Video 1> is the panel's own clip; anything after it came from the extra context slots.
    sources = [(getattr(job, "h3_video", "") or "", list(getattr(job, "h3_trim", None) or []))]
    sources += [(p, list(t) if isinstance(t, list) else [])
                for p, t in zip(getattr(job, "h3_videos", None) or [],
                                list(getattr(job, "h3_trims", None) or []) + [[]] * 9)]
    sources = [(p, t) for p, t in sources if p][:max_vids] if vspec else []
    source = sources[0][0] if sources else ""
    # A layer stack is flattened to one clip here, before anything is uploaded, and that clip is
    # the reference. The arrangement stops being a sentence the model has to believe and becomes
    # pixels, which is a thing the model can simply see.
    stack = layers.clean(getattr(job, "layers", None) or []) if vspec else []
    if not refs and not sources and not stack:
        raise CloudError(f"{name} needs a source clip, or a picture in a Reference slot.")

    frames = frame_count(getattr(job, "h3_seconds", 5.0), length)
    width, height = fit_size(job.width, job.height, w.get("size"))
    fps = float(length.get("fps", 24))
    drawn = layers.plan(stack)
    parts = ([f"{len(refs)} reference{'s' if len(refs) > 1 else ''}"] if refs else []) + \
            ([f"a plate of {len(drawn)} layer{'s' if len(drawn) > 1 else ''}"] if stack else
             [f"{len(sources)} video contexts"] if len(sources) > 1 else
             ["the source clip"] if sources else [])
    log(f"{name} on Comfy Cloud · {width}x{height} · {frames} frames "
        f"({frames / fps:.1f}s) · {' + '.join(parts)}")

    if stack:
        progress("uploading", 0, frames, "flattening the layers")
        for L, m in drawn:
            log(f"  {Path(L['path']).name} · {L['mode']} {L['opacity'] * 100:.0f}%"
                + (f" · matted by {Path(m['path']).name}" if m else ""))
        plate = layers.flatten(stack, out / "comfy_plate.mp4",
                               width=width, height=height, frames=frames, fps=fps)
        log(f"  plate · {plate.stat().st_size / 2**20:.1f} MB, {frames / fps:.1f}s")
        sources = [(str(plate), [0.0, round(frames / fps, 3)])]
        source = str(plate)

    progress("uploading", 0, frames, "sending the references")
    names = []
    for i, p in enumerate(refs[:max_refs], 1):
        names.append(upload(key, base, p))
        log(f"  <Picture {i}> {p.name}")

    videos = []                               # [(stored name, carry its sound)] in <Video k> order
    for k, (path, trim) in enumerate(sources, 1):
        if cancelled():
            break
        progress("uploading", 0, frames, f"sending video {k}" if len(sources) > 1 else "sending the clip")
        a, span, clipped = trim_span(trim, limit)
        seg = clip_segment(path, trim, out / f"comfy_reference{k}.mp4", limit)
        sound = has_audio(seg) and bool(vspec.get("audio"))
        got = seconds_of(seg) or span            # what actually went up, not what was asked for
        stored = upload_video(key, base, seg)
        videos.append((stored, sound))
        log(f"  <Video {k}> {Path(path).name} · {a:.1f}s to {a + got:.1f}s"
            f"{f' (cut to the {limit:.0f} s it looks at)' if clipped and got > limit - 0.1 else ''} "
            f"· {seg.stat().st_size / 2**20:.1f} MB"
            f"{f', with its sound as <Audio {k}>' if sound else ', no sound track'}")
        log(f"     stored as {stored}")     # the content name the graph asks LoadVideo for
        seg.unlink(missing_ok=True)
    video = videos[0][0] if videos else ""
    if cancelled():
        return []

    if tracks:
        got = track_list(tracks)
        if not got:
            raise CloudError(f"{Path(job.comfy_tracks).name} holds no usable paths")
        was = len(got[0])
        if was != frames:                         # arithmetic, not a reason to refuse someone's work
            got = fit_tracks(got, frames)
            log(f"  fitted the motion from {was} points per path to this render's {frames} frames")
        tracks = json.dumps(got)                  # whatever shape it was saved in, as the node's own
        log(f"  tracks {Path(job.comfy_tracks).name} \u00b7 {len(got)} paths"
            f" \u00b7 {len(got[0])} points each")
    pull = getattr(job, "comfy_strength", None)
    if tracks and pull is not None and w.get("strength"):
        log(f"  strength {float(pull):g}")
    fast = bool(getattr(job, "comfy_fast", False))
    if w.get("speed"):
        note = w["speed"].get("fast_note" if fast else "note", "")
        log(f"  {'fast' if fast else 'quality'}{f' · {note}' if note else ''}")
    graph = build(w, graph_of(w), job.prompt, names,
                  width, height, frames, job.seed, videos, tracks, pull, fast)
    prompt_id = submit(key, base, graph)
    log(f"queued on Comfy Cloud ({prompt_id})")
    detail = wait(key, base, prompt_id,
                  tick=lambda note: progress("cloud", 0, frames, note), cancelled=cancelled, log=log)
    if detail is None:
        return []

    files = outputs(detail, w.get("saver", ""), skip=video)   # `skip`: never hand back the clip we sent up
    if not files:
        raise CloudError("the cloud render finished but wrote no file")
    progress("downloading", 0, frames, "downloading the clip")
    log(f"downloading {files[0]['filename']}")
    got = fetch(key, base, files[0], out / "video.mp4", log=log)
    keep = getattr(job, "h3_sound", True) or not w.get("sound")
    log(f"downloaded video.mp4 ({got.stat().st_size / 2**20:.1f} MB"
        f"{', with sound' if w.get('sound') else ''})")
    if not keep:
        mute(got)
        log("dropped the audio track (Sound: silent)")

    extra = w.get("extra") or {}
    # strict: outputs() falls back to scanning every node, which would fetch the render a second time
    if extra.get("saver") in (detail.get("outputs") or {}):
        more = outputs(detail, extra["saver"], skip=video)
        if more:
            name = extra.get("name", "extra")
            side = fetch(key, base, more[0], out / f"{name}{Path(more[0]['filename']).suffix or '.mp4'}")
            log(f"downloaded {name} ({side.stat().st_size / 2**20:.1f} MB)")
            # it is a video you would want to look at, so it joins the runs strip rather than
            # sitting in the button bar as a file chip
            try:
                import runs as rr
                rid = rr.adopt(side, came=w.get("short") or w["id"], verb=name, rate=fps,
                               source=out.name)
                log(f"  run {rid} · {name} · {w.get('short') or w['id']}")
            except Exception as e:                # noqa: BLE001 - the render itself is what matters
                log(f"  couldn't make a run of the {name}: {e}")

    written = extract(got, out / "frames")
    progress("done", len(written), frames, "done")
    log(f"extracted {len(written)} frames for the viewer")
    return []                       # already an mp4: nothing for the caller to assemble
