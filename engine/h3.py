"""H3: a clip with sound from reference pictures, rendered on Comfy Cloud.

MiniMax-H3 is open weights, but the parts that matter are a 61 GB transformer and a 32 B conditioner, and
the small quantised ones on the Hub are pruned in a way diffusers refuses. Running it here would mean ~20 GB
of RAM before the card is even touched. So this mode doesn't run it here: DWARP sends the job to Comfy Cloud
and brings back the mp4. The only thing it needs is an API key (Settings -> Comfy Cloud), and the only thing
it costs is Comfy credits.

The model is Pika H3 Flex, a Pika adapter on MiniMax-H3 that lands a finished 1344x768 clip in six sampling
steps. `comfy/pika_h3_ref2va.json` is Comfy's own `video_pika_h3_flex_ref2va` template in API form, with the
resolution and duration helper nodes taken out — this file computes both and writes them straight onto the
H3 node, so there is less graph to get wrong.

Reference to video: up to nine pictures go in, and whatever is in them — a face, a costume, a painted DWARP
frame's whole look — carries into a new clip that the prompt describes. Refer to them in the prompt as
<Picture 1>, <Picture 2> ... in the order they were added; that tagging is how H3 was trained to read them.

Unlike every other DWARP mode this one generates sound as well, so the mp4 it writes is kept as it arrives
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

HERE = Path(__file__).resolve().parent
GRAPH = HERE / "comfy" / "pika_h3_ref2va.json"
CLOUD = "https://cloud.comfy.org"

FPS = 24                       # H3 is fixed at 24 fps
SNAP = 17                      # and at 17n + 5 frames: 124 (~5.2 s) is the shortest it will make
MIN_FRAMES = 124
MAX_REFS = 9
SAVER = "92"                   # the graph's SaveVideo: the one node whose output is the render
TIMEOUT = 45                   # per request; a render takes minutes, but each poll is small
POLL = 3.0


class CloudError(RuntimeError):
    """Anything Comfy Cloud said no to, phrased for the page's log."""


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


def _call(key: str, base: str, path: str, data: bytes | None = None, ctype: str = "", method: str = ""):
    req = urllib.request.Request(f"{base}{path}", data=data, method=method or ("POST" if data else "GET"))
    req.add_header("X-API-Key", key)
    req.add_header("User-Agent", "DWARP")
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with _opener.open(req, timeout=TIMEOUT) as r:
            body = r.read()
    except _Redirect:
        raise
    except urllib.error.HTTPError as e:
        detail = (e.read() or b"")[:400].decode("utf-8", "replace").strip()
        raise CloudError({
            401: "Comfy Cloud rejected the API key — check it in Settings.",
            402: "Comfy Cloud: out of credits.",
            429: "Comfy Cloud: this account has no active subscription (a render needs one).",
        }.get(e.code, f"Comfy Cloud {e.code} on {path}{': ' + detail if detail else ''}")) from e
    except urllib.error.URLError as e:
        raise CloudError(f"couldn't reach Comfy Cloud ({e.reason})") from e
    return json.loads(body) if body[:1] in (b"{", b"[") else body


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
    r = _call(key, base, "/api/upload/image", body, ctype)
    if not isinstance(r, dict) or not r.get("name"):
        raise CloudError(f"upload of {path.name} gave back no file name")
    return r["name"]


def _asset(r) -> dict:
    """The asset out of an /api/assets answer, whether it came back bare or wrapped."""
    if not isinstance(r, dict):
        return {}
    inner = r.get("asset")
    return inner if isinstance(inner, dict) else r


def upload_video(key: str, base: str, path: Path) -> str:
    """A clip into the asset store. /api/upload/image is for pictures only, so a video goes through
    /api/assets.

    LoadVideo names its files by content, not by what the file was called here, so the answer's `hash`
    is the name to put in the graph — it already carries the extension ("b5def...157a.mp4"). Sending
    `display_name` instead gets "the input file ... doesn't exist" back from the node. `asset_hash`
    (`blake3:<hex>`, no extension) is the spelling in the API docs; both are accepted here."""
    body, ctype = _multipart({"tags": "input", "name": path.name,
                              "mime_type": "video/mp4"}, "file", path)
    asset = _asset(_call(key, base, "/api/assets", body, ctype))
    if not asset:
        raise CloudError(f"upload of {path.name} gave back no asset")
    name = asset.get("hash") or ""
    if not name and asset.get("asset_hash"):
        name = asset["asset_hash"].split(":")[-1] + (path.suffix or ".mp4")
    if not name:
        raise CloudError("Comfy Cloud stored the clip but gave back no content hash, so the graph "
                         f"can't name it: {json.dumps(asset)[:300]}")
    return name


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


def frame_count(seconds: float) -> int:
    """H3 takes 17n + 5 frames at 24 fps and nothing shorter than 124 (~5.2 s)."""
    n = max(MIN_FRAMES, round(max(0.0, seconds) * FPS))
    return n + (5 - n % SNAP) % SNAP


def fit_size(width: int, height: int, megapixels: float = 1.03, multiple: int = 32) -> tuple[int, int]:
    """The asked-for shape at H3's working area, on the multiple of 32 it needs. 1.03 MP is what puts
    16:9 on exactly the 1344x768 Pika Flex is tuned for."""
    w, h = max(1, int(width)), max(1, int(height))
    k = (megapixels * 1_000_000 / (w * h)) ** 0.5
    fit = lambda v: max(multiple, int(round(v * k / multiple)) * multiple)        # noqa: E731
    return fit(w), fit(h)


def build(graph: dict, prompt: str, refs: list[str], width: int, height: int,
          frames: int, seed: int, video: str = "", video_audio: bool = False) -> dict:
    """The shipped template with this job's values on it. The reference pictures replace the two the
    template carries: one LoadImage per picture, wired into ref_image_0, ref_image_1, ...

    A source clip joins them as <Video 1>: LoadVideo -> GetVideoComponents gives the frames (and the
    sound, as <Audio 1>) that the H3 node's ref_videos / ref_video_audios slots take."""
    g = {k: {**v, "inputs": dict(v["inputs"])} for k, v in graph.items()}
    g["138"]["inputs"]["value"] = prompt
    g["129"]["inputs"]["noise_seed"] = int(seed) % (2 ** 63)
    h3 = g["136"]["inputs"]
    h3["width"], h3["height"], h3["length"] = int(width), int(height), int(frames)

    for node_id, node in list(g.items()):                       # the template's own two pictures
        if node["class_type"] == "LoadImage":
            del g[node_id]
    for slot in [k for k in h3 if k.startswith("ref_images.")]:
        del h3[slot]
    if not refs and not video:
        raise CloudError("H3 needs a source clip or at least one reference picture")
    for i, name in enumerate(refs[:MAX_REFS]):
        node_id = str(900 + i)
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name},
                      "_meta": {"title": f"Picture {i + 1}"}}
        h3[f"ref_images.ref_image_{i}"] = [node_id, 0]

    if video:
        g["910"] = {"class_type": "LoadVideo", "inputs": {"file": video}, "_meta": {"title": "Video 1"}}
        g["911"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["910", 0]},
                    "_meta": {"title": "Video 1 frames"}}
        h3["ref_videos.ref_video_0"] = ["911", 0]             # IMAGE: the frames
        if video_audio:
            h3["ref_video_audios.ref_video_audio_0"] = ["911", 1]   # AUDIO: what it sounded like
    return g


def submit(key: str, base: str, graph: dict) -> str:
    r = _call(key, base, "/api/prompt", json.dumps({"prompt": graph}).encode(), "application/json")
    if not isinstance(r, dict):
        raise CloudError("Comfy Cloud gave an unexpected answer to the job")
    if r.get("node_errors"):
        raise CloudError(f"the graph was rejected: {json.dumps(r['node_errors'])[:300]}")
    if not r.get("prompt_id"):
        raise CloudError(f"no job id came back: {json.dumps(r)[:300]}")
    return r["prompt_id"]


def outputs(detail: dict, saver: str = SAVER, skip: str = "") -> list[dict]:
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


def fetch(key: str, base: str, f: dict, dest: Path) -> Path:
    """Two steps on purpose: read where the file lives, then fetch it with no API key attached."""
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                "type": f.get("type", "output")})
    try:
        body = _call(key, base, f"/api/view?{q}")
        dest.write_bytes(body if isinstance(body, bytes) else json.dumps(body).encode())
        return dest
    except _Redirect as r:
        with urllib.request.urlopen(r.url_to, timeout=TIMEOUT * 4) as got:   # signed URL, no headers
            dest.write_bytes(got.read())
        return dest


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
def render_h3(job, progress: Callable = lambda *a: None, cancelled: Callable[[], bool] = lambda: False,
              log: Callable[[str], None] = print) -> list[Path]:
    """Called by engine.warp.render for family "h3". Returns [] so the caller doesn't reassemble the mp4
    (that would lose the audio); the file is already written."""
    key, base = api_key(), base_url()
    if not key:
        raise CloudError("no Comfy Cloud API key — add one in Settings. "
                         "Keys come from platform.comfy.org/profile/api-keys.")

    out = Path(job.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    refs = [Path(p) for p in (getattr(job, "h3_refs", None) or []) if p and Path(p).is_file()]
    source = getattr(job, "h3_video", "") or ""
    if not refs and not source:
        raise CloudError("H3 needs a source clip, or a picture in a Reference slot.")

    frames = frame_count(getattr(job, "h3_seconds", 5.0))
    width, height = fit_size(job.width, job.height)
    parts = ([f"{len(refs)} reference{'s' if len(refs) > 1 else ''}"] if refs else []) +             (["the source clip"] if source else [])
    log(f"H3 on Comfy Cloud · {width}x{height} · {frames} frames ({frames / FPS:.1f}s) · {' + '.join(parts)}")

    progress("uploading", 0, frames, "sending the references")
    names = []
    for i, p in enumerate(refs[:MAX_REFS], 1):
        names.append(upload(key, base, p))
        log(f"  <Picture {i}> {p.name}")

    video, sound = "", False
    if source and not cancelled():
        progress("uploading", 0, frames, "sending the clip")
        trim = getattr(job, "h3_trim", None) or []
        a, span, clipped = trim_span(trim)
        seg = clip_segment(source, trim, out / "h3_reference.mp4")
        sound = has_audio(seg)
        video = upload_video(key, base, seg)
        log(f"  <Video 1> {Path(source).name} · {a:.1f}s to {a + span:.1f}s"
            f"{' (cut to the 15 s H3 looks at)' if clipped else ''} · {seg.stat().st_size / 2**20:.1f} MB"
            f"{', with its sound as <Audio 1>' if sound else ', no sound track'}")
        log(f"     stored as {video}")      # the content name the graph asks LoadVideo for
        seg.unlink(missing_ok=True)
    if cancelled():
        return []

    graph = build(json.loads(GRAPH.read_text(encoding="utf-8")), job.prompt, names,
                  width, height, frames, job.seed, video, sound)
    prompt_id = submit(key, base, graph)
    log(f"queued on Comfy Cloud ({prompt_id})")

    # Comfy's REST API reports no percentage, so the bar can't move and sits at 0 until the clip
    # arrives. Elapsed time is the only honest signal there is, so it goes out every poll and into
    # the log every half minute — otherwise a four-minute render looks like a hung one.
    t0, seen, beat = time.time(), "", 0.0
    while True:
        if cancelled():
            try:
                _call(key, base, "/api/queue", json.dumps({"delete": [prompt_id]}).encode(),
                      "application/json")
                log("cancelled on Comfy Cloud")
            except CloudError as e:
                log(f"couldn't cancel the cloud job: {e}")
            return []
        detail = _call(key, base, f"/api/jobs/{prompt_id}")
        state = (detail.get("status") or "") if isinstance(detail, dict) else ""
        mins, secs = divmod(round(time.time() - t0), 60)
        label = {"pending": "waiting for a GPU", "in_progress": "rendering"}.get(state, state or "working")
        progress("cloud", 0, frames, f"{label} on Comfy Cloud · {mins}m {secs:02d}s")
        waited = time.time() - t0
        if state != seen or waited - beat >= 30:
            log(f"{label} ({mins}m {secs:02d}s)")
            seen, beat = state, waited
        if state == "completed":
            break
        if state in ("failed", "error", "cancelled"):
            why = (detail.get("error_message") or
                   json.dumps(detail.get("execution_error") or {})[:300] or state)
            raise CloudError(f"the cloud render {state}: {why}")
        time.sleep(POLL)

    files = outputs(detail, skip=video)       # `skip`: never hand back the clip we just sent up
    if not files:
        raise CloudError("the cloud render finished but wrote no file")
    progress("downloading", 0, frames, "downloading the clip")
    log(f"downloading {files[0]['filename']}")
    got = fetch(key, base, files[0], out / "video.mp4")
    log(f"downloaded video.mp4 ({got.stat().st_size / 2**20:.1f} MB, with sound)")

    written = extract(got, out / "frames")
    progress("done", len(written), frames, "done")
    log(f"extracted {len(written)} frames for the viewer")
    return []                       # already an mp4: nothing for the caller to assemble
