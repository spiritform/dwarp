# -*- coding: utf-8 -*-
"""The H3 brief, composed from what is loaded rather than typed.

H3 does not read a pile of style words. It reads a short document in labelled sections, and inside
``retention_analysis`` it reads particular words -- ``fully_preserved``, ``attribute_transfer``,
``weak_reference`` -- that decide how hard each thing you handed it binds. Those words are the
difference between a clip that follows your plate and one that ignores it, and nobody should have
to know them to get the second.

So the panel asks plain questions -- what is the shot, what should this clip hold on to, what is
this picture for -- and this composes the document. The wording lives in ``comfy/briefs.json``
beside the graphs, so a workflow that wants different words is a data edit, not a page edit.

``compose`` is the only entry point. It is a pure function over a dict, which means the page can
ask for a preview of exactly what will go up before anything is spent.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIEFS = HERE / "comfy" / "briefs.json"

SECTIONS = ["subject_definitions", "summary", "retention_analysis", "video_contexts",
            "detailed_description", "overall_soundscape", "non_diegetic_music"]
FEATURES = "Features to follow exactly:"


def _data() -> dict:
    """Read fresh: a brief is text, and text gets tuned while the server is up."""
    try:
        return json.loads(BRIEFS.read_text(encoding="utf-8"))
    except Exception:
        return {"roles": {"video": [], "picture": []}, "recipes": []}


def catalogue() -> dict:
    """What the panel draws its form from: the starting points, and the ways a slot can bind."""
    d = _data()
    def pub(r: dict) -> dict:
        return {k: v for k, v in r.items() if k not in ("line",)}
    return {"recipes": [pub(r) for r in d.get("recipes") or []],
            "roles": {kind: [pub(r) for r in rs]
                      for kind, rs in (d.get("roles") or {}).items()}}


def _role(d: dict, kind: str, rid: str, fallback: str = "") -> dict:
    rs = (d.get("roles") or {}).get(kind) or []
    for r in rs:
        if r.get("id") == rid:
            return r
    for r in rs:
        if r.get("id") == fallback:
            return r
    return rs[0] if rs else {}


def _recipe(d: dict, rid: str) -> dict:
    for r in d.get("recipes") or []:
        if r.get("id") == rid:
            return r
    return {}


def compose(spec: dict) -> str:
    """The brief for one job.

    spec: recipe, seconds, shot, sound, music, moments,
          videos   [{tag, what, role, legend}]   -- <Video 1> is the panel's own clip or its plate
          pictures [{tag, name, role, label}]    -- <Picture n>, labelled when the workflow names them
    """
    d = _data()
    r = _recipe(d, str(spec.get("recipe") or ""))
    secs = float(spec.get("seconds") or 0)
    vids = list(spec.get("videos") or [])
    pics = list(spec.get("pictures") or [])
    # A workflow whose picture slots are moments on the clock (first frame, last frame) does not have
    # subjects to hold: the pictures are instants, and what they bind to is the time code.
    moments = bool(spec.get("moments") or r.get("moments"))

    def vrole(v):
        return _role(d, "video", v.get("role") or r.get("video") or "shot", "shot")

    def prole(p):
        return _role(d, "picture", p.get("role") or r.get("picture") or "identity", "identity")

    out: list[str] = []

    # ---- who is in it -------------------------------------------------------------------------
    if pics and not moments:
        out += ["subject_definitions:"]
        for i, p in enumerate(pics, 1):
            role = prole(p)
            named = role.get("role") or role.get("label") or "reference"
            # A picture that carries the look is not someone in the shot, and saying it is invites
            # the model to put the reference's own subject into the frame.
            if role.get("restyle"):
                out += [f"<Picture {i}> is a style reference, not a subject: "
                        f"{named}. Take nothing from it but the way it is made — not its "
                        f"subject, not its composition, not its background."]
            else:
                out += [f"<Subject {i}> is the subject of <Picture {i}>. Reference role: {named}.",
                        f"Do not carry over the background of <Picture {i}>."]
            if p.get("features"):
                out += [f"{FEATURES} {p['features']}"]
        out += [""]

    # ---- what it is ---------------------------------------------------------------------------
    # Only the pictures that are someone get named here: a style anchor is not in the shot.
    named = [i for i, p in enumerate(pics, 1) if not moments and prole(p).get("id") == "identity"]
    who = (" with " + " and ".join(f"<Subject {i}>" for i in named) + " in it") if named else ""
    follow = ", following <Video 1>" if vids else ""
    out += ["summary:",
            f"A {secs:.1f}-second single continuous shot{who}{follow}. No cut.", ""]

    # ---- how hard each thing binds: the part that actually steers the model --------------------
    out += ["retention_analysis:"]
    if moments:
        out += _moments(pics, secs)
    else:
        for i, p in enumerate(pics, 1):
            tag = f"<Picture {i}>" if prole(p).get("restyle") else f"<Subject {i}>"
            out += [f"{tag}: {prole(p).get('line', '')}"]
    for v in vids:
        out += [f"<Video {v.get('tag') or 1}>: {vrole(v).get('line', '')}"]
    out += [""]

    # ---- which clip is which, and what a mask's colours mean ----------------------------------
    labelled = [v for v in vids if v.get("what") or v.get("legend")]
    if len(vids) > 1 or any(v.get("legend") for v in vids):
        out += ["video_contexts:"]
        for v in labelled:
            tag = f"<Video {v.get('tag') or 1}>"
            if v.get("legend"):
                out += [f"{tag}: a colour mask. It is an instruction, never something to draw: "
                        f"{v['legend']}. Follow it frame by frame — wherever a colour is, that "
                        f"video's content belongs there, and the mask's own colours never appear in "
                        f"the picture."]
            else:
                out += [f"{tag}: {v['what']}."]
        out += [""]

    # ---- the one thing that is yours -----------------------------------------------------------
    shot = (spec.get("shot") or "").strip()
    out += ["detailed_description:", shot or "...", ""]

    # ---- sound ----------------------------------------------------------------------------------
    # One line describes the whole track. Say nothing and both sections take their defaults; say
    # something and it is the soundscape, with no music section left behind to contradict it.
    sound = (spec.get("sound") or "").strip()
    music = (spec.get("music") or "").strip()
    if sound or music:
        out += ["overall_soundscape:", sound or d.get("sound_default") or "..."]
        if music:
            out += ["", "non_diegetic_music:", music]
    else:
        out += ["overall_soundscape:", d.get("sound_default") or "...",
                "", "non_diegetic_music:", d.get("music_default") or "..."]
    return "\n".join(out).strip() + "\n"


def _moments(pics: list, secs: float) -> list:
    """First to last: the pictures are instants, and the model is told where on the clock each sits."""
    lines = []
    if pics:
        lines.append("<Picture 1> is the first frame: it aligns with the 0.00-second mark of the "
                     "target video.")
    if len(pics) > 1:
        lines.append(f"<Picture 2> is the last frame: it aligns with the {secs:.2f}-second mark.")
    else:
        lines.append("<Picture 2> is missing: add a second picture for the frame it should close on.")
    lines.append("Both are fully_preserved - the shot opens exactly on one and settles exactly into "
                 "the other.")
    return lines
