"""Checkpoint discovery: which files in a models folder are usable SD1.5 / SDXL checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

_kind_cache: dict = {}


def model_family(path: Path) -> str | None:
    """'sd15' / 'sdxl' for plain txt2img checkpoints of those families, 'maybe' for an
    unverifiable SD1.5-sized .ckpt, else None (SD2, inpainting, refiner, v-pred, video…).
    Reads only the safetensors JSON header."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    if key in _kind_cache:
        return _kind_cache[key]
    kind = None
    if path.suffix.lower() == ".ckpt":
        # Pickles can't be inspected without executing them; SD1.5 full/pruned sizes only.
        gb = st.st_size / 1e9
        kind = "maybe" if 1.9 < gb < 2.3 or 3.9 < gb < 4.4 or 7.0 < gb < 7.8 else None
    else:
        try:
            with open(path, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                header = json.loads(fh.read(n)) if 0 < n < 50_000_000 else {}
        except (OSError, ValueError):
            header = {}
        stem = header.get("model.diffusion_model.input_blocks.0.0.weight", {})
        sd1_te = any(k.startswith("cond_stage_model.transformer.") for k in header)
        sdxl = any(k.startswith("conditioner.embedders.1.") for k in header)   # base, not refiner
        vpred = "v_pred" in header
        # in_channels 4 = txt2img model (inpainting checkpoints have 9)
        if stem.get("shape", [0, 0])[1:2] == [4] and not vpred:
            if sdxl:
                kind = "sdxl"
            elif sd1_te:
                kind = "sd15"
    _kind_cache[key] = kind
    return kind


def list_checkpoints(root: str) -> list[dict]:
    found = []
    base = Path(root) if root else None
    if not base or not base.is_dir():
        return found
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() not in (".safetensors", ".ckpt") or not path.is_file():
            continue
        kind = model_family(path)
        if kind:
            found.append({"path": path.as_posix(), "name": path.relative_to(base).as_posix(),
                          "family": "sdxl" if kind == "sdxl" else "sd15", "verified": kind != "maybe"})
    return found
