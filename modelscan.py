"""What is on disk, answered without loading a deep-learning framework.

Listing the LoRAs in a folder means reading a filename and a few hundred bytes of JSON out of each
file's header. It does not mean importing torch. But these helpers used to live in `engine/warp.py`,
which imports torch at module scope, so the page asking "which LoRAs do I have?" paid for a cold
CUDA DLL load first: `/api/loras` measured **50 s** on its first call and 533 ms on its second.

So they live here instead, in a module that imports nothing heavier than the standard library. The
engine re-exports them, so the render path is unchanged and there is one copy of each. The single
case that genuinely needs torch -- a `.pt` embedding, which is a pickle -- imports it inside the
branch that reads one.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

EMBED_EXTS = (".pt", ".safetensors", ".bin")

# A directory walk, remembered for as long as the folder has not been written to. Scanning is cheap
# once the OS has the directory cached and dear when it has not; this is what keeps the second
# question of a session off the disk entirely.
_walks: dict[tuple, tuple[float, list[Path]]] = {}


def _walk(root: Path, pattern: str = "*") -> list[Path]:
    """Every file under `root` matching `pattern`, cached against the folder's own mtime.

    The mtime of a directory changes when an entry is added or removed, which is exactly when this
    answer goes stale. Editing a file in place does not change it, but nothing here reads contents,
    so that does not matter.
    """
    try:
        stamp = os.stat(root).st_mtime
    except OSError:
        return []
    key = (str(root), pattern)
    hit = _walks.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    found = sorted(p for p in root.rglob(pattern) if p.is_file())
    _walks[key] = (stamp, found)
    return found


def embedding_files(dirs) -> dict[str, Path]:
    """Embedding name (the file name, as typed in a prompt) -> file, first folder wins."""
    found: dict[str, Path] = {}
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            for f in _walk(d):
                if f.suffix.lower() in EMBED_EXTS and f.stem not in found:
                    found[f.stem] = f
    return found


def lora_files(folder: Path) -> list[Path]:
    """Every .safetensors under a LoRA folder, in name order."""
    folder = Path(folder)
    return _walk(folder, "*.safetensors") if folder.is_dir() else []


def lora_family(f: Path) -> str | None:
    """What a LoRA was trained for, from a cross-attention layer's input width (the text features):
    768 = SD 1.5, 1024 = SD 2.x, 2048 = SDXL. Reads only the safetensors header. None = not an SD LoRA
    (Flux, video models…) or unreadable."""
    try:
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n)) if 0 < n < 100_000_000 else {}
    except (OSError, ValueError):
        return None
    for k, v in header.items():
        low = k.lower()
        if "attn2" in low and ("to_k" in low or "k_proj" in low) and ("lora_down" in low or "lora_a" in low) \
                and isinstance(v, dict) and len(v.get("shape", [])) == 2:
            return {768: "sd15", 1024: "sd2", 2048: "sdxl"}.get(v["shape"][1])
    return None


def embedding_family(f: Path) -> str | None:
    """What an embedding was trained for, from its vector width: 768 = SD 1.5, 1024 = SD 2.x, clip_g /
    clip_l pairs = SDXL. None if unreadable. Only the tensors' shapes matter; files are small."""
    try:
        if f.suffix.lower() == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(f), "pt") as st:
                keys = list(st.keys())
                if any("clip_g" in k for k in keys):
                    return "sdxl"
                widths = {st.get_slice(k).get_shape()[-1] for k in keys}
        else:
            widths = _pickled_widths(f)
            if widths is None:
                return None
    except Exception:
        return None
    return "sd15" if widths == {768} else "sd2" if widths == {1024} else None


def _pickled_widths(f: Path) -> set | None:
    """The vector widths inside a .pt / .bin embedding.

    These are pickles, and the obvious reader is `torch.load` -- a CUDA DLL load to learn one
    number. `models.zip_pickle` unpickles the shapes instead, with every tensor stubbed out, which
    is what `ckpt_index` already does for .ckpt files. Only a legacy, non-zip file falls back to
    torch, and then only that one file pays for it.
    """
    from models import zip_pickle
    obj = zip_pickle(f)
    if obj is None:
        try:
            import torch
            d = torch.load(str(f), map_location="cpu", weights_only=True)
        except Exception:
            return None
        d = d.get("string_to_param", d) if isinstance(d, dict) else {}
        return {v.shape[-1] for v in d.values() if hasattr(v, "shape")}
    d = obj.get("string_to_param", obj) if isinstance(obj, dict) else {}
    if not isinstance(d, dict):
        return None
    return {v[-1] for v in d.values() if isinstance(v, (tuple, list)) and v}


def animatelcm_lora(motion: str) -> str | None:
    """AnimateLCM's spatial LoRA (AnimateLCM_sd15_t2v_lora), looked for in the models root's loras."""
    for root in (Path(motion).parents[1] / "loras", Path(__file__).resolve().parent / "models"):
        hit = next((p for p in lora_files(root) if re.search(r"animatelcm.*lora", p.name, re.I)), None)
        if hit:
            return str(hit)
    return None
