"""Checkpoint discovery: which files in a models folder are usable SD1.5 / SD2 / SDXL checkpoints."""
from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

_kind_cache: dict = {}


class _Stub:
    """Anything in a checkpoint's pickle that isn't a plain container: never executed, just absorbed."""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *a, **k):
        return _Stub()


class _ShapeUnpickler(pickle.Unpickler):
    """Reads a torch .ckpt's index (data.pkl) without its tensor data or any code: tensors come back as
    their shapes, every other class as an inert stub."""
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, *a, **k: tuple(size)
        return _Stub

    def persistent_load(self, pid):
        return None


def ckpt_index(path: Path) -> dict:
    """{'keys': {weight name: shape}, 'global_step': int | None} for a zip-format .ckpt; {} if unreadable."""
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
            obj = _ShapeUnpickler(z.open(name)).load()
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    sd = obj.get("state_dict", obj)
    return {"keys": {k: v for k, v in sd.items() if isinstance(k, str) and isinstance(v, tuple)} if isinstance(sd, dict) else {},
            "global_step": obj.get("global_step") if isinstance(obj.get("global_step"), int) else None}


def classify(keys: dict) -> str | None:
    """Family from weight names / shapes: SD1 has a CLIP-L text encoder, SD2 OpenCLIP-H, SDXL two.
    in_channels 4 = a txt2img model (inpainting has 9)."""
    stem = keys.get("model.diffusion_model.input_blocks.0.0.weight")
    if not stem or tuple(stem)[1:2] != (4,):
        return None
    if any(k.startswith("conditioner.embedders.1.") for k in keys):     # base, not refiner
        return "sdxl"
    if any(k.startswith("cond_stage_model.model.") for k in keys):
        return "sd2"
    if any(k.startswith("cond_stage_model.transformer.") for k in keys):
        return "sd15"
    return None


_STORAGE_DTYPES = {"FloatStorage": "float32", "HalfStorage": "float16", "BFloat16Storage": "bfloat16",
                   "DoubleStorage": "float64", "LongStorage": "int64", "IntStorage": "int32",
                   "ShortStorage": "int16", "ByteStorage": "uint8", "CharStorage": "int8", "BoolStorage": "bool"}


def ckpt_to_safetensors(src: Path, dst: Path, log=print) -> Path:
    """A .ckpt's weights as an fp16 .safetensors, read without running the pickle: only tensors are
    rebuilt (from the zip's raw storages), everything else (training callbacks, optimizer classes) is
    an inert stub. Old SD 2.x files trip torch's weights_only loader; this reads them anyway, safely."""
    import torch
    from safetensors.torch import save_file

    with zipfile.ZipFile(src) as z:
        pkl = next(n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
        prefix = pkl[: -len("data.pkl")]

        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "collections" and name == "OrderedDict":
                    import collections
                    return collections.OrderedDict
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return lambda storage, offset, size, stride, *a, **k: storage.as_strided(tuple(size), tuple(stride), offset)
                if module == "torch" and name in _STORAGE_DTYPES:
                    return getattr(torch, _STORAGE_DTYPES[name])
                return _Stub

            def persistent_load(self, pid):          # ('storage', dtype, key, location, numel)
                dtype, key = pid[1], pid[2]
                return torch.frombuffer(bytearray(z.read(f"{prefix}data/{key}")), dtype=dtype)

        obj = Reader(z.open(pkl)).load()
    sd = obj.get("state_dict", obj)
    tensors = {k: (v.half() if v.is_floating_point() else v).contiguous().clone()
               for k, v in sd.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    save_file(tensors, str(tmp))
    tmp.replace(dst)
    log(f"converted {src.name} to fp16 safetensors ({dst.stat().st_size / 1e9:.1f} GB, once)")
    return dst


def converted_path(src: Path, cache_dir: Path) -> Path:
    """Where a .ckpt's converted copy lives; the name follows size and date, so an edited file converts again."""
    st = src.stat()
    return cache_dir / f"{src.stem}-{st.st_size}-{int(st.st_mtime)}.safetensors"


def sd2_prediction(path: Path) -> str:
    """SD 2.x: the 512 base models predict noise, the 768 ones v. diffusers only knows 2.0-base's step
    (875000) — 2.1-base (v2-1_512, step 220000) would come out v and render mush — so a "512" in the
    name counts too. Decided here because DWARP swaps the scheduler after loading."""
    if "512" in path.name:
        return "epsilon"
    if path.suffix.lower() == ".ckpt" and ckpt_index(path).get("global_step") == 875000:
        return "epsilon"
    return "v_prediction"


def model_family(path: Path) -> str | None:
    """'sd15' / 'sd2' / 'sdxl' for plain txt2img checkpoints of those families, 'maybe' for an
    unreadable SD1.5-sized .ckpt, else None (inpainting, refiner, video…). Reads only the
    safetensors JSON header or the .ckpt's index — never the weights."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    if key in _kind_cache:
        return _kind_cache[key]
    kind = None
    if path.suffix.lower() == ".ckpt":
        idx = ckpt_index(path)
        if idx:
            kind = classify(idx["keys"])
        else:                                   # old / odd pickle: fall back to SD1.5 file sizes
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
            elif any(k.startswith("cond_stage_model.model.") for k in header):
                kind = "sd2"
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
                          "family": "sd15" if kind == "maybe" else kind, "verified": kind != "maybe"})
    return found
