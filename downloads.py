"""Models that download on first use.

DWARP installs without models. Every file it knows how to fetch is listed here by the name DWARP looks for,
with where it comes from (a Hugging Face repo + file), its size and SHA-256. When a render needs one that
isn't on disk, `fetch()` streams it into place, checks the hash as it goes, and only then moves it in. The
server marks missing-but-known files so the panel can say what a first render will download, and how much.
"""
from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path

GB = 1e9

# file name DWARP looks for -> (repo, file in the repo, bytes, sha256); a repo that's an https:// URL is the file itself
KNOWN: dict[str, tuple[str, str, int, str]] = {
    # SD 1.5 checkpoints (the Looks' default)
    "dreamshaper_8.safetensors": ("Lykon/DreamShaper", "DreamShaper_8_pruned.safetensors", 2132625894,
                                  "879db523c30d3b9017143d56705015e15a2cb5628762c11d086fed9538abd7fd"),
    # SDXL default: Flash Mini (SSD-1B based, fast)
    "SDXL-Flash_Mini.safetensors": ("sd-community/sdxl-flash-mini", "SDXL-Flash_Mini.safetensors", 4465671322,
                                    "2abc508f756ac79687d1cd7b520aa9e4b93b76dc9c7b9e9ace24da4762ea1381"),
    # SD 2.1 (768-v), from the community mirror (Stability's repo now needs a login)
    "v2-1_768-ema-pruned.safetensors": ("sd2-community/stable-diffusion-2-1", "v2-1_768-ema-pruned.safetensors", 5214604494,
                                        "dcd690123cfc64383981a31d955694f6acf2072a80537fdb612c8e58ec87a8ac"),
    # SD 1.5 ControlNets (fp16)
    "control_v11f1p_sd15_depth_fp16.safetensors": ("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11f1p_sd15_depth_fp16.safetensors",
                                                   722601100, "1c4a79aa52fb63f607cb9ff479ea5aa1923b6ceb21267bd14b69bd05d7b617be"),
    "control_v11p_sd15_softedge_fp16.safetensors": ("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_softedge_fp16.safetensors",
                                                    722601100, "e78fea5b4599fec2ecd7e3f14b171feb290b88200c95d569ec0ff59a19bc3478"),
    "control_v11f1e_sd15_tile_fp16.safetensors": ("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11f1e_sd15_tile_fp16.safetensors",
                                                  722601104, "2f31868eedb243a77932e3c63907a6ba0a2058b6d65b5c27b89ee1b7f618ea33"),
    # QR Code Monster
    "control_v1p_sd15_qrcode_monster_v2.safetensors": ("monster-labs/control_v1p_sd15_qrcode_monster", "v2/control_v1p_sd15_qrcode_monster_v2.safetensors",
                                                       722596344, "fc985da5850a03033c9e28032532f406ae04bd127178ae5bc6d3ec0502b25253"),
    "control_v1p_sdxl_qrcode_monster.safetensors": ("monster-labs/control_v1p_sdxl_qrcode_monster", "diffusion_pytorch_model.safetensors",
                                                    5004167864, "11e49b4e272fabda60094b35be6fd3e215e551211210938e381d27497fd2215c"),
    # SDXL: xinsir's Union ProMax (depth + soft edge + tile in one)
    "controlnet-union-sdxl-1.0-promax.safetensors": ("xinsir/controlnet-union-sdxl-1.0", "diffusion_pytorch_model_promax.safetensors",
                                                     2513342408, "9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc"),
    # Motion (AnimateDiff)
    "temporaldiff-v1-animatediff.safetensors": ("CiaraRowles/TemporalDiff", "temporaldiff-v1-animatediff.safetensors",
                                                1672020256, "016275b9e5ce42edaf4726970d6092d6a9a511b356a26821333c4b045b0dac2a"),
    "mm_sd_v15_v2.ckpt": ("guoyww/animatediff", "mm_sd_v15_v2.ckpt",
                          1817888431, "69ed0f5fef82b110aca51bcab73b21104242bc65d6ab4b8b2a2a94d31cad1bf0"),
    "AnimateLCM_sd15_t2v.ckpt": ("wangfuyun/AnimateLCM", "AnimateLCM_sd15_t2v.ckpt",
                                 1813041929, "b46c3de62e5696af72c4056e3cdcbea12fbc19581c0aad7b6f2b027851148f5f"),
    "AnimateLCM_sd15_t2v_lora.safetensors": ("wangfuyun/AnimateLCM", "AnimateLCM_sd15_t2v_lora.safetensors",
                                             134621556, "8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4"),
    # ZipDepth (the light depth model), from the author's GitHub: a direct URL in place of a repo
    "zipdepth_base.pth": ("https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base.pth", "",
                          27298978, "a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65"),
}


def known(path: str | Path) -> tuple[str, str, int, str] | None:
    return KNOWN.get(Path(path).name)


def size(path: str | Path) -> int:
    k = known(path)
    return k[2] if k else 0


def missing(path: str | Path) -> bool:
    """A file DWARP can fetch that isn't on disk yet."""
    return bool(path) and not Path(path).is_file() and known(path) is not None


def fetch(path: str | Path, log=print, progress=None) -> bool:
    """Download a known model to `path` if it isn't there. Streams to `<name>.part`, hashing as it goes; a
    mismatch deletes it and raises. Returns True if it downloaded. `progress(done_bytes, total_bytes)`."""
    p = Path(path)
    if p.is_file() or not known(p):
        return False
    repo, rfile, total, sha = known(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    part = p.with_name(p.name + ".part")
    url = repo if repo.startswith("https://") else f"https://huggingface.co/{repo}/resolve/main/{rfile}"
    log(f"downloading {p.name} ({total / GB:.2f} GB) from {repo.split('/')[2] if '://' in repo else repo} — first use, once…")
    t, h, done, said = time.time(), hashlib.sha256(), 0, -1
    req = urllib.request.Request(url, headers={"User-Agent": "dwarp"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as fh:
            while chunk := r.read(4 << 20):
                fh.write(chunk); h.update(chunk); done += len(chunk)
                pct = int(done * 100 / total) if total else 0
                if progress:
                    progress(done, total)
                if pct // 10 != said:                # a log line every 10%
                    said = pct // 10
                    log(f"  {p.name}: {pct}% ({done / GB:.2f} / {total / GB:.2f} GB)")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    if h.hexdigest() != sha:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{p.name} didn't match its checksum — deleted; try again")
    os.replace(part, p)
    log(f"downloaded {p.name} in {time.time() - t:.0f}s (checksum ok)")
    return True
