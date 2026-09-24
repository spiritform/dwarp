# Models

DWARP reads models from one folder, `models_root` in `config.json`. The installer sets it to this
folder by default. If you already use ComfyUI, point it at ComfyUI's `models` folder instead:
DWARP uses the same layout and finds everything there.

```
models/
  checkpoints/      Stable Diffusion checkpoints (.safetensors); subfolders are fine
  controlnet/       ControlNet weights
  upscale_models/   ESRGAN-family upscalers (optional, for Enhance)
  rife/             RIFE frame interpolation (the installer fetches this)
```

## Start here (SD 1.5, ~3.6 GB)

Runs well on 8 GB GPUs. Download these into the folders below. **Keep the file names as they are.**

| File | Put it in | Size |
|---|---|---|
| [DreamShaper_8_pruned.safetensors](https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors) | `checkpoints/` | 2.1 GB |
| [control_v11f1p_sd15_depth_fp16.safetensors](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors) | `controlnet/` | 0.7 GB |
| [control_v11p_sd15_softedge_fp16.safetensors](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors) | `controlnet/` | 0.7 GB |

Any other SD 1.5 checkpoint works too. DWARP lists every one it finds under `checkpoints/`.

## SDXL (optional, ~9.4 GB, 12 GB GPU recommended)

| File | Put it in | Size |
|---|---|---|
| [DreamShaperXL_Turbo_v2_1.safetensors](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_v2_1.safetensors) | `checkpoints/` | 6.9 GB |
| [diffusion_pytorch_model_promax.safetensors](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors) → **rename to** `controlnet-union-sdxl-1.0-promax.safetensors` | `controlnet/` | 2.5 GB |

The ControlNet is xinsir's Union ProMax: one network that does both depth and soft edge, so
SDXL gets "depth + edge" in the memory of a single ControlNet.

Turbo, Lightning, Hyper and LCM checkpoints are fast (DWARP sets 8 steps, CFG 2 for them).

## Upscaling (optional)

| File | Put it in | Size |
|---|---|---|
| [4x-UltraSharp.pth](https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth) | `upscale_models/` | 67 MB |

Any ESRGAN-family `.pth` / `.safetensors` upscaler works.

---

ControlNet is optional: with it set to **off**, DWARP runs plain img2img, and the optical-flow
warp alone keeps frames coherent. So a single checkpoint is enough to start.
