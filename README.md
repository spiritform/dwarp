# DWARP

**Diffusion warp** — turn a video into a painted, drawn or dreamed version of itself, frame by frame, locally on your GPU.

DWARP repaints each frame with Stable Diffusion, but instead of starting every frame from scratch it
carries the previous painted frame forward along the video's own motion (optical flow). Brushstrokes
travel with the things they belong to, so the result moves like the footage instead of flickering.

One screen: drop a clip, pick a model and a style, write what's in the shot, hit **Single Frame** or **Preview**.

## How it works

For every frame:

1. **Flow** — [RAFT](https://github.com/princeton-vl/RAFT) optical flow between the previous and current source frame, both directions.
2. **Warp** — the previous *stylized* frame is pushed forward along that flow.
3. **Trust** — a forward/backward consistency check masks out occlusions, new content and off-screen areas; those fall back to the raw source frame.
4. **Repaint** — img2img from that blend, steered by ControlNets (depth + soft edge) computed on the source frame. The first frame gets the full style strength; later frames a lower one, so they refine what's carried forward instead of re-rolling it.
5. **Colour** — each frame's colour statistics are pulled toward frame 0 to stop feedback drift.

## Features

- SD 1.5 and SDXL checkpoints — reads your ComfyUI models folder and sorts checkpoints by family from their headers
- Style presets, Style (how much it repaints) and Hold (how tightly it follows the footage) sliders, steps, CFG, lockable seed
- **DepthDiff**: the source's luma or depth decides where the style repaints, per pixel ([differential diffusion](https://differential-diffusion.github.io/)). Keep the subject and dissolve the room, or the other way round. Works together with ControlNet
- **Single Frame** renders just the first frame; **Preview** renders 5 frames (every 2nd) in seconds; **Render** does the whole clip
- Live split view (source / output), frame scrubber, live log, run history with reload-settings and delete
- **Enhance**: upscale any finished run with your ESRGAN-family models (via [spandrel](https://github.com/chaiNNer-org/spandrel)) and smooth it with [RIFE](https://github.com/hzwer/Practical-RIFE) frame interpolation

## Requirements

- Windows 10/11 with an NVIDIA GPU: 8 GB+ VRAM for SD 1.5, 12 GB recommended for SDXL
- ~6 GB of disk for DWARP itself (PyTorch + CUDA), plus your models

## Install

1. Download or clone this repo.
2. Double-click **`install.bat`**. It sets everything up inside the folder: its own Python,
   PyTorch with CUDA, ffmpeg if you don't have it, and the RIFE weights. It asks once where
   your models are. Press Enter to use DWARP's own `models` folder, or paste a ComfyUI `models`
   folder to reuse what you already have.
3. Add models: see **[models/README.md](models/README.md)** for a short list of links. A single
   SD 1.5 checkpoint is enough to start.
4. Double-click **`run.bat`**. DWARP opens at http://localhost:8013.

Re-running `install.bat` is safe: finished steps are skipped. Every download is pinned and
checked against its SHA-256.

## Credits

Based on the work of **Alex Spirin ([Sxela](https://github.com/Sxela))**, who pioneered this technique with [DiscoDiffusion-Warp](https://github.com/Sxela/DiscoDiffusion-Warp), [WarpFusion](https://github.com/Sxela/WarpFusion) and [VibeWarp](https://github.com/Sxela/VibeWarp) ([Patreon](https://www.patreon.com/sxela)). DWARP is an independent re-implementation and contains none of their code.

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet) (Lvmin Zhang), [diffusers](https://github.com/huggingface/diffusers), [controlnet_aux](https://github.com/huggingface/controlnet_aux)
- [RAFT](https://github.com/princeton-vl/RAFT) (Teed & Deng) via torchvision
- [RIFE](https://github.com/hzwer/Practical-RIFE) (hzwer); model code vendored from [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) (MIT, see `vendor/LICENSE-rife`)
- [spandrel](https://github.com/chaiNNer-org/spandrel) for loading upscale models
- [Differential Diffusion](https://differential-diffusion.github.io/) (Levin & Fried); the DepthDiff mask comes from [Comfy-DepthDiff](https://github.com/spiritform/Comfy-DepthDiff)

## License

MIT — see [LICENSE](LICENSE).
