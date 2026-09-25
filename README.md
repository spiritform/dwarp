# DWARP

**Diffusion warp** — turn a video into a painted, drawn or dreamed version of itself, frame by frame, locally on your GPU.

DWARP repaints each frame with Stable Diffusion, but instead of starting every frame from scratch it
carries the previous painted frame forward along the video's own motion (optical flow). Brushstrokes
travel with the things they belong to, so the result moves like the footage instead of flickering.

**What sets DWARP apart is DepthDiff**: it doesn't have to repaint everything. Built on spiritform's
[Comfy-DepthDiff](https://github.com/spiritform/Comfy-DepthDiff), a luma or depth map of each frame sets the
denoise *per pixel*, deciding **where** the style goes — keep a face photographic and dissolve the room into
paint, or paint only the shadows. [More below](#depthdiff-choose-where-it-repaints).

One screen: drop a clip, pick a model and a style, write what's in the shot, hit **Single Frame** or **Preview**.

## How it works

For every frame:

1. **Flow** — [RAFT](https://github.com/princeton-vl/RAFT) optical flow between the previous and current source frame, both directions.
2. **Warp** — the previous *stylized* frame is pushed forward along that flow.
3. **Trust** — a forward/backward consistency check masks out occlusions, new content and off-screen areas; those fall back to the raw source frame. Everywhere else a share of the fresh source is mixed back in (**Source mix**, 15% by default), so paint carried over hundreds of frames levels off instead of overcooking.
4. **Repaint** — img2img from that blend, steered by ControlNets (depth + soft edge) computed on the source frame. The first frame gets the full Denoise; later frames a lower one, so they refine what's carried forward instead of re-rolling it. With DepthDiff on, Denoise is set per pixel (below).
5. **Colour** — each frame's colour statistics are pulled toward frame 0 to stop feedback drift.

## DepthDiff: choose where it repaints

DWARP builds in **[DepthDiff](https://github.com/spiritform/Comfy-DepthDiff)** by spiritform — a ComfyUI node that
drives [Differential Diffusion](https://differential-diffusion.github.io/) with a luma or depth map. DWARP ports the
node's mask to its own pipelines (SD 1.5 and SDXL, with or without ControlNet) and runs it on every frame of the clip.

Instead of one Denoise for the whole frame, every pixel gets its own:

- **luma** — the source frame's brightness
- **depth** — its distance, from the same depth map the depth ControlNet uses

The map is shaped with the node's controls: **◐ invert**, **Black / White** levels, **Gamma**, **Contrast**, **Blur**
and **Amount**. White repaints at the full Denoise, black keeps the source, greys fall in between. During sampling
each pixel is held at the source (re-noised to the current step) until its turn comes: white pixels get the whole
schedule, mid-grey ones join halfway, black ones never change.

What it's good for:

- **Keep the subject, dissolve the world** — depth, inverted so the far background repaints and the person stays
- **Paint the shadows** (or the highlights) — luma, so only the dark (or bright) areas take the style
- **Stack it with ControlNet** — ControlNet holds the shapes, DepthDiff picks the areas

The **Diff** tab above the image shows the map live on the source while you drag the sliders — no render needed.

## Features

- SD 1.5 and SDXL checkpoints — reads your ComfyUI models folder and sorts checkpoints by family from their headers
- Style presets, Denoise (how much it repaints), Source mix (how much fresh source each frame gets back) and Hold (how tightly it follows the footage) sliders, steps, CFG, lockable seed
- **Prompt travel**: drop keyframes on the clip's timeline, give each its own prompt; the render morphs from one to the next over a Blend of frames
- **Live**: while a Preview or Render runs, type a prompt and press Enter — the video morphs into it from the frame being rendered, saved as a keyframe
- **[DepthDiff](#depthdiff-choose-where-it-repaints)**: the source's luma or depth decides where the style repaints, per pixel, with a live preview of the map
- **Single Frame** renders the frame under the playhead; **Preview** renders 5 frames (every 2nd) in seconds; **Render** does the trimmed clip. Every run fills the stage at the same size
- Source playback, live split view (source / output), a scrubber with clip timecode, live log, run history with **use settings** and delete
- A built-in guide: the **?** button explains every control
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
- **[Comfy-DepthDiff](https://github.com/spiritform/Comfy-DepthDiff)** (spiritform) — the luma / depth mask behind DepthDiff, ported from the ComfyUI node
- [Differential Diffusion](https://differential-diffusion.github.io/) (Levin & Fried)

## License

MIT — see [LICENSE](LICENSE).
