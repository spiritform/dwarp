# Changelog

What changed in DWARP, newest first. Small styling tweaks are left out; the commit history has everything.

## 2026-10-07

- **Depth: ZipDepth option.** Settings → Depth picks the depth model behind the depth ControlNet, DepthDiff's
  depth and the 3D camera: MiDaS (as before, the default) or [ZipDepth](https://github.com/fabiotosi92/ZipDepth),
  about 20× lighter. It downloads on first use (27 MB).

## 2026-10-02

- **Get-started tour.** On the first visit, a short walk through the screen, area by area. Skippable, and
  replayable from Help → Tour.
- **Motion: Smooth long clips** (Settings, experimental). Clips longer than 16 frames render in one pass with the
  motion model's windows blended at every step (FreeNoise), for one motion all the way through. Needs more VRAM.
- **Motion:** noise tied to each frame (no repeating loop), an 8-frame eased handover between windows, and the
  Camera card works with Motion.
- **animate →** on a single Frame: it becomes Image mode's picture with Motion on.
- GPU drop-down: VRAM, load, temperature, power, CPU and RAM.
- Fixed: Motion stretched wide pictures and clips to a square (512×512); it now keeps the render size.
- Fixed: after loading a picture, the Size / FPS buttons were empty until a page refresh.

## 2026-10-01

- **Motion (AnimateDiff)** as the third way frames connect: Warp · Boil · Motion. SD 1.5; Quality / Fast;
  motion LoRAs for camera moves and effects in Image mode. Style ref and the Style card's LoRA work with it.
- **Shape library:** ready-made black / white masks and loops, plus your own.
- Shape: loop a mask video; drag the stage or a run in as a mask video.
- **Models download on first use**, checked against their SHA-256.
- All runs: a Removed filter.

## 2026-09-29

- Preview renders at the Render size, so the same seed paints the same frame 1.
- Runs strip: arrows browse runs, Delete removes the run on the stage (press twice to confirm).
- One-frame runs offer "download image" (PNG) instead of an mp4.
- Fast models get their settings (steps, CFG) when picked.
- ControlNet gets an "off" button, like DepthDiff's.
- Unlocked seed: the box shows each run's roll, so locking keeps the seed you just saw.

## 2026-09-28

- **Edges: padded.** Each frame is painted with a mirrored margin, so long Warp runs don't darken or smear at the border.
- **QR Code Monster** for Shape: the scene's light and dark bend into your mask.
- **All runs:** every run on disk, with sizes. Hide, show back or delete from disk.
- Use settings also brings back the run's Shape and Edges.
- Page intro animation; progress shown on the transport.

## 2026-09-27

- **Shape:** a mask picture or video (white = the subject) that steers where things are and how they move. Repaint,
  Blur, Opacity, Background, 3-point levels and a filmstrip with in / out.
- Frame Lock in Text / Image mode.
- Live Diff in Image mode.

## 2026-09-26

- **Text → Video** and **Image → Video**, with a 2D / 3D camera that can be steered live.
- **SD 2.1** mode, embeddings, samplers and schedules.
- **Style card:** LoRAs, embeddings, a **Style ref** picture (IP-Adapter) and Looks.
- **Enhance panel:** Refine (a second diffusion pass, 1× to 2×), Upscale and Smooth (RIFE or FILM).
- **Frame Lock**; negative prompt; random prompt.
- Settings: model, embedding and LoRA folders.
- Drag a render from the stage onto the source or Style ref slot; browse runs during a render.

## 2026-09-25

- **Boil:** repaints every frame fresh, for a hand-painted shimmer (next to Warp).
- **Source mix:** stops Warp's feedback from overcooking over long clips.
- FPS on ones / twos / threes.
- Help panel; download mp4; play the source clip; timecode on the scrubber; "use settings".
- Draggable seed; sliding trim selection.
- Removing a run only takes it off the list; the files stay.

## 2026-09-24 — first release

- **DWARP:** restyles a video frame by frame with Stable Diffusion, carrying each painted frame along the
  footage's motion (RAFT optical flow).
- **DepthDiff:** luma or depth decides per pixel where the style repaints (differential diffusion), with
  invert, levels, gamma and amount.
- **Prompt travel:** prompt keyframes on the clip's timeline, with a blend, and Live prompts during a render.
- **SDXL** (Flash Mini by default), with depth + edge through one ControlNet Union ProMax.
- Single Frame / Preview / Render; Source, Depth / Edge / Diff, Output and Split views.
- One-click installer (uv, PyTorch cu128, ffmpeg, RIFE; all SHA-256 pinned).
