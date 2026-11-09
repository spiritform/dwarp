# Changelog

What changed in DWARP, newest first. Small styling tweaks are left out; the commit history has everything.

## 2026-11-08

- **The H3 tab is now **Comfy**, and H3 is one workflow inside it.** The mode is no longer one model: it runs
  whichever ComfyUI graph is picked in the new **Workflow** dropdown, and the controls under it follow what that
  graph takes — no reference slots in its manifest, no References card; no audio, no Sound switch. Adding one
  means exporting a graph in API format into `engine/comfy/` and naming it in `engine/comfy/workflows.json`,
  which says where the prompt, seed, size, length and reference slots go. No Python. `engine/h3.py` became
  `engine/cloud.py`; runs made before this keep working.

- **The H3 prompt templates moved below the Prompt header and work without a picture.** The picker was in
  the card header, where opening it collapsed the card; it is now a **Start from** dropdown above the prompt
  box. All three briefs — character swap, full restyle, new scene — now come in two cuts: with a reference
  loaded they point at `<Picture 1>` as before, and with none they leave you a line to describe the medium,
  line and palette in words. A reference picture holds an identity; a style can just be written down.
  Switching between them now **replaces** the brief rather than folding the old one into the new one's
  description, and keeps the lines you wrote into it — your features, soundscape and music carry across,
  our `...` placeholders do not. **no template** is in the list too: it strips the sections away and leaves
  only what you wrote.
- **Drag the stage into References.** Whatever frame you are looking at becomes `<Picture n>` — a run's
  frame as before, and now a frame of the source clip too, taken at the playhead and saved at the size H3
  actually reads. Handy for pinning your own footage as the look reference.
- **The runs strip is easier to move.** Grab it anywhere but a thumbnail and throw it sideways, or roll the
  wheel over it — no aiming at the scrollbar, which is also now a 12 px target. Dragging a card no longer
  paints its label blue. The thumbnail itself is left alone: it still picks a run, and still drags one onto
  a clip or picture slot.
- **You can hear an H3 render in the viewport.** ▶ on a run with sound plays its mp4 over the stage instead of
  stepping the PNGs — the frames never had the audio. Scrubbing and every other mode are unchanged.
- **H3 ☁ — a new Model tab that makes a clip with sound.** [MiniMax-H3](https://huggingface.co/MiniMaxAI) through
  Pika's H3 Flex adapter. Your loaded clip goes in as `<Video 1>` and its sound as `<Audio 1>`, alongside up to
  nine reference pictures — a face, a costume, or a painted DWARP frame for its whole look — called `<Picture 1>`,
  `<Picture 2>` … in the prompt; either input is enough on its own. Out comes a new 5–15 s clip **with its own
  soundtrack**, the only DWARP mode that makes audio. H3 reads what you give it as *references* rather than
  repainting frame by frame, so it follows your footage without tracking it shot for shot — for that, Warp and
  Motion are still the modes.
- **Comfy Cloud, optional and off by default.** H3 is the one thing in DWARP that doesn't run on your own GPU: the
  model is a 61 GB transformer and a 32-billion-parameter text encoder, far past a 12 GB card, so the job goes to
  Comfy Cloud and the mp4 comes back. Paste an API key in **Settings → Comfy Cloud** to switch it on; without one
  the mode just says so, and nothing else in DWARP changes. The key lives in `config.json` (gitignored) and is
  never shown back. A render spends Comfy credits and takes a few minutes.
- **H3's log says which part of the clip it sent** — "2.1s to 14.0s" — and says so when a trim longer than
  the 15 s H3 reads gets cut short, instead of quietly dropping the tail.
- **Prompt templates for H3.** H3 wants a sectioned brief, not style words — subjects, summary, how hard each
  reference binds, the shot, the soundscape, the music. The **template** picker in the Prompt card lays one out,
  filled in from the clip and references you have loaded, keeping anything you had already written. Three to pick
  from, and the difference between them is how hard the clip binds: **character swap** keeps your footage exactly
  as filmed and replaces who is in it, **full restyle** redraws every surface in the reference's style, and
  **new scene** takes only the energy and pacing.
- **H3 Length: auto**, matching the trimmed clip — the only way to ask for a length the chips don't offer. The
  chips never gave the length they name (H3 only makes 17n + 5 frames, so "5s" is 5.2s), so the line under them
  now says what you will actually get.
- **H3 Sound: keep / silent.** Picture and sound are made in one pass, so silent leaves the track out of the file
  rather than saving time — for when you are cutting your own music over it.

- **Denoise rules the whole render.** Frame Lock is a share of Denoise now, not its own separate amount:
  later frames paint at Denoise x (1 - lock), so turning Denoise down turns the whole clip down. Before
  this, Denoise 0.07 with Frame Lock 0.17 still repainted every frame after the first at 0.83. Saved
  settings and older runs carry over to the same strengths.
- **Shape: Aspect** (stretch / fit / fill). A mask whose shape doesn't match the frame no longer has to be
  squashed into it — fit keeps it whole inside black bars, fill crops it to the frame. A circle stays a circle.
- **Shape library: circle, expanding** replaces "circle, pulsing" — white opens from nothing until it fills
  the frame, then black does the same over it, and it loops.
- **Collage look.** Torn paper cutouts, hard cut edges, halftone print texture — in the Look drop-down next
  to Comic.
- **Motion renders again** with motion modules that have no mid block (TemporalDiff, v1): FreeNoise's memory
  split skips that block instead of crashing on it, and a render's clean-up no longer swallows the real error.

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
