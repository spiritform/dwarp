# Changelog

What changed in DWARP, newest first. Small styling tweaks are left out; the commit history has everything.

## 2026-11-09

- **Render in the shape of your picture.** Aspect gained a **ref** button: the box takes the first
  reference's own proportions, so none of it is cropped away — and until you click Aspect yourself,
  that is what a loaded picture gets. Give a model one picture and the obvious output is that picture
  moving, not a crop of it; the canvas you draw on *is* the box, so this is also what makes drawing
  one to one with what gets rendered. The reference chip keeps the picture's shape too, instead of
  showing a square crop that disagreed with the canvas below it. The model fits a picture into the render
  box with a centre crop — a 4:5 photo in a 16:9 box loses its top and bottom — and until now that was
  something to find out about rather than something to choose. 16:9, 1:1 and 9:16 still crop, and now
  that is a decision. The button appears once DWARP knows the picture's shape: new references record
  it, older ones learn it from the preview.
- **“measure | field” is now “track motion | draw motion”**, and the dropdown under it says **Path**
  rather than Field. Two things you do, named as such; *field* stopped being the right word the moment
  you could draw the paths by hand. And the empty tracking slot no longer dead-ends: its note offers
  *“or draw the motion yourself”*, which puts you on the canvas with your reference underneath and
  Draw already picked. Loading a picture and finding nowhere to draw on it was the way in that was
  missing.
- **Click to pin.** A click on the Draw canvas, rather than a drag, leaves an **anchor**: a path that
  holds its point for every frame, saying *this stays put*. It is the difference between moving an
  object and moving the camera — a couple of lines and nothing else is a sparse instruction, and the
  model spreads that motion over the whole picture, which reads as a camera move in the opposite
  direction. Pin the walls and the floor, draw lines across the thing that should move, and the
  instruction stops being ambiguous. Anchors show as grey rings and are counted apart from the lines;
  a field of nothing but anchors is refused, since it asks for no motion at all.
- **The preview crops the reference the way the model does.** WanMove fits your picture into the render
  box with a **centre crop**, and DWARP was stretching it to fill the preview instead — so with a
  portrait reference in a 16:9 render you were drawing on one picture while the model looked at
  another, and a line aimed at a chair could land on the floor at the bottom edge. The backdrop is now
  cropped exactly as the node crops it, so what you draw on is what gets rendered.
- **A prompt box the model barely reads says so.** A workflow can declare `prompt_hint` in its manifest
  entry; the Prompt card then dims and prints what was measured under it. WanMove is the first: four
  renders on one seed and field showed a completely different scene prompt moving colour a little and
  changing nothing else, and nothing at all on *fast* — where the box is now disabled outright. Dimmed
  rather than hidden, because it still nudges colour, and typing in it brings it back.
- **Draw the motion yourself.** The Field dropdown gained a **Draw** kind: drag across the preview and
  that line becomes a path, from where you started to where you let go, one drag per path. The picture
  underneath is the picture about to move, which is the point of drawing there rather than on a blank
  square — you aim the lines at what is in the frame. How fast you drag is not part of it: a line is
  resampled by its length, so a hand slowing down to aim is never read as the picture slowing down.
  **Amount** becomes how far along your lines it travels, **Paths** goes away because you decided that
  by drawing, and **undo** takes back the last line. The lines are kept normalised, so they survive a
  change of Aspect, Length or seconds, and a reload. A tap, or a line that goes nowhere, is refused
  rather than becoming a path that never moves.
- **Amount and Motion strength said apart.** They are two different dials and read like one: Amount
  belongs to the paths and decides how far they travel; Motion strength is the graph's own, handed to
  the model with the paths, and decides how much the picture obeys them. Both tooltips and the help
  now say so.
- **Videos DWARP makes are runs now, not chips.** An upscale, a smooth, or the track visualiser a
  motion graph writes used to land as a file in the button row, where none of them could be scrubbed,
  compared or thumbnailed — but it is a video you want to
  look at, so it goes in the runs strip like Refine's does. It gets its own frames and a label naming
  what it came from: **upscale · wanmove** beside the **wanmove · render** it was made from, with the
  settings it used kept on it and the source run's own settings still behind *use settings*. The page
  jumps to it when it finishes, the way a render jumps to itself. WanMove's tracks visualiser arrives
  the same way, as **tracks · wanmove**. Videos made before this stayed where they were; moving them
  across is `runs.adopt()` on each.
- **Enhance keeps a log.** A cloud upscale has as much to say as a render — what it sent, the size it
  asked for, the queue id, how long it has waited — and all of it was going into a field that holds
  one line. Post jobs now carry a log to the same panel renders use, with the once-a-second poll line
  left in the status bar where it belongs. Local upscales log what they are about to do too.
- **Upscale on your card, or in the cloud.** Enhance's Upscale section gained a **your card | cloud ☁**
  switch. The cloud side lists video upscalers that run on Comfy Cloud, and picking one swaps the
  controls with it, because each declares its own: **FlashVSR** a target height, **Clarity Crystal** a
  scale factor, **Wan 2.2 Creative** a scale, a *Creative* amount and a prompt. They live in
  `engine/comfy/workflows.json` next to the render graphs, so another one is a graph plus an entry —
  no page edit. The result lands in the run's post folder and shows as the same chip a local upscale
  does, and because none of it touches your card it can run while a render has the GPU. Enhance was
  until now entirely local, so every cloud entry carries a line saying what it spends — partner API
  upscalers bill per call, a GPU workflow bills GPU time — shown before the button, not after.
- **Refine picks its own model, and a Comfy run can be refined at all.** Refine always runs here, on
  your card — but it took its model and family from the panel, which is nothing at all with the panel
  on Comfy, and would have posted the frames back to the cloud as an engine job. It now has a **Model**
  row of its own — an SDXL checkpoint, picked there and only there, never changing the panel behind
  the dialog. A fast model (⚡) brings its own steps and CFG, shown under Scale. And a cloud run's frames
  were always on disk — only the engine's `job.json` was missing, which Refine reported as *“no engine
  frames to refine”*. Their size is read off the first frame, and the prompt and seed off the saved page
  state, so **WanMove → Refine** now works: the graph gives it the motion, diffusion gives it the look.
- **Motion you draw instead of measure.** Measured tracks turned out to be a motion *signature* rather
  than a match for what is in the picture, so the video was only ever a roundabout way of asking for
  "drift left" or "turn slowly". The Motion slot now takes a **Field** as well: a **measure | field**
  switch under it picks which, and whichever side you are not on is kept rather than thrown away, so
  going back to a track you measured never means measuring it again. Pick one of six fields — **Pan**,
  **Push / pull**, **Orbit**, **Swirl**, **Wave**, **Drift** — set **Amount**, **Direction** and
  **Paths**, and the same path JSON comes out of a formula in milliseconds, with no clip and no GPU.
  It is the identical format through the identical slot, so WanMove cannot tell the difference.
- **A measurement follows the panel now, instead of asking to be redone.** All the cost of tracking is
  the chase — RAFT, both ways, over every frame — and Length, output shape and how many paths to keep
  were only ever decided out of that array afterwards. So the chase is saved beside the JSON (about a
  megabyte), and changing any of those three refits the same measurement in a blink, on the CPU, while
  a render has the card. Change **Paths** and the preview is redrawn too, since it would otherwise be
  showing dots that are no longer the ones going up. Only a different clip or trim measures again.
- **Measured motion no longer dictates the render's shape.** It used to, because path coordinates are
  pixels in the output frame and a measurement had nothing but its numbers. Now that it can be refit,
  **Aspect** is yours again with a track loaded — so a landscape clip's movement can drive a tall
  frame. It stretches the motion into that shape, which is exactly what a motion *signature* is for.
- **Speed: quality or fast, for a workflow that carries both.** WanMove's graph has one switch that
  moves model, steps and CFG together — 20 steps at CFG 6 on the full model, or 6 steps at CFG 1 with
  the lightx2v step-distill LoRA, about 3× quicker. DWARP was always running the quality side (and the
  workflow list claimed a distill LoRA it wasn't using, which is fixed); now it is a **Speed** row in
  Model, remembered per workflow. A graph that declares no `speed` switch shows no row.
- **The field animates while you set it.** A measured track gets an mp4 because it has frames of its own
  to be checked against; a drawn one has none, so it is drawn on the panel over your first reference —
  dots and trails, live, following the dials as you drag them. Click it to pause.
- **A drawn field follows the render instead of dictating it.** Measured paths set the output's shape,
  because their coordinates are pixels in the frame they were measured against. A field has no shape of
  its own, so it takes the size and Length you picked and is silently redrawn when either changes —
  no mismatch warning to clear, because redrawing costs nothing. **Drift** reads the seed, so a roll
  gives a different wander.

## 2026-11-08

- **WanMove joins the Workflow list.** Wan 2.1 + WanMove with the lightx2v distill LoRA: one picture
  plus measured motion, and whatever the tracks do is what moves. Pick it and the **Motion** slot appears;
  drop a clip or one of your own renders in, and the paths DWARP measured go up with the job. It also
  brings back the graph's own **track visualiser** — the dots drawn on the finished render — as a
  `tracks` chip under the stage, so the cloud's view of your paths and DWARP's can be compared.
- **The Comfy tab opens a Workflow picker inside Model**, with only the controls that workflow actually
  takes under it. Model's own summary line names the workflow and what is loaded into it.
- **The prompt box follows the workflow.** H3's sectioned brief — and the **Start from** template
  picker that lays it out — is H3's own vocabulary, so it now appears only for a workflow that declares
  `brief`. WanMove prompts like everything else: an ordinary description, the **random** link back, and
  the normal-sized box.
- **Two dials for driving a picture with motion.** **Paths** (8 / 16 / 24 / 48) is how many trajectories
  to measure — a few read as camera drift, a dense field churns the whole surface — and **Motion
  strength** is how hard they pull, from a drift to being dragged along the vectors. Strength is a
  manifest key, so it appears only for a workflow that has such an input, and is remembered per workflow.
- **The motion source sets the render size, and Length gets an `auto` that matches it.** Path
  coordinates are pixels in the output frame, so whatever the motion was measured against decides the
  shape — measure in one box and render in another and the paths land squashed into part of the picture.
  After the motion it is the clip, then the Aspect buttons; Aspect hides itself when something else is
  deciding. If the two ever drift apart, the slot says so and offers to measure again rather than
  letting the render go out wrong, and the engine refuses one point count against a different frame
  count before spending anything.
- **Motion tracks (experimental).** A **Motion** slot on the Workflow card, shown only for a workflow
  whose manifest asks for tracks: drop a video in — your footage,
  or one of your own warp renders — and DWARP measures its movement into the path data that motion-control
  models like Wan-Move follow. It is the same RAFT optical flow the warp already runs, chased frame to
  frame, with the occlusion test deciding when a point has been lost. Out comes the JSON plus a preview
  with the dots and trails drawn on your frames, because flow chaining drifts and watching it is the only
  honest way to know whether it drifted too far. A workflow declares `tracks: {"coords": [node, input]}`
  and the measured JSON is written straight onto that input when the job goes up; **copy** on the slot
  puts it on the clipboard for pasting into ComfyUI by hand in the meantime.

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
