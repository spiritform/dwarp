# Changelog

What changed in DWARP, newest first. Small styling tweaks are left out; the commit history has everything.

## 2026-11-27

- **A Strength slider for embeddings.** Under the embedding picker, 0 to 1.5. There is nothing to
  turn down after an embedding has loaded — it is not a module hanging off the model, it is extra
  rows in CLIP's token table, and once they are in, its name is a word like any other — so the
  slider scales the trained vectors on the way in instead, and changing it reads the file again
  rather than rescaling what is already there. It is **not** a LoRA weight: every real word in the
  model's vocabulary is about the same size, so turning this up makes that one word shout over the
  rest of the prompt rather than adding more of the style. Below **1** it reads honestly as less of
  it, past about **1.3** it starts drowning everything else, so it earns its keep mostly on an
  embedding that is too strong at **1**. (Writing `(name:1.2)` in a prompt still does nothing —
  DWARP has no prompt weighting; this is the knob.)

- **A stack that covers the frame is a picture, not an overlay — so Text → Video paints from it.**
  A layer set to `normal` at `100%` with nothing matting it covers everything beneath it. In Text →
  Video the thing beneath it is *the painting so far*, so such a layer was not sitting on the render,
  it was throwing the render away and putting itself there, every frame — which is why a single
  Frame looked right and a sequence came out as the raw layer. It was never an overlay: it is the
  picture. The stack from the bottom up to and including the **topmost covering layer** is now
  flattened once into frame 1, and the render carries on from it the way Image → Video starts from
  a still. Only what sits above the coverer stays a stack, composited onto every frame — so a pulse
  over an opaque picture still beats on each one instead of being baked into the first. There is no
  switch for this, because it is a tautology: if it covers, there is nothing for it to be over. Take
  a point off its opacity, give it any blend but normal, or matte it, and it is an overlay again.
  The Layers card says which it is.

- **ffmpeg was compositing every `normal` layer backwards, and nobody could see it.** The flatten
  wrote `[under][over]blend=all_mode=normal:all_opacity=X`, which reads as *how much of the layer
  underneath survives*: a `normal` layer at 100% vanished completely and the same layer at 0%
  replaced everything below it. The per-frame numpy compositor was always right, so renders looked
  correct while the plate behind them — the Source tab, the cloud upload, and the picture Image →
  Video starts from — quietly dropped every opaque layer. Measured across all 256×256 pairs of
  values, ffmpeg's `blend` does **not** take its inputs the same way round for every mode: `normal`
  wants the layer first, while `overlay` and `hardlight` want it second and were already correct.
  Only `normal` is swapped now, and `tools/blend_test.py` checks every mode against numpy, with and
  without a matte, so this cannot come back. (Soft light still differs by ~30/255 — ffmpeg's own
  formula, as before.)

- **A solid layer, so a stack can stand on something you made rather than on footage.** **Solid** is
  one flat level over the whole frame — black by default, white at the top of its dial, every grey
  between — and like noise it is never written to a file: the page fills it into a canvas, the local
  render into the numpy stack, and the flatten hands ffmpeg a `color` source, so moving the dial
  costs nothing. It goes on at the *bottom* of the stack rather than the top, because the only
  reason to add one is to give the rest of the stack a floor: noise over black, a pulse over white.
  It is not in the noise picker — a frame of one colour is not an answer to "what kind of grain".

- **Layers can now be added while a render is running, as long as nothing has to be loaded for
  them.** A live edit used to be thrown away whole if it changed how many layers there were, so
  adding noise to a running render looked like noise that did not work. A generated layer — noise,
  a solid — has nothing behind it on disk: it is drawn at the moment it is composited, so one can
  simply begin to exist mid-render, and dropping one puts the picture back exactly. In Text → Video
  the stack can start from nothing at all this way. A layer with a *file* behind it still cannot:
  its frames are pulled at the render's size before the first one is painted, and one that turned
  up since has none — the log now says so instead of going quiet. A matte that arrives live mattes
  the layer under it, because the pairing is read off the stack as it stands rather than as it was.

- **One line for a pulse's beat, and no die where the seed does nothing.** The speed now leads the
  row — bpm, take, swing, drift, four boxes across the four columns the card is already built on —
  instead of the count wrapping onto a line of its own with the metronome beside it. The metronome
  is gone: it clicked out a number you can already hear in the tile. And the die only appears where
  the seed is actually read. Grain and Burst draw their static from it, Grid picks which cells
  light, Slabs where the rectangles fall, and Clouds, Smoke, Fall and Veins their fractal fields —
  but Flash, Bloom, Blocks, Circle, Wipe and Rings are the same picture at every seed, because the
  shape *is* the envelope. Turn **drift** up and the die comes back whatever the pattern, because
  where a loose hit lands is chosen out of that seed too.

## 2026-11-26

- **Layers in Text → Video, so a pulse can beat over a thing that isn't there yet.** The stack
  needed a source to stand on — a clip, or the picture Image → Video starts from — and Text → Video
  has neither, so the whole card was hidden there and **add audio** with it. But Text → Video does
  have something each frame is painted *from*: grey to begin with, then the picture so far, carried
  on by the camera. That is what the stack goes over now. The engine composites it at the moment
  each frame exists rather than flattening anything up front, so a pulse lands on the paint every
  frame instead of being dragged along by the move — and every layer in the stack is an overlay,
  the bottom one included, which means it keeps its blend mode and its matte. Drawn layers are cut
  to the render's own size and length and recut when you change either. There is nothing to show on
  the stage until it renders; the layer's own tile plays the beat.

- **A pulse set to a count stops being redrawn for nothing.** A pattern that repeats exactly is
  drawn one beat long and tiled, but it was being measured against the render's whole length — so
  it counted as stale every time and was redrawn, identically, whenever anything changed around it.
  Fixed, along with a beat that could not be drawn at all: picking flash, bloom, blocks, circle or
  wipe on a count failed with "could not draw that pulse".

## 2026-11-25

- **Sound on a run that is still rendering, and a speaker that covers it.** Playing a live run was
  silent on purpose — the sound would have run on past frames that did not exist. But the frames
  that *do* exist are a stretch of clip like any other, so it plays them with their sound now, and
  the sound stops when it reaches the head rather than looping; from there it goes back to taking
  each new frame as the renderer finishes it. The speaker on the transport also reaches a run's
  soundtrack at last: it is an `Audio()` rather than a `<video>`, so muting the stage silenced the
  clip and the live stack but left a run playing on, out loud.

- **A folder per day, and a delete that doesn't make you wait.** A thousand runs in one flat `renders/`
  folder is not something you can look through, so runs now go in `renders/<YYYYMMDD>/<n>/`, a folder
  per day. A run is still named by its number alone — the day folder is only where it sits — so
  nothing that points at a run has to change, and runs made before this stay put and open exactly as
  they did. `tools/tidy_runs.py` moves the old ones under their day when you want them tidied.
  Deleting is the other half: a run is thousands of PNGs, so **All runs** used to sit there for
  minutes while the disk chewed through a selection. Now the folders are moved aside in one rename
  each — the cards go the instant you confirm — and a background thread empties the bin, picking up
  again at the next start if DWARP was closed half way. **All runs** also opens straight away now,
  with the runs it already knows, instead of showing you nothing until the disk has been read.

- **Noise is generated, not rendered — and it has a size.** A noise layer used to be an mp4 drawn at
  the render's size and length: ffmpeg over the whole clip before any of it could be seen, and a
  frame extraction after it, every time the grain dial moved. For a layer whose entire content is
  `random()`. It has no file under it now. It is generated at the moment it is composited — in the
  canvas for the preview and the tile, in numpy for a local render, as a generated ffmpeg input for
  the flatten — so adding one is instant and the dials move under your hand. Which leaves room for
  the dial that was missing: **Cells**, **Clouds** and **Lines** each take a **size** as well as a
  grain, from a coarse static up to slabs the warp drags about whole, and the size is a share of the
  frame, so what you judge on the preview is what the render draws.

- **A drawn layer can no longer leave a corpse behind.** A pulse, a noise layer and a drawn matte are
  all cached under a name made from their settings, and the check for "is it already drawn?" was
  whether a file of that name existed and was not empty. An mp4 keeps its index at the *end*, so a
  write that was interrupted left a file that passed that check and could never be read — and from
  then on every composite and every Frame was handed the same dead clip, reporting *invalid data* for
  a layer that had worked seconds earlier. They are written to a part file and moved into place now,
  which is atomic: a reader sees either nothing or the whole clip, and an interrupted write leaves
  something no cache will ever accept.

- **A pulse on a count is drawn once, not once per beat.** Flash, Bloom, Blocks, Circle and Wipe do
  the identical thing on every beat, so a 1437-frame clip was 120 copies of the same 12 frames. Only
  one beat is drawn now and the compositors loop it — which they already did for any layer shorter
  than the clip. At 120 that is 12 frames instead of 1437: instant, and kilobytes instead of
  megabytes. It picks the fewest beats that land on a whole frame, so 133 bpm draws six beats rather
  than rounding one into a slightly wrong tempo. Anything that varies still draws in full: a drifted
  count, a beat taken from sound, and the patterns that boil on a clock of their own.

- **Source plays the part you selected.** The transport spanned the whole source file while the trim
  selected a piece of it, so you could scrub into footage that was not in the render — and then
  Frame and Use frame pointed at a frame nothing was going to paint. The in and out points are the
  track now, playback stops at the out point, and pausing lands inside it.

- **Thumb through a render while it runs.** A frame landing used to move the playhead to it, whatever
  you were looking at. Now it only follows if you were already on the newest frame: park on frame 12
  and you stay there while the renderer works ahead. Press **play** on a run still being made and it
  runs up to the newest frame and waits there, taking each new one as it arrives — so play becomes
  watching it being made rather than looping the opening second.

- **A sound switch on the transport**, after the timecode. One switch over everything the stage plays:
  the clip's own audio, a finished run's mp4, and the layer carrying the sound while a stack plays
  live. It applies to whatever is playing at the time, not just the next thing you start.

- **The beat row is one line.** The die leads it and the metronome ends it, instead of the two
  buttons wrapping onto a row of their own: seed, swing, drift, speed, tick.

## 2026-11-24

- **Drift: a count that is not a metronome.** A hit used to land on exactly the beat, forever, which
  is why even a pattern boiling on its own clock pulsed like a machine. Turned up, each hit wanders
  off the line by up to half a beat, lands with a different weight, and now and then does not land
  at all. The count is still the count — it has stopped being quantised. It is a count's dial and
  shows only there, since a sound has no grid to come off.

  Beside it, **a die**: a seed for the layer. It decides which take of a drifted beat you get, and
  it also reaches the patterns that were always random but never repeatable — which cells Grid
  lights, where Slabs fall, how the noise fields boil. The same seed draws the same clip, so one you
  like comes back.

- **Wipe, Fall, and Smoke redrawn.** **Wipe** is a band of light falling down the frame, linear and
  untextured: every row is one value, so what travels is where the white is. **Fall** is the same
  travel with a noise field inside it, drifting downward with the front rather than against it.
  **Smoke** is no longer thick at the floor and thin at the ceiling — it billows out of nothing on
  every hit until it has taken the whole frame, then draws back. Its masses are stretched taller
  than they are wide now, because an even field reads as foam however it is cut.

- **Circle closes instead of vanishing.** It filled the frame, held white, and then cut to black in
  a single frame — a pop you feel more than see. It opens over nine frames, holds whole for one and
  shuts in three. The hold had to go entirely: at 120 the opening plus the old hold came to exactly
  one beat, so the next hit restarted it before the close could ever run.

- **A pulse used as a matte keeps its beat.** Turning one into a matte took its speed, swing and
  count away and put the matte's levels there instead — so the thing driving the shape you were
  cutting with became unreachable, and the only way back was to turn the matte off. The layer gets
  both rows now: what it makes, and what is done with it. Noise keeps its grain the same way.

- **add pulse is now add audio.** The button named one of the things the layer can be. It is a layer
  drawn to sound, and a pulse is one of the ways it can answer.

- **The playhead stays where it is when you change tabs.** Source and Diff scrub the clip; Output and
  Split scrub the run. Those are different timelines — a run is a trimmed, every-nth window on the
  clip — and each tab kept its own index, so switching dropped you wherever the other one happened
  to be pointing. The moment under the playhead is carried across now instead of the number. Going
  back to a run it finds the rendered frame nearest that moment, because on twos and fives most clip
  frames have no frame of their own.

- **A layer's thumbnail shows what its dials are doing.** Blur and opacity are applied to the tile
  as you drag them, so softening a hard-edged pulse is something you watch rather than guess at and
  check on the stage. It is the compositor's own curve, measured against the tile instead of the
  frame, so the number means the same softening on a 128px thumbnail as it does on a 1280px render.
  Blend mode is left off on purpose: a tile has nothing underneath it, so every mode would draw the
  same picture.

- **Two hard-edged pulses, and one of them is the beat itself.** **Blocks** was a stack of squares
  travelling outward and dimming as they went; it is now a single white square whose *size* is the
  envelope — nothing from nothing on the hit, out to the whole frame, and closed again, once a beat
  for as long as the clip runs. **Circle** is new beside it: a disc opening out of the middle until
  it has filled every corner, holding the frame completely white for a few frames, then gone until
  the next hit. Neither dims as it moves. Fading a shape that is already changing size says the same
  thing twice and leaves the moment it covers the frame a grey field, which is the one thing these
  two are not for — what you get is white, black, and the line between them, which is what makes
  them hold up as a matte where Bloom's falloff does not.

  **Prism** is gone. Twelve patterns now.

- **A redrawn pulse is no longer invisible.** The picker's previews are cached for an hour and their
  addresses said nothing about which draw they were, so a pattern that had changed kept showing the
  old loop — through a reload, through a restart, with nothing to suggest why. The draw version
  travels with the catalogue now and rides on every preview URL, so changing a pattern changes its
  address and the browser fetches it. The hour still stands for everything that has not changed.

- **Picking a pulse pattern lands on Pulse.** Choosing a new pattern for a layer that was paced by an
  imported track used to keep the track, leaving the Import tab naming a file you had chosen for the
  pattern you just replaced. A pattern now comes with its own count, and the track is let go.

## 2026-11-22

- **The row under the stage reads as a row of actions.** Every label is sentence case now, and
  **Enhance run** leads it — it is the one that makes something new, the rest copy or download. The
  arrow came off *Use frame*, which was the only label wearing one. While the Enhance panel is open
  its button stays lit in a brighter grey, so you can see which panel you are in without hunting for
  it; the button watches the panel rather than being told, which matters because a dozen things close
  that panel.

- **Wan 2.2 Creative is gone from Enhance.** It was the one upscaler that loaded a real model on a
  Comfy Cloud GPU instead of calling a partner API, and it broke at their end: the WanVideoWrapper
  sampler calls `comfy.model_management.is_directml_enabled`, which ComfyUI core no longer has, so a
  run queued, waited for a GPU, sampled for a minute and then died. Nothing here could pin around it
  — the versions in a graph's `_meta` are editor notes, and `/api/prompt` runs whichever build the
  cloud has installed. **FlashVSR**, **Topaz Astra**, **Flux BFL** and **Magnific** are untouched,
  and a saved choice that no longer exists falls back to the first in the list on its own. The graph
  is kept in git, so it can come back if the pack is repaired.

## 2026-11-17

- **The tour walks the panel as it is now.** It dims one area at a time and says what it is for — but
  the panel has been reorganised under it, and a stop whose card is hidden is dropped without a word.
  On Comfy that left six stops of twelve, and the two that mattered most were among the missing: the
  prompt (its boxes move into the **Workflow** card there, so the stop was pointing at an empty one)
  and **Layers**, which never had a stop in any mode. Both are in now, and the stops that stayed no
  longer describe things that aren't on screen — no Stable Diffusion families on a cloud workflow, no
  *Frame* and *Preview* where only **Render** exists, no "drag the strip's ends" on a still. **copy
  prompt** joined the list of what a run offers, and **Frame Lock** is described as holding each frame
  to the one before, which is what it does with nothing loaded as much as in a warp.

## 2026-11-16

- **Magnific · creative**, a fifth cloud upscaler under Enhance. It is a *picture* upscaler, so the
  graph pulls the run apart into frames, sends each one, and puts the clip back together with its
  sound: 2x to 16x with the detail invented, steered by a prompt, plus Creativity, Resemblance, HDR
  and Fractality on their −10..+10 dials, a subject hint and Magnific's own engines. Auto-downscale
  is pinned on, so a scale that would pass the 25.3 megapixel ceiling comes back smaller instead of
  failing the run. It bills **per image**, and a clip is images — which is why its cost line says so
  and why a single Frame is where it fits best.

  Its nine subject presets are the first cloud control too long to be a row of buttons, so a choice
  with more than four options is drawn as a dropdown now, and a control may carry `labels` beside its
  `options` — the option stays the word the API wants, the label becomes the words you read
  (`science_fiction_n_horror` → *Sci-fi & horror*). Nothing is ever chosen by label: the clamp that
  guards the graph only knows the options.

- **The orange fill on the transport means this track, or it is not drawn.** The playhead has always
  stopped at the last frame that exists — but a **refine** renders a *different* run, of a different
  length, and its progress was being painted on the track of whatever was on the stage. So the fill
  stopped early, the playhead carried on past it, and the bar looked broken when it was only
  describing something else. The fill and the stop come from one place now: it fills when the run on
  the stage is the one being rendered, or when the clip view has a live edge to hold the playhead at,
  and otherwise a refine reports itself in the status chip alone. It is also re-read when the stage
  changes view, not only when a frame lands.

- **copy prompt**, beside *copy settings* under a run. *copy settings* brings a run's whole panel
  back — model, sliders, seed, clip, stack — which is more than you want when the only thing you
  liked about an old run is what it was asked to paint. This takes the words and nothing else: the
  prompt, the keyframes and the negative. Keyframes come because a keyframe *is* a prompt, and
  leaving them would quietly drop most of what a travelling run was made from.

- **Dropping a render on the Input slot shows it there.** With nothing loaded the panel is in its
  text-to-video state, and that state has an Input card of its own: the empty one. Loading a clip
  drew the card before it cleared the flag, so the drop landed, the clip became the source and the
  stack showed it as SOURCE — while the card at the top still read *nothing yet · the prompt paints
  it* and offered to import media. The flag falls first now, so the card you get is the clip's.

## 2026-11-15

- **Frame on a picture renders the frame you are on.** It always rendered frame 1 instead. A
  picture is a single frame as far as ffprobe is concerned, so the frame range was pinned to 0 to
  keep ffmpeg from being asked for a frame the file does not have — but once there is a stack over
  that picture the plate is a proper clip: the picture holds still while the layers over it move,
  and a pulse at frame 30 is a different plate from the one at frame 1. The number is now read as
  where the overlays start rather than as a trim into the picture, which has nothing to trim, and
  the engine leaves a picture base untrimmed so the render cannot fail on it. Without a stack
  nothing changes — every frame of a picture is the same picture.
  This also explains the playhead jumping back to frame 1 after a Frame on a still: it was not the
  playhead moving, it was the run honestly reporting that frame 1 is what it had rendered. The run
  now records the frame it came from, so the playhead stays where you left it and **use frame →**
  and the clip-time readout line up with it.
- **Blur at 100 is three times the blur it was.** The dial was linear and topped out at a sigma of
  4% of the short side — 31px on a 768 render, which softens an edge but leaves a square a square.
  The bottom of the dial is unchanged, because that is where a matte's feather lives and it wants
  the fine travel it has always had; a cubic term now rides on top of it and only really arrives
  near the end, so 100 is about an eighth of the short side (92px at 768) and a pulse turns into
  light rather than a soft-edged shape. Anything already set below half moves by a pixel or two;
  above that it gets noticeably softer. The page's canvas, PIL and ffmpeg all take a standard
  deviation and all three read the same curve, so the preview still predicts the render — and
  ffmpeg's gblur gets three passes above a sigma of 20, where its one-pass approximation starts to
  show seams.
- **The mode is called Warp on a picture too.** The Warp | Boil switch renamed itself to
  *Camera* whenever the source was a still, which put a second Camera on the page right under the
  Camera card and made the mode look like a different one. It is the same mode: carry the paint
  forward. All that changes on a still is what does the carrying — the camera move rather than the
  footage's own motion — and the button's tooltip says so, along with the part that matters, which
  is that **Source mix** is what lets the layer stack land on each frame rather than only on frame 1.
- **Colour: a picture to hold the render's colour to.** Every render has always pulled each painted
  frame's colour back toward the render's own first frame, at a strength of 0.5 that nothing in the
  panel could reach. Two things change. The strength is a dial now, under Style ref. And the
  reference can be a picture you give it, instead of frame 1 — drop one on the slot, or drag a frame
  off the stage that looks the way you want the whole render to look. A picture is a target that
  cannot itself drift, where frame 1 is only ever whatever the model happened to make that time;
  with one set, every frame is matched to it, the first one included. It moves colour and nothing
  else — per-channel mean and spread — so unlike **Source mix** it puts nothing spatial back into
  the frame, which is why it costs nothing while the camera is moving: Source mix at a camera move
  is the stationary plate being laid back over paint that has just travelled. Motion renders take
  the same reference.
- **The runs strip keeps its place.** Removing a run you had scrolled a long way back to find put
  you at the beginning of the list again, and so did every render finishing while you were reading
  the far end of it: the strip is rewritten from nothing on each refresh, and a rewritten element
  is scrolled back to its start. It remembers where it was now and goes back there — and because
  the first paint is only twelve cards wide, a place further right than that is reached again as
  the background fill appends the chunk it was in. Touching the strip yourself calls the restore
  off, since a strip scrolling itself under your hand is worse than one that lost where it was.
- **The mattes are transitions now, and there are nine of them.** A matte that ends where it
  started says nothing — it is a loop, and a matte that loops is a flicker. Every pattern now runs
  from one end to the other across the clip: as a matte that is white to black, the layer there and
  then gone, with **invert** on the card to turn it round. **Circle** opens until it has swallowed
  the corners rather than stopping at the short side. **Split** closes in from both edges instead of
  drifting either side of the middle. **Orbit** still goes round, but swells as it goes and comes
  home having taken the whole frame. **Blobs** is discs arriving one after another and running
  together, every other one mid grey, rather than two discs circling forever. **Thirds** is three
  steps in time — white, then mid grey, then black — so the levels can single out any one of them.
  Three are new: **Polygon**, a six-sided shape opening out of the middle and turning as it grows;
  **Clouds**, fractal noise eating the frame away; and **Ink**, a blot spreading out of the middle
  with its edge broken up by that same noise. The two noise ones are brought up to size smoothly
  rather than with nearest-neighbour, which the flat-colour ones still need.
- **A gallery for noise, and a grain dial.** *add noise* made one kind of static and dropped it on
  the stack. It opens a picker now, the same one the mattes and pulses use, with four kinds that
  are genuinely different instructions rather than one blurred six ways: **Static** (per-pixel,
  redrawn every frame), **Cells** (the same thing with a far bigger brush, hard-edged blocks),
  **Clouds** (fractal noise boiling slowly) and **Lines** (whole rows at a time, the way a tape
  drops out). Each sits on its card as a moving loop, because four grey rectangles all look alike
  until they move. The layer itself gets a **grain** dial: every kind rests at mid grey, so at 0 the
  layer is not there under overlay or soft light and at 100 it is black-to-white. Dragging it
  redraws the clip underneath without rebuilding the row, the way a pulse's swing does.
  Double-clicking a drawn layer's thumbnail — noise, pulse or matte — now opens the gallery it came
  out of rather than a file picker. There is no mp4 you could load into a noise layer that would
  still be noise.
- **A matte can be put on the source itself.** The bottom of the stack could not be matted: a
  matte there was quietly thrown away, on the grounds that there is nothing under the base to
  reveal. There is — black. It is the same black the base's own opacity has always faded into, and
  every compositor fills with it before the source goes down, so there was a floor there all along
  and only the matte was being denied it. A matte directly above the source card now punches the
  source out against that black: a drawn shape, a pulse, a luma pulled off another clip, anything
  that makes a matte for a layer now makes one for the source. Nothing new appears in the panel —
  a black layer you would have to see and manage is not the point, the floor is.
  All three compositors do it: the per-frame one for local renders, the ffmpeg flatten for the
  cloud (where the dropped matte used to leave an unconsumed filter output and take the render down
  with it), and the browser's canvas, so the preview still predicts the render.
- **The noise patterns are not soft any more.** Clouds, Smoke and Veins came out looking blurred
  with the blur dial at 0, which is the layer panel telling you something untrue about its own
  control. It was not blur: three octaves off a three-cell lattice stop at twelve cells across the
  frame, so the smallest thing in the picture was a twelfth of it. Five octaves (four for the
  ridged one) reach forty-eight — the masses are the same size and the same shape, and now they
  have grain on their edges. The cut is tighter to match, since the field's own detail makes the
  edge rather than a gradient across it. About 50 ms a frame at 768, up from 40.
- **Import asks for the track.** Choosing Import with nothing loaded opened a row that said "no
  sound yet" next to a load button — a tab whose job is to take a file, asking you to press
  something else to give it one. It opens the picker itself now. The chip stays for swapping the
  track afterwards, and for the case where you close the picker without choosing, so the row is
  never a dead end.
- **An inverted matte's tile is inverted.** The *invert* button sits right beside the tile and
  changed nothing you could see there, so the card and the stage disagreed about what the matte was
  doing — the stage has always drawn it inverted. The caption over it stays the right way up.
- **A video handed to a pulse that has no soundtrack says so.** An mp4 is a perfectly good thing to
  drive a beat with: its audio is what gets read. A video with no audio in it is the usual way to
  get this wrong, and ffmpeg's own words for it were "Stream map '0:a:0' matches no streams", which
  tells you nothing about what to do next. It now names the file and says a beat needs a track with
  sound in it.
- **The layers card says add media, because that is what it took all along.** The button said
  *add video*, and the picker, the drop handler and the layer card had all accepted a picture since
  the day they were written — the label was the only thing claiming otherwise, so a picture on the
  stack was a feature you had to find by ignoring the button. A picture layer holds still under
  everything moving over it, which is what you want from a texture, a logo or a plate.
- **Seven more pulse patterns: five kinds became twelve.** The five that were there were round and
  soft — a ring, a disc, static — and nothing in the list made a shape you could point at.
  Four hard ones: **Blocks**, a square launched at nothing on every hit and expanding until it is
  off the frame on all four sides, fading as it spreads, so a square inside a square is the one that
  just landed; **Grid**, the frame cut into big squares with half of them lit at random on each hit,
  each fading on its own clock; **Prism**, Rings with corners — a wide hexagon opening out of the
  middle, turned differently every time; and **Slabs**, big flat rectangles thrown across the frame
  and held a moment, never twice in the same place. Three are fractal noise, written here rather
  than pulled in: a coarse lattice per octave, smoothed on the way up to the frame and crossfaded
  between two draws of itself, so the field boils instead of flickering. **Clouds** opens whole
  billows out of nothing on the hit, **Smoke** drifts the same field upward with every layer at its
  own speed, and **Veins** ridges it into thick marbled filaments. The beat moves a threshold set
  from the field's measured range, which is what keeps them properly black at rest. About 40 ms a
  frame at 768. The picker's cards are bigger too — a pattern is easier to recognise than to read
  about.
- **The Pulse and Mattes panels say what they are properly.** The line of text under the title was
  laid out as a row of controls, which is what a header usually holds, so every bold word in the
  sentence was pushed into a column of its own and the sentence read in pieces across the page. The
  title sits on its own line now with the text under it, at a width you can read across, and the
  sentence about blend modes is gone: the stack itself says that.
- **The strip draws what you can see, then the rest.** Every run on disk became a card in one go,
  each with a thumbnail to fetch — on a folder with a few hundred runs that is a few hundred
  requests for pictures sitting off the right-hand edge where nobody is looking. Twelve cards are
  drawn now, which is about a screen of them, and the others are fed in a chunk at a time while the
  page is idle. Their thumbnails still wait until you scroll to them. Nothing is hidden: the whole
  strip is there a moment later, and it arrives without the page stopping to build it.
- **The loading shimmer sweeps the whole cell.** It was a wide gradient slid behind the cell, which
  meant the light started and finished inside it and every loop snapped back across — the jolt you
  could see while the runs arrived. The sheen is its own layer now, entering off one edge and
  leaving by the other, so at both ends of the cycle the cell is simply flat and there is nothing to
  snap. Each cell starts a little after the one to its left, so the light crosses the strip rather
  than every cell flashing at once.
- **download gif, next to download mp4.** A Preview is five frames meant to be looked at and shown to
  someone, and an mp4 of it is an awkward thing to paste into a message — so any run of more than one
  frame now also saves as a looping gif. It is made from the run's own mp4 rather than from the
  frames, so every timing decision the assembler already made is baked into it: the clip's rate, a
  run held on twos or threes, where the trim started, a render you stopped half way. The long side
  is fitted to 640 and a smaller run is left at its own size, the palette is chosen from the whole
  clip rather than frame by frame, and the dither is ordered — error diffusion re-rolls its noise
  every frame, which crawls on screen and roughly doubles the file. The gif is kept beside the mp4,
  so asking twice is a file read, and it is made again if the mp4 changes underneath it. A gif
  carries no sound, and the size is reported when it lands rather than being a surprise. A one-frame
  Frame still offers download image, which is what it is.

## 2026-11-14

- **The stack plays live, so the dials stay live while it does.** Playing Source with layers on used
  to show the flattened file, which has every opacity, blend and blur baked into it — move a pulse's
  opacity while it played and nothing happened until ffmpeg had been round the whole clip again, and
  the repaint when it landed stopped the playback. Now each layer plays in the browser on a shared
  clock and the stack is composited every frame, by the same code that draws the still. Drag an
  opacity, a blur, a blend mode or a matte's levels and the picture changes under your hand, while it
  runs. The clip underneath is heard as it plays, a layer redrawn mid-play joins in on the next
  frame, and pausing hands back to the exact frame it stopped on.
- **The frames play with the clip's sound under them.** The stage shows PNGs, not the mp4 — while a
  render is still going there is no mp4 yet, and the Depth, Edge, Diff and Split views are frames
  whatever happens — so pressing play gave you a silent sequence. It now plays the clip the run was
  made from underneath, at the right place in it: the run knows which source frame each of its
  frames came from, trim and twos included. The sound is the clock and the picture follows it, so
  they cannot drift apart, and the loop takes the sound round with it. It plays while the render is
  running, which is most of the point of it — and where a run was made from a flattened plate, as a
  cloud render is, the plate is pictures only, so the sound comes from the clip that was flattened
  instead; the run wrote down which one that was. A clip with no soundtrack
  steps the frames as before, and so does a Preview, whose frames are not a continuous stretch of
  anything. Playing the Source tab is no longer muted either.
- **"copy settings" on an upscale opens the Enhance panel.** An upscale, a smooth and a refine are
  not made by the left panel, so copying their settings into it only ever got you back to the run
  underneath. Those runs now keep the Enhance settings they were made with — upscaler and scale,
  smoothing and its timing, or the cloud upscaler and all its controls, or Refine's whole card with
  the prompt you typed — and "copy settings" puts them back, pointing the panel at the run they
  were run on rather than at the upscale itself. The left panel still gets the render settings that
  made the picture underneath. Runs made before this say so instead of pretending.
- **The playhead stops where the render has got to.** It already did in the run's own view; the
  Source and Diff views scrub the clip itself, so they let you drag off into frames that do not
  exist yet and sit there looking at black. They stop at the frame being generated now — wherever
  that frame sits in the clip, trim and twos included — and move on as it does. Playing the source
  stops there too instead of running on ahead. When the render ends the clip scrubs end to end again.
- **Refine is its own pass now.** It had been borrowing steps, CFG, sampler, schedule and the LoRA
  from the render panel, which is how a ⚡ pick quietly drove a detail pass at 8 steps and CFG 3 —
  and the panel might be on H3 or SD 1.5, where none of those numbers mean anything to an SDXL pass.
  It carries its own, following the model you give it the way the panel's do, and its own LoRA
  picker for a detail or realism LoRA that has nothing to do with what the render is using.
- **Carry.** How much of the last refined frame, warped into this one, the next frame starts from.
  It was fixed at 0.85, inherited from a render, where carrying the painted frame forward is the
  whole point of the warp. In a detail pass it is why a long clip goes soft: every frame is
  resampled once more than the one before it, a hundred frames deep. It starts at **0** now — each
  frame detailed from the run's own frame, nothing accumulating — and goes up if a clip needs
  steadying more than it needs edges.
- **Refine repaints a tile at a time.** A new **Detail** row — off, 512, 768, 1024 — says how big a
  bite the second pass takes. Off is one pass over the whole frame, as before. A tile smaller than
  the frame means the model paints at the size it paints best at, inventing texture rather than
  enlarging what is there, and it does that at 1×: this is about detail, not size. What holds the
  picture together across a hard repaint is **Hold**, the tile ControlNet — so Denoise goes up to 1
  now, and the panel says plainly when Hold is at 0 (each tile is replaced, not detailed) and when a
  high Denoise will make a clip boil because every frame is painted from noise.
- **A drawn layer follows the length it is drawn for.** A pulse, a drawn matte and static are all
  made here at the frame's size and length, so changing a still's length left them at the old one
  and the flatten looped or cut them — a pulse keeping time against nothing. They are redrawn now
  whenever the frame moves under them: a new length, or another clip loaded. A layer you loaded off
  disk is yours as it is and is never touched.
- **Black is black again.** A pulse lays down what it draws, like every other layer, and the blend
  mode decides how that reads. Driving a pulse's alpha from its own brightness was tried and taken
  out: it made a pattern safe in every mode at the cost of half of them doing nothing at all. The
  rule to keep instead is that a pattern made of light wants a mode that adds light — and the ◐ in
  a pulse layer's header draws it the other way up, white at rest, which wants multiply.
- **A blurred layer reaches the edge of the frame.** The preview's blur now carries the edge pixel
  outward the way PIL and ffmpeg do, so a dark layer on multiply no longer leaves a bright rim all
  the way round. The render was always right; it was the preview that lied.
- **A still's FPS row says what it will draw.** On ones, twos or threes, with the drawing count for
  the length you gave it — the clip card always filled those in, a picture never did.
- **A picture arrives with a length.** Eight seconds, so an imported picture is a source you can
  stack over, pulse on and run a camera across from the moment it lands. Clear the length and it is
  frame 1 of something the camera carries on, as before — that is now something you ask for rather
  than what you get by accident. It was the quiet mode: the stack is flattened into that one frame
  and never again, Source mix does nothing, and a pulse layer sits there keeping a beat that cannot
  happen. So a beat on a picture with no length now says **"no length — flattened into frame 1"**
  instead of offering a speed, a swing and a metronome for something that will not move.
- **A picture with a length can have a camera on it.** A still renders as an image sequence like any
  clip — the stack re-composited over every frame, the transport, the mp4 — but it has no motion in it
  to carry the paint along, which is what Warp means. So on a still the first choice is **Camera**: the
  previous painted frame moved through space (2D, or 3D where the frame's own depth lifts it and near
  things slide faster), with this frame's plate mixed back in at Source mix. Overlays land on every
  frame while it generates rather than being dragged along by the move, and the camera can be steered
  live mid-render. The Camera card sits under Sampler now, where the rest of how a frame is made lives.
- **A beat is an opacity, and at rest it is nothing.** A pulsing layer rests at 0% and rises on the hit,
  so no blend mode can touch the picture between hits — a bloom on overlay used to darken every frame
  and crush the render. Patterns drawn on black matte themselves, keeping all their motion; grain and
  anything else get the beat drawn as a flat clip. **Any layer can take a beat** now (the metronome in
  its header), so a video or a picture can arrive on the music, and **invert** turns it over: full at
  rest, punched out on the hit — multiply's neutral where the light one is screen's.
- **A ramp says which end it is.** The sliders read **Denoise start** and **Denoise end** when a ramp is
  on. The note under them went stale in Video mode whenever anything but its own slider moved Denoise,
  and stayed on screen in Motion where there is no ramp; it repaints with everything else now. Frame
  Lock is sent as the share it is on the slider, so a ramp that starts at 0 keeps the lock you set
  instead of silently falling back. A first frame asked for less than one step is left exactly as it
  arrived — the picture you loaded, so a transition opens on it.
- **copy settings brings a run back whole.** The ramp was the one setting that survived every copy, so
  nothing you loaded looked like what you loaded; a run that held Denoise still now turns yours off.
  The **stack** comes back with it (each layer's file checked, a dead one dropped), and a picture-with-a-
  length comes back as that rather than being pushed through the video prober.
- **use frame keeps the mode you are in.** It used to switch you to Motion — right when it was the only
  way to make a still move, wrong now that a source can be painted frame by frame with a camera on it.
  A picture with no length still turns Motion on.
- **A pulse tile beats at its own speed.** The thumbnail is a loop drawn at a steady 120, so it
  now plays at bpm/120 — the number has a picture of itself beside it, and it keeps up while you
  drag. Nothing is redrawn for it: the same loop, faster.
- **Play no longer sticks on a one-frame run.** It was starting a loop over a single frame: the
  button flipped to pause and nothing could move. Pressing play there now drops back to Source
  and plays what actually moves — the still with the stack over it — which is where scrubbing
  already took you.
- **A picture's length is whole seconds.** Tenths were a precision nobody was using; the floor is
  a second, and below the dash the picture is animated as before.
- **The bar under the stage reads copy settings · use frame → · enhance.** "use settings" is
  **copy settings** again, and the two that take something out of a run and put it back in the panel
  now sit together; enhance follows, since it is the one that leaves the run as it is.
- **A layer is a card.** It has a header — the name, and the marks for matte, hide and remove —
  a shade brighter than the card, running corner to corner, with an outline round the whole thing
  and twice as much air between one card and the next. You drag a layer by that header now, not by
  its picture: a card is picked up by its lid, and the thumbnail is the thing you most want to
  double-click. Pressing hide or remove no longer picks the card up by accident.
- **The thumbnail is 16:9**, across the left half of the card — the shape of the thing being
  rendered, rather than a square crop of it.
- **The card is halved, and each half halved again.** Four columns, one 6px gap between everything,
  so the picture's edges and every control in the card land on the same five lines. The right half
  stretches to the picture's height and splits into rows of equal height. A value box carries its
  name above the number instead of beside it, because at a quarter of the card a label and a number
  on one line cut each other in half.
- **One corner radius, everywhere.** Cards, value boxes, selects, chips and thumbnails all round by
  the same very small amount, so a control sits inside a card instead of arguing with it.
- **A pulse is paced by the clip by default.** The tabs are **Clip · Pulse · Import** — what sets
  the speed of the layer, not where a sound comes from. The clip leads, because pacing the layer to
  what it is going over is what you get without choosing anything; Pulse is a speed you set and
  Import is a track you bring. "BPM" is no longer a tab, since that is the name of the box
  underneath it, and the hits-per-beat dropdown is gone — a pulse is a pulse, and types of pulse
  can come back as types.
- **The pulse ticks out loud.** A metronome beside the bpm box clicks at the speed you have set, so
  you can judge the cue without rendering anything — and it reads the number as you drag it, so the
  clicking speeds up under your hand. Scheduled on the audio clock, so it does not drift while the
  page is busy drawing.
- **"animate →" is "use frame →", and it works on any run.** It used to appear only on a one-frame
  Frame. Scrub a render to the frame you want — while it plays, if you like — and that frame
  becomes Image mode's picture with Motion on.

- **A layer row has a bottom row.** The beat and the length were squeezed into the right-hand
  column under the dials, lining up with nothing. The row is a grid now — picture, properties, and
  a row across both of them, ruled off with a hairline — which is where anything belonging to the
  whole layer goes. Beats per minute and a still's length are value boxes like opacity and blur:
  drag them, double-click to reset. The length box has a floor, and at nothing it reads as a dash,
  which is the picture being animated rather than painted — so one box says both things, and it is
  all that is left of the mode switch.
- **The source is a card now: Input.** With the tabs gone it was the one thing in the panel with no
  header, floating above the prompt — it folds, carries a line saying what is loaded (62 frames,
  a still and its length, animated, or nothing yet) and reads like everything else. The import slot
  is in it even when nothing is loaded, which it never used to be: that was a tab you picked on
  purpose, and now it is simply where you start.
- **No more Video / Image / Text tabs — what you load is the mode.** Three buttons were asking a
  question the panel could already answer. Import media and it renders from it; import nothing and
  the prompt paints it alone (Frame gives you one picture, Render a video). A picture is the only
  genuinely ambiguous source, since it can be frame 1 of something the camera moves through or the
  source itself — so that is the one question left, and it is not a mode either: it is the length a
  picture hasn't got. Leave it empty and the picture is animated; type seconds into it and the
  picture becomes the source, repainted every frame with the stack over it. The source slot says
  **Import media**, and it takes either.
- **A still can be the source, with a length you give it.** A picture has everything a clip has
  except a length, so that is the only thing it asks for. The seconds box sits in the Layers panel on the
  picture's own row, and everything follows from it — the stack is cut to that length, a pulse
  layer has frames to move over, the transport scrubs it, and the mp4 comes out that long. Boil
  repaints the same picture fresh every frame; Warp has no motion to follow, so the paint builds on
  itself. (The Image tab still does what it did: that is where a picture goes to be animated, this
  is where it goes to be painted.)
- **Layers in Image → Video.** The stack was tied to the clip, so loading a picture took the card
  away — but a picture is a source like any other, and putting something over it is the obvious
  thing to want. The bottom of the stack is now whichever source the mode has, and when that is a
  picture the plate is a picture: flattened to the one frame the render starts from, which is what
  the stage, the live Diff and the job all read. A video on the stack contributes its first frame,
  since that is the only frame of it that could have been used. Pulse is not offered there — a beat
  over a single frame is one frame of it.
- **Pulse: a layer that moves on a beat.** Next to add video / matte / noise there is now **add
  pulse** — a layer drawn at the render's size and length whose content follows a beat. Five kinds
  to start: **Burst** (static inside a ring opening out of the middle), **Grain** (static thickening
  on the hit), **Bloom** (a soft disc of light), **Flash** (the flat frame) and **Rings**. The beat
  comes from a BPM you type, from the clip's own soundtrack, or from a track you load — all three
  end up as the same signal, so the patterns cannot tell them apart. After that it is an ordinary
  layer: the blend mode decides how a hit reads (screen to bloom, multiply to stomp), and under Warp
  the paint from one hit is carried into the frames after it. Each kind rests at black or mid grey,
  so a pulse layer between hits costs you nothing.


- **A render keeps the clip's sound.** The frames are PNGs and silent, so an SD render always came
  out mute. Now, once the mp4 is assembled, the clip's own audio goes back on it — the exact span
  the run covers, since frame n is source frame (frame_start + n) × nth, cut to what was actually
  rendered, so a render you stopped half way gets the half you have pictures for. Nothing to take
  (a still, text-to-video, a silent clip) and it stays silent. Playing a run on the stage switches
  to the file when there is sound to hear, and the downloaded mp4 drops into an edit with its audio
  already in place. An upscale or a smooth carries it across too.

## 2026-11-12

- **A still layer no longer fails the render.** Every layer was read over the clip's trimmed range,
  so a picture — which has exactly one frame — had nothing to give when the trim started at frame
  84, and ffmpeg failed the whole job. The trim belongs to the clip: layers over it are read whole
  from their own first frame now and loop by frame number, which is what the preview was already
  showing you.
- **A layer keeps its shape in the live preview.** The flatten always cropped a layer to fill the
  frame; the browser preview stretched it to the frame instead, so a square picture over a wide clip
  looked squashed until the render came back. All three cover now — the flatten, the preview and
  the renderer — filling the frame, keeping the aspect, letting the overflow go off the edges.
- **An enhance no longer draws a false edge on the transport.** The orange fill is the live edge of
  the run you are scrubbing — frames exist to its left, and the playhead stops there. An enhance is
  building a different run with a different frame count (62 in, 123 out), so its progress on that
  track was a line that meant nothing to the frames under it. It reports itself in the status chip,
  which was always the more precise of the two.
- **The Prompt card came back from Comfy.** Switching from a Comfy workflow to SD 1.5 left the
  prompt box, and the card it lives in, still packed away in the Workflow panel: putting the boxes
  back asked for a neighbour that had not been put back yet, so the whole restore gave up halfway.
  They now go home from the bottom up.
- **Render is Sampler.** The card is Denoise, the frame modes and their dials — it has always been
  the sampling half, and the thing called Render is the button.
- **The layers card lost its paragraph.** What it said is in the tooltips and, better, on the Source
  tab, which shows the flattened stack itself.
- **The Denoise ramp turns itself off.** The little "off" link is gone: the end handle now starts
  level with Denoise, which is the flat line, and dragging it away is what makes a ramp. That also
  gives 0 back its real meaning — ramp down to 0 and the paint fades out into the source by the
  last frame, instead of reading as "no ramp".

## 2026-11-10

- **MiniMax H3 is in the workflow list, plain.** The same reference-to-video weights Pika Flex is
  built on, without the Flex adapter: `res_multistep` at 20 steps on the beta schedule instead of
  six distilled ones. It is the control sample — when Flex holds your clip so hard that the prompt
  cannot move it, this says whether that is the adapter or the model. Nine pictures, three videos,
  sound, and a Fast switch that drops to MiniMax's own 4-step turbo LoRA.
- **Context Smasher is parked.** It is out of the workflow picker while the reference behaviour is
  being worked out — still in the manifest, still renderable by id, nothing deleted. One flag back.
- **The colour mask is only offered where it means something.** A mask is an instruction the brief
  has to explain, and that brief belongs to Context Smasher — anywhere else it would go up as a
  video of coloured rectangles. The manifest says which workflow has them now.
- **The H3 brief is a form now, and its wording moved to the server.** H3 reads a sectioned
  document, and inside it the words `fully_preserved`, `attribute_transfer` and `weak_reference`
  are what decide whether the thing you loaded is followed or ignored — and they were sitting in a
  text box looking like prose you had written. That is how a layer stack came back unused: "new
  scene" had quietly written `weak_reference only` over your plate. The Prompt card now asks what
  the shot is, and gives every loaded clip and picture one row saying what it is *for* — keep the
  shot · keep it exactly · loose reference for a clip, the subject · the look · a hint for a
  picture. The document is composed from the answers and shown before you spend anything. A new
  starting point, **follow the plate**, is the one for a layer stack: it tells H3 the clip's
  surfaces are the scene. "Write it myself" hands you the composed text; a brief you had already
  written by hand opens that way and is never composed over.
- **No negative prompt on Comfy Cloud.** None of the cloud graphs has a negative conditioning
  input, so the fold is not offered there — it went nowhere.
- **The Diff tab is gone from Comfy Cloud.** Leaving the guidance tab on DepthDiff and switching to
  a cloud workflow left a Diff tab on the stage for a thing that workflow cannot do.
- **The playhead stops at the frame being rendered.** While a run was being made the track covered
  only the frames that already existed, so it grew under your hand — and the orange fill, which is
  a percentage of the whole job, ran on a different scale from it: the playhead could sit well past
  the orange and look like it had got ahead of the render. The track is now the whole run from the
  first frame, the orange marks how far it has got, and the playhead stops at the last frame that
  exists. This much is made, that much is still coming.
- **H3 First to Last shows both of its slots.** The References row was built for workflows that
  simply number their pictures, so it offered one "add" box at a time — which hid the fact that
  this one has two sockets that mean particular things, and made the closing frame unreachable
  until the opening one was filled. A workflow with named slots now draws every slot, labelled,
  and the one you drop on is the one you fill. Taking a picture out leaves that slot empty instead
  of promoting the next picture into it, which would have turned a closing frame into an opening
  one without saying so. An empty slot is drawn as the card it is about to become — the same
  picture-shaped well, the slot's name on the same band — so the row reads as two image slots,
  one of which happens to be empty.
- **Flattening the layers no longer freezes the whole server.** The plate is ffmpeg over the
  entire clip, and it was running inside the request's own thread — so while it composited, DWARP
  answered nothing else: the frame thumbnails the live preview is built from never arrived (the
  layers simply never appeared over the source), the status sat on "compositing the layers", and
  reloading the page spun. It runs beside the server now instead of in front of it. The same goes
  for the noise clip and the colour masks.
- **The plate waits a moment longer before it rebuilds.** Nudging a dial used to queue a full
  flatten 400 ms later, so a drag could start several of them; it is 900 ms now. The preview in
  the browser is the live one and is unaffected.
- **Dragging a value box keeps up with the pointer.** Every pointer event was compositing the
  stack, running the matte's levels over every pixel in the browser, encoding a frame and handing
  it back to the stage — more work than the screen has frames for, so the picture ran behind your
  hand. A repaint is asked for rather than done (one per frame at most), the same arrangement is
  never drawn twice, and a matte's alpha is kept until something it depends on actually changes.
- **The value boxes are grey, not orange.** The fill is a step up from the track rather than a
  wash of accent colour, and the text sits lighter on it. Orange is for the box you are dragging.
- **Layer rows: a square thumbnail down the left, everything else beside it.** The dropdown and
  both dials now stack to the right of the picture instead of below it, which takes about 25 px off
  every row — a five-layer stack fits on screen without scrolling. The dials keep their shared
  grid, so they still line up down the whole stack.
- **The source on its own is a stack of one.** Fading or blurring the clip with no layers over it
  always rendered, and the stage did catch up once the server had composited a plate — but the
  instant in-browser preview was gated on there being two layers to combine, so dragging the slider
  sat still for the best part of a second before anything moved. It responds as you drag now, the
  same as every other arrangement.
- **A layer row is a picture and three rows.** The name, then how it is laid down, then what it is
  set to — with a 90 px thumbnail as tall as all three. The file name finally has somewhere to
  live: it was only ever a tooltip, which is no help in telling two clips apart at a glance. A matte
  puts LUMA MATTE where the blend mode would be, with invert and rgb beside it, so it is the same
  three rows and the same height as everything else.
- **Opacity and blur are draggable value boxes, side by side.** Drag one sideways to change it the
  way the seed field already works (1 per 2 px, Shift for quarters), double-click for its default.
  The proportion is the fill behind the text, so nothing is lost by not having a track — and two
  of them fit on the row a single slider used to take.
- **Black, mid and white are one levels bar.** Three handles on a single track, painted with the
  curve they make: solid black up to the black point, solid white from the white point, the ramp
  in between. The grey handle is the gamma — it sits where the half-way value lands, which is
  exactly what the engine builds its lut from, so dragging it is dragging the curve. A matte row is
  now feather + levels, the same height as opacity + blur.
- **The eye and the delete stay visible.** They were appearing on hover, which meant a hidden layer
  looked like a dimmed one with no explanation unless you went looking.
- **The thumbnail is the handle.** The drag grip is gone and the picture does its job: drag the
  thumbnail to reorder, double-click it to put a different video or picture in that slot. Swapping
  this way keeps how the layer is being used — blend mode, opacity, blur, matte and its levels —
  and changes only what is in it, which is not the same as deleting the layer and adding another.
  Double-clicking SOURCE loads a different clip.
- **matte is a mark, not a word.** Between it and the grip, the blend dropdown gets about 45 px
  back, which is the difference between reading "difference" and reading "differe…".
- **An eye on every layer.** Next to the remove ✕, and hidden means hidden everywhere: out of the
  live preview, out of the server's plate and out of the render. The layer keeps its place and all
  its settings, the row dims, and the eye stays showing while it is off so something says why the
  picture changed. Hide every layer and the clip renders on its own with no flatten at all.
- **No more window-wide dropzone.** Dropping a video anywhere on the page replaced the source
  clip, and a run dragged off the strip did the same anywhere in the panel. That was fair when the
  source was the only place a video could go — it now competes with references, layers, video
  contexts and Motion, so missing a slot by twenty pixels quietly threw away the clip you had
  loaded. Every drop has to land on the slot it is meant for now, the source among them, and any
  drag — a file off the desktop or a run off the strip — lights up the slots that can take it. A
  drop that lands nowhere says where it could have gone and changes nothing.
- **H3 · First → Last.** A second H3 workflow on Comfy Cloud, built on MiniMax-H3's FL2VA model
  with Pika's matching Flex adapter. Give it two pictures — Reference 1 is the frame the shot opens
  on, Reference 2 the frame it closes on — and the prompt describes the journey between them. Both
  ends are held, so a shot lands exactly where you want it instead of wherever the model drifts to.
  Five to fifteen seconds, with sound, same six steps and native 1344×768 as Pika Flex. It brings
  a brief of its own, **first → last**, which states on the clock where each picture sits and leaves
  you the timed description to fill in. The References card names the two slots FIRST FRAME and LAST
  FRAME rather than numbering them, which a workflow can now ask for in the manifest.
- **A render you stopped part-way is a render.** Cancelling used to skip assembling the mp4, so
  those runs could not be played, downloaded, enhanced or dragged back in as a source — Enhance
  simply said “this run has no mp4 to upscale” and left you with no way to make one. The frames
  cost what they cost, so now they are assembled like any other run's, and the log says it stopped
  early. Enhance also builds the mp4 itself before a cloud upscale, which fixes the runs already
  sitting there without one.
- **A dropped connection no longer throws away a paid cloud job.** Every call to Comfy Cloud is
  now sent again when the failure is the network rather than an answer — a reset connection, a
  timeout, a 502 — with the one exception of `/api/prompt`, because a second of those is a second
  job and a second bill. A 4xx still fails at once, as it should. Uploads and downloads also stop
  sharing the 45-second budget meant for a status poll: a whole clip goes up in one POST and now
  gets fifteen minutes, and the finished file gets the same on the way back. Before this, a single
  blip while polling a running job lost the job — the cloud finished it and billed for it, and
  DWARP had already given up.
- **Flux Video Upscale joins them, with two minds.** Black Forest Labs' super-resolution as a
  fourth cloud upscaler: **creative** restores and invents fine detail, **precise** only
  sharpens what is already there, and a prompt steers what gets invented. Target 1080p, 2K or 4K.
  It takes 1–20 seconds of video and renders at 24 fps whatever went in, so a clip rendered on
  twos or threes comes back retimed — that is in its limits line. Upscale graphs can now declare
  a `seed` the way render graphs do, and it is rolled fresh every run, so asking again is another
  roll rather than the cloud's cache.
- **Topaz Astra 2 replaces Clarity Crystal in the cloud upscalers.** Clarity restored what was
  already there and nothing more, which is the wrong instinct for a render that was painted rather
  than filmed. Astra 2 enlarges to 1080p or 4K and rebuilds as it goes, with a **Creative** slider
  saying how far it may invent, a **Realism** pull (leave it at 0 on a stylised clip) and an
  optional scene **Prompt** — setting the prompt caps the clip at 450 frames. Still a partner API
  node, so it bills per call rather than against GPU time, and it keeps the frame count, the fps
  and the sound. FlashVSR and Wan 2.2 Creative are unchanged.

## 2026-11-09

- **Denoise can climb across the clip.** A **ramp to** slider under Denoise: the model keeps more of
  the source at the start and invents more by the end, so a shot drifts out of itself as it runs
  rather than hallucinating evenly throughout. Frame Lock is a share of Denoise, so it rides the
  same ratio instead of staying pinned to where it was at frame 0 — at 0.40 → 0.85 with lock 0.6,
  the carried-paint strength climbs 0.24 → 0.51 alongside it. Off by default, and `off` puts it back.
- **Denoise, Frame Lock and Source mix are live too.** The loop already read Source mix per frame
  and Denoise for the first one; Frame Lock's derived value is simply re-derived the moment any of
  them moves. So all three ride the same channel as the prompt, the camera and the stack, and land
  on the frames still to come — recorded in `job.json` beside every other live edit.
- **Live layers: move a slider while the render runs and the frames still to come follow.** The
  painting modes no longer flatten the stack up front — every layer is kept as frames and the one
  being painted is composited at the moment it is painted, which costs **11 ms** against roughly two
  seconds of diffusion. So the stack arrives on the same channel Live camera and Live prompt come
  down: drag opacity at frame 40 and frames 41 onward are composited differently, recorded in
  `job.json` as a live edit like any other. A layer cannot be added or removed mid-render, since the
  frames for it were never extracted; the settings are what moves.

  The blend maths is the separable set from the compositing spec, which is what a browser canvas
  computes too — so what the panel previews is what the engine paints. The cloud path still
  pre-flattens, because there the plate is the thing that gets uploaded.
- **The source is a layer too.** It has its own opacity and blur. Nothing sits under it, so turning
  its opacity down cannot reveal anything — it scales the channels instead, which is a fade to
  black, and that is the useful thing about it. The engine does the same with `colorchannelmixer`
  rather than an alpha blend, so the preview and the render agree.
- **One grid down the whole stack.** Every dial sits on the same three columns — name, bar, number —
  so `opacity` and `feather` and `white` cannot push each other's sliders out of line, however long
  the word is. The select, the chips and the delete share one height and one row; the delete stays
  quiet until the row is under the pointer but keeps its place, since a control that appears and
  shoves the rest of the row along is worse than one that is simply quiet.
- **Blur per layer, and the dials moved under the thumbnail.** Opacity and blur get the card's full
  width in their own rows rather than a third of it squeezed beside a picture — a slider you cannot
  aim is worse than one more row. A matte gets **feather** instead, applied after its levels so a
  hard-edged pattern can be softened without the levels clipping the softness straight back off.
  Blur is stored as a share of the frame, not a number of pixels, so what you judge in a 480px
  preview is what the flatten makes at 1440. The base row says **SOURCE** and has dropped its
  caption; a thumbnail at the bottom of a stack does not need telling it is at the bottom.
- **The stack previews live on the stage**, the way DepthDiff's mask does: each layer's frame is
  fetched once and kept, and the composite is rebuilt in the browser on every change — **4–6 ms** to
  redraw, so an opacity slider moves the picture under your hand. Play, and it plays the server's
  plate instead, because twenty-four composites a second is what the flatten is for; pausing and
  scrubbing hand straight back to the local one, which is never a few hundred milliseconds behind
  whatever you last moved.

  The blend modes are now only the eleven a browser canvas and ffmpeg both have and agree on. A mode
  the preview cannot show is a mode that would make the preview lie, so ffmpeg's `vividlight`,
  `pinlight`, `phoenix` and the rest are out and a saved layer wearing one falls back to `normal`.
- **Layers: a stack of sources flattened into one clip before anything is rendered.** Its own card
  below Style, in every mode that paints from a clip. Each layer has a blend mode (ffmpeg's eighteen
  Photoshop modes) and an opacity, and a black and white layer can be set as a **luma matte** for the
  layer below it — it is not drawn itself, its brightness becomes that layer's alpha, and Black / Mid
  / White shape an arbitrary grey clip into a usable matte. The matte sits in the stack where you can
  see it rather than hiding inside a slot, which is the After Effects rule and the right one. Drag to
  reorder; the clip is the base underneath it all.

  In the cloud the stack flattens at the render's size and length and becomes the reference. In the
  painting modes it flattens to the shape of its own base — the clip — and stands in for it, so trim,
  range, fps and every size decision downstream behave exactly as they did and the only difference is
  what the frames have in them.

  This replaces the colour masks, and it replaces them because they were measured and they do not
  work: handing H3 a mask video and explaining the colour code in the brief makes the model **paint
  the mask**. A test render with a brief that said in as many words that the mask's own colours must
  never appear came back 50% pure red and 50% pure green by its middle frame. A mask is a thing to
  composite with, not a thing to describe — so the mask library now draws in black and white, is
  called **Mattes**, and the patterns are cut to the size and length of whatever they are matting.
- **The reference pictures are cards too, the same size as the video contexts.** `<Picture 1>` was
  faint grey type beside a 38px thumbnail, which is not where your eye goes — and that tag is what
  the prompt calls it. Same card, same `PICTURE 1` band across the foot of the picture, same delete
  box, two across. The filename moved to the tooltip, where it is still there when you want it.
- **The log says how much video actually went up.** A video context has no trim of its own, so the
  window it asks for is H3's whole 15 s ceiling — and printing *that* made a 5-second mask look like
  it was being stretched to fifteen next to a 5-second generation. Nothing was: trimming past the end
  of a file just stops at the end. It now probes what came out and prints that.
- **Picking a mask was taking twelve seconds, every single time.** Two reasons, both silly. The file
  was written under a random id, so a pattern you had already chosen was drawn again from nothing on
  every pick — it is named after the request now (pattern, size, length, amount), and asking for the
  same mask twice is a `stat()`. And every frame was painted at the full render size: a megapixel of
  flat colour per frame, 380 MB of raw video pushed through a pipe for one 5-second clip. These are
  flat regions with hard edges, so they are painted at a whole-number fraction of the size and scaled
  back up with nearest-neighbour, which reproduces hard edges exactly — anything smoother would
  invent colours along every edge that are neither red nor green, and the colour *is* the instruction.
  **12.1 s → 0.50 s on a first draw, 1 ms on a repeat.** Needs a restart to take effect.
- **One delete control.** The ✕ on a run card, a library card and a video slot were three
  slightly different black circles at three sizes. They are one subtle rounded box with a small bold
  ✕ now, and the ones that sit on a panel rather than on a picture wear the same glyph without the box.
- **The colour strip under the mask tile is gone**, and its chip says `colour mask` rather than
  `Circle · library`. Every slot wears its colour on its own frame now, so the strip was a second
  reading of the same thing in a place that invited you to read it as part of the pattern.
- **Each slot is framed in its colour, with its tag across the foot of the picture.** `<Video 3>` in
  small green type under a thumbnail is something you have to go and read; a green frame round the
  frame with VIDEO 3 on it is not. The label sits on a dark band in white, not on the colour itself —
  white on `#19d23c` is hard to read, and the mask colours are not up for changing: those pixels are
  the instruction the model reads, and the brief says "green is `<Video 3>`" in words, so the word has
  to match the pixel. The clip card wears one too, which also gave it its meta line back — the tag
  used to wrap `440×768 · 24 fps · 124 frames` onto two rows. The mask's own frame is a bright
  neutral grey: it hands the colours out rather than being one of the things aimed.
- **The colour key follows you down to the prompt box.** Which colour means which video is the whole
  technique, and it was only legible on the cards — which sit above the prompt and scroll away exactly
  when you are writing the line that uses them. The same bands now sit above the box you type in.
  Click one and it drops the tag in at the cursor in its exact form, `<Video 2>` with the brackets,
  because that is the string the model matches on and VIDEO 2 on a band is not it.
- **A workflow with one brief arrives with it chosen.** Picking Context Smasher already writes the
  brief into the box, so "no template" sitting over the top of it was never true — and after a reload
  it said so again, because the pick was read off the dropdown rather than remembered. It is saved
  with everything else now, and an explicit choice (including no template) sticks.
- **Every picture of a mask is 16:9 now.** They used to be drawn in the shape of the render, on the
  grounds that a mask is a map of the frame — but a portrait render turned the library into a row of
  thin strips with no pattern readable in any of them. The mask that actually renders is still drawn
  at the render's own size; this is only the picture of it, and one shape means one cached preview per
  pattern instead of one per render size.
- **In Comfy the prompt sits below Model.** The clip, the reference pictures and the video contexts
  are one argument — the slots are named against the clip and the prompt calls them all by tag — and
  a prompt box between the clip card and those slots meant the source video was off the top of the
  panel while you filled them in. The other modes keep the order they had.
- **Opening the mask library a moment too early no longer tells you to restart the server.** The list
  arrives with the rest of boot; before it lands it is empty, which is not the same thing as a server
  too old to draw them. It says it is looking, and fills in when the list turns up.
- **Context Smash is now Context Smasher**, and its video slots look like what they are. Each one is
  a card with a frame of the video in it — a loaded clip showed a grey film icon before, so the only
  way to know which video was in which slot was to read the filename. The card is outlined in the
  colour the mask sends its way, and so is the clip card, since the clip is `<Video 1>`: one glance
  says red is the clip and green is the other one, instead of a legend to go and find. The empty slot
  wears the colour it is about to take, so you can see where a video will land before you drop it in.
- **The slots sit side by side, and you can drag one past another.** The order they are in is what
  hands the colours out — first colour to the first video that is not the mask — so reordering is how
  you re-aim the mask, and until now the only way to do it was to remove both and add them back the
  other way round. The tags renumber as you drop, and the status line says what the colours mean
  afterwards, because a brief you have already written still names the old numbers.
- **A mask tile that lost its source no longer sits black for ever.** A video element does not retry
  on its own; if the request was dropped while the server was busy the tile stayed empty until the
  page was reloaded, which is the thing these tiles were brought in to fix. It asks once more.
- **The console window says something.** uvicorn's own banner is logged below the level the server
  runs at, so start-up was a black window with a cursor in it — indistinguishable from the hang it
  used to be while torch loaded. It now prints when it starts and prints the address when it is up.
- **Nothing loads a deep-learning framework to read a filename.** Listing your LoRAs meant importing
  `engine/warp.py`, which imports torch — so asking "which LoRAs do I have?" paid for a cold CUDA DLL
  load first: **50 s** on the first call, 533 ms on the second. The scanning helpers moved to a new
  `modelscan.py` that imports nothing heavier than the standard library, and the engine re-exports
  them so the render path is untouched. Embeddings were the stubborn case — a `.pt` file is a pickle,
  and the obvious reader is `torch.load` — so they now go through the same shape-only unpickler that
  already read `.ckpt` files, with every tensor stubbed out and no code from the file run. Checked
  against `torch.load` on all 12 embeddings on disk: same answer every time.
- **A folder is walked once per session**, not once per question, keyed on the folder's own mtime so
  adding a file still shows up. Second call to any of these dropped from seconds to 5–350 ms.
- **The page asks only for what it is about to show.** The model folders are no longer read at
  start-up; each list is fetched by the control that paints it — the Style card asks for LoRAs and
  embeddings, the Model row for checkpoints, Motion for its modules. In Comfy Cloud mode those cards
  are hidden, so none of it is ever read. Start-up also stopped asking the server whether each
  preset's checkpoint exists (seven round-trips feeding a flag nothing read) and stopped making
  `/api/config` walk every checkpoint just to count them — Settings asks for the total when it opens.
- **A cloud workflow stops pretending it has a prompt timeline.** The keyframe lane sat under the
  prompt box in Comfy mode, where there is one prompt and nothing to travel along. The code already
  said so — but the lane had no `hidden` of its own, so it was on screen from the first paint, and
  nothing called `paintKeys` on the way into a cloud workflow. It starts hidden now and is shown by
  the one function that knows whether it applies. H3's brief box also opens shorter: 300px of empty
  monospace pushed everything under it off the panel.
- **The workflow's help is a grey disc at the end of its row**, instead of an orange link wedged in
  among the words.
- **DWARP opens in seconds instead of minutes.** `run.bat` used to sit on a black terminal long
  enough to look broken, and it was: `post.py` imports torch, torch loads its CUDA DLLs, and off a
  cold disk that is minutes — all of it before the port could be bound, so nothing would answer.
  Nothing the page asks for at start-up touches a tensor (which upscalers are on disk, whether
  RIFE's weights are there, what job is running), so torch now waits for the first thing that
  actually needs one, and warms on a background thread meanwhile. **Ready to answer in 26s instead
  of about 222s**, and `/api/post/options` went from 12.7s to nothing.
- **The page stops asking one question at a time.** Seven independent lookups ran in a row while the
  panel sat empty — cold, `/api/loras` and `/api/post/options` alone were 8s and 13s. They go out
  together now and cost the slowest one.
- **A library for the colour masks, so you can see one before you pick it.** The row used to wear two
  colour swatches, which are the *legend* and not the mask — Circle, Wipe, Split and Orbit all drew
  the same two stripes, so the only way to find out what a pattern did was to render with it. The
  pattern chip now opens a grid of every mask actually moving, drawn in the frame shape you are
  rendering, and the slot's own tile loops the mask you chose. The swatches stay behind it, so a mask
  whose file has gone still says which videos it speaks for.
- **Split is half and half, and the blobs stop eating each other.** Split opened near 20/80 and closed
  near 80/20, sweeping straight past the middle, so a frame you asked to halve started lopsided — it
  now holds about a half each way throughout. Blobs sent both discs along one path at different
  speeds, so they met and blue (drawn second) swallowed green: *"<Video 3> only"* for those frames, at
  exactly the moments that read as keyframes. They now ride opposite ends of one ellipse and can
  never touch.
- **H3 · Context Smash, with the mask already in the slot.** H3 reads up to three whole videos as
  context at once and the prompt names them — `<Video 1>`, `<Video 2>`, `<Video 3>` — so three
  unrelated clips come back blended. DWARP wired one of those slots; now a workflow of its own opens
  all three. It is the same graph as Pika Flex asked a different question, and **Pika Flex is exactly
  what it always was**, one clip as `<Video 1>`.
- **DWARP draws the colour masks.** The way you aim a smash is to make one of the videos flat colour:
  red where `<Video 1>` belongs, green for `<Video 2>`. `engine/masks.py` renders them — **Circle**,
  **Wipe**, **Split**, **Orbit** for two videos, **Thirds** and **Blobs** for three — flat, saturated
  and hard-edged on purpose, since a gradient reads as content rather than as an instruction. No GPU,
  a second or two. Picking Context Smash puts one in a slot straight away, and the dropdown on that
  row swaps the pattern. Change Aspect or Length and the mask is redrawn before the render: a map of
  the frame has to be the frame's own shape.
- **The brief writes the plumbing.** The mapping only exists in words — to the model a mask is pixels
  like any other video — so Context Smash fills the `video_contexts:` block in for you, naming every
  slot and spelling out which colour means which. It works the mapping out from the slots as they
  stand, skipping the mask's own tag (it cannot stand for itself), and says *"green has no video yet"*
  rather than pointing at a slot that is not there. The tags on the panel wear their colours, so you
  can see the mapping without reading it. The only line left blank is what the shot is.
- **Less to read on the panel.** The paragraph under the Workflow dropdown is gone, replaced by a **?**
  beside the label that opens the help, where there is now a Context Smash section. The add rows are
  one line once a slot is filled instead of a card-sized landing strip, a video and a colour mask sit
  side by side as the two ways in, and the card hover lift is gone — it was lifting the card to the
  point where the rows sitting on it stopped reading as controls.
- **Motion is translated to the render, never refused by it.** A path node takes exactly one point per
  frame, so motion drawn at 3s and rendered at 5s used to come back as an error telling you to do it
  again. That is a refusal over arithmetic. `fit_tracks()` re-spaces the paths to the render's frame
  count instead: the shape you drew is untouched, it simply takes the new number of frames to describe
  it, the ends stay exactly where you put them, and anchors hold their point at any length. The log
  says it happened. The panel still redraws the field first when it can — a redraw re-eases the
  motion where the engine can only re-space it — and a render now waits for that rather than leaving
  with the old file, which is how the mismatch got out in the first place.
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
