# Kandinsky 6 Motion Context

Clip chaining for Kandinsky 6.0: pin the tail of the previous clip —
picture **and** sound — so the next clip genuinely continues it, and pin a
supplied speech track clean on the audio stream so a talking person
lip-syncs to the exact words.

Adapted from [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
(GPL-3) for the Kandinsky 6 joint video+audio model.

## How it works

Kandinsky 6 has two pinning channels, and this pack uses both:

- **Picture** — the K6 DiT accepts a `k6_reference` that it appends to the
  tail of the video stream as *clean* tokens (visual token type 1, mask
  channel 1, RoPE positions overlapping the start of the clip). Stock
  ComfyUI builds that tail for one frame (I2VA). This pack needs the
  **multi-frame** tail: a window of `K` latent frames from the previous
  clip's tail, so the model reads real motion instead of guessing it from
  a single still. That support lives in ComfyUI
  (`comfy/ldm/kandinsky6/model.py`), not here.
- **Sound** — the K6 PiFlow guider keeps the part of the audio stream
  covered by the latent noise mask clean. The pinned sound sits at the
  head of this clip's own timeline, end-aligned with the pinned picture,
  and reaches *backward* into sound that has already played.

After sampling, the pinned head occupies the start of the delivered
timeline. The Trim node removes it from the decoded clip, picture and
sound together, and the Save/Load pair carries the full latent across
runs so the next clip can pin from it.

Nothing in ComfyUI is modified by this pack. Before the first render,
`k6_contract.py` checks its assumptions against the live K6 code (the
multi-frame tail, the audio mask, and the grid constants the arithmetic
is written against) and refuses to run with a clear message if an update
broke one.

## Requirements

A ComfyUI with the Kandinsky 6 support:

- the K6 nodes (`Kandinsky6EmptyLatent`, `Kandinsky6PiFlowGuider`, ...)
- the PiFlow guider audio mask (custom-audio clips)
- the **multi-frame I2VA reference tail** (`k6_reference` with K frames)
- the K6 audio VAE (`kandinsky6_audio_vae_full.safetensors`)

A loud `k6_motion_context: ...` error on the first render means one of
these moved; update ComfyUI.

## Nodes

| Node | Role |
| --- | --- |
| `K6 Motion Context` | Pins the previous clip's tail (picture via the multi-frame `k6_reference`, sound via the audio mask) or a window of a custom track. Outputs the guider's conditioning and the sampler's initial joint latent, plus the trim counts. |
| `K6 Motion Context Trim` | Drops the pinned head from a decoded clip, picture and sound together; `match_tail` keeps audio exactly `frames/fps` so per-clip A/V error does not accumulate. |
| `K6 Motion Context Save Latent` | Saves the sampler's output latent (the nested video/audio pair that stock Save Latent cannot serialize) to a fixed per-clip slot under the output folder. |
| `K6 Motion Context Load Latent` | Loads a saved slot as a standard K6 joint latent. Clip 0 loads nothing (first clip of the chain). |
| `K6 Motion Context Chain` | Web UI: Approve / Run-Re-roll / Chain / Reset / Clear latents. Load, Save, and Chain must sit in one canvas group. |

## Wiring

```
LoadAudio ─> (trim to the clip window, for a custom track) ─> VAEEncodeAudio
                                                          |
Kandinsky6EmptyLatent ─> K6 Image to Video+Audio (clip 0, optional)   |
                |                                                     |
                +--> K6 Motion Context <------+                       |
                          ^                   |                       |
LoadLatent (prev clip) ---+                   +--> Kandinsky6PiFlowGuider
                          |                              |
SaveLatent (this clip) <--- SamplerCustomAdvanced <------+
                          |
                          +--> VAEDecode ─> K6 Trim ─> Create Video / save
                          +--> VAEDecodeAudio ┘

Motion Context trim_frames / trim_samples --> Trim
```

- `empty_latent` is this clip's joint latent (from `Kandinsky6EmptyLatent`,
  or the one an upstream K6 node passes through).
- `context_latent` is the previous clip's **sampler output** (from the Load
  node, or wired straight through on a single-run test). It supplies both
  picture and sound, sliced out of the latent — no decode/re-encode at the
  link.
- `context_frames` / `context_audio` are the decode-and-encode fallback for
  when only pixels are on hand (wire the matching VAE).
- `custom_audio` is the full track (e.g. the speech). This clip's window of
  it is pinned clean over the whole audio stream, so the model generates
  only the picture and lip-syncs to it. The window is derived from the
  audio VAE's rate, so any track sample rate works.

## Custom audio windows

A clip's audio latent is `T` frames of 1024 samples at 44.1 kHz
(218 frames ≈ 5.06 s for a 121-frame clip, against 5.04 s of picture).
When chaining with `custom_audio`, clip N's window starts

```
start(N) = N × (clip_samples − head × 1024)
```

where `head` is the pinned audio head (the `audio_context_frames` widget,
or the audio frames covering the pinned picture window). Advancing by the
*delivered* length means the next clip's pinned head is exactly the tail of
the sound the listener already heard — the track plays once, in order, with
no gap or repeat at the join. `clip_index` selects the window and must
match the Load/Save indices.

The audio latent runs at 43.07 Hz against 24 fps picture, so the grids
never meet exactly. The head trim and `match_tail` absorb the remainder:
a few milliseconds per join, never accumulating.

## Chaining

First clip: Load `0` / Save `1`. Next: Load `1` / Save `2`, and so on.
`K6 Motion Context Chain` advances the pair and queues (use its Run/Re-roll
instead of ComfyUI's Run for chained graphs); `Chain` is Approve on a loop,
`segments` caps how many clips it runs (0 = until Stop), `Reset` sets
Load 0 / Save 1, `Clear latents` deletes the numbered slots.

## Tests

`tests/` covers the window and trim arithmetic, the pin/mask construction,
the track slicing, and the save/load round trip. Run from the ComfyUI root
with the ComfyUI environment's Python:

```
python -m pytest custom_nodes/ComfyUI-Kandinsky6-Motion-Context/tests -v --import-mode=importlib
```

`--import-mode=importlib` is needed because the pack folder's hyphen keeps
pytest's default mode from resolving the test modules.

## End-to-end verification

`e2e/` holds the three-process harness (VAE-only prep, DiT-only rollout,
VAE-only decode) that ran a 3-clip chain against a real speech track, and
`e2e/RESULTS.md` records the setup, the per-clip and full-chain checks
(bit-exact pinned audio, A/V drift, track gaps, mel and video-seam
correlation), and the measured numbers.
