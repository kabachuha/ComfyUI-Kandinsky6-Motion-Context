# End-to-end verification: 3-clip chain with a custom speech track

A full rollout through the pack's nodes — not unit arithmetic — to prove the
chain holds together across clip boundaries, with a real track pinned clean on
the audio stream.

## Setup

- Model: Kandinsky-6.0-Lite-Distilled-5s (int8/convrot), 480x480, 121 frames
  per clip (5.04 s picture / 5.06 s audio latent), 24 fps, 10 denoise steps,
  K = 6 pinned latent frames.
- Track: a 125 s mono speech mp3, resampled to 44.1 kHz.
- Clip 0: I2VA from a character image + track window 0.
- Clips 1-2: the previous clip's sampler output pinned as the multi-frame
  `k6_reference` (picture) and the next track window pinned clean over the
  whole audio stream (sound).

Three single-GPU processes keep the ~22-24 GB decode transients off the rollout
card and the two VAEs off each other:

```
1. k6_chain_prep.py   <out> <track.mp3> <audio_vae> <video_vae> <image.png>
2. e2e_k6_chain.py    (rollout: DiT only, windows pre-encoded)
3. k6_chain_decode.py (decode + Trim + assemble + verify, VAE only)
```

The scripts carry the paths of the machine that produced this run; adjust the
`sys.path` insert, `PACK`, and the model constants at the top of each.

## What each process checks

**Rollout (`e2e_k6_chain.py`)** — per clip:
- the sampler input latent carries the pinned track window exactly
  (max diff 0);
- the output audio latent **is** the encoded track window, bit-exact
  (the all-zero audio mask keeps it clean) — max diff 0.000e+00;
- the video seam: clip N+1's first K latent frames vs clip N's last K
  (regenerated conditioned on the pin, so *close*, not bit-exact).

**Decode (`k6_chain_decode.py`)** — per delivered clip:
- delivered length is `frames` and `frames/fps` samples exactly
  (`match_tail`);
- decoded audio vs the delivered track window: mel-spectrogram
  correlation (the perceptual metric; the audio VAE reconstructs the
  spectral envelope faithfully but not the raw phase, so raw waveform
  correlation is not the right measure);
- decoded video seam in pixel space.

Across the whole chain: total delivered samples vs total frames at fps
(A/V drift), and the track gaps between delivered windows.

## Results

| clip | frames | samples | picture | audio | vs track window | mel corr |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 121 | 222338 | 5.0417 s | 5.0417 s | [0, 222338) | 0.9949 |
| 1 | 100 | 183750 | 4.1667 s | 4.1667 s | [223232, 406982) | 0.9945 |
| 2 | 100 | 183750 | 4.1667 s | 4.1667 s | [407552, 591302) | 0.9944 |

- **A/V drift**: 0.5 samples over 13.375 s (0.011 ms) — locked, does not
  accumulate across the joins.
- **Track gaps** at the joins: 894 samples (20.3 ms) then 570 (12.9 ms).
  Bounded per join, never growing: the window advances by the *delivered*
  length, so the next clip's pinned head is the tail the listener just heard.
- **Audio fidelity**: the output audio latent is the encoded track window
  bit-exact; decoded, it tracks the window at mel corr 0.994+.
- **Video seam** (decoded, mean abs / rel L1 on the 21-frame overlap):
  clip 0->1 0.049 / 0.112, clip 1->2 0.040 / 0.088. Identity, scale, pose and
  background carry across each join; the overlap is regenerated, so it is
  close rather than identical by construction.

`DECODE_OK` / `CHAIN_OK` are printed on a clean run. Outputs: `clip{0,1,2}.mp4`,
`chain_full.mp4` (321 frames), and `seam_{0_1,1_2}.png` contact sheets (the
previous clip's tail over the next clip's head).
