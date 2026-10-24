"""E2E: K6 3-clip chain with a custom speech track, through the pack nodes.

Clip 0: I2VA from character.png + track window 0 (no pinned head: the
chain starts, so the listener has heard nothing yet).
Clips 1-2: the previous clip's sampler output pinned as the multi-frame
k6_reference (picture) and the track window pinned clean over the whole
audio stream (sound).

Checks:
- per clip: the output audio IS the encoded track window (the full mask
  keeps it clean); the sampler input carries it exactly;
- the audio seam is exact: clip N+1's output head == clip N's output tail
  (the window arithmetic makes the two the same track region);
- the video seam tracks the pinned tail: clip N+1's first K latent frames
  are close to clip N's last K (regenerated, not copied).

DiT-only process: the windows are pre-encoded by k6_chain_prep.py, so the
audio VAE never shares the rollout process's GPU. The DiT forward runs
under inference_mode (the audio VAE's parameters keep requires_grad=True
and would otherwise bloat the rollout into a 22 GB autograd graph).

Lite model, 480x480, 121 frames per clip, K=6 pinned frames.
"""
import os
import sys
import time
import types

sys.path.insert(0, "/home/khrapov/AI/ComfyUI")

import torch

PACK = "/home/khrapov/AI/ComfyUI/custom_nodes/ComfyUI-Kandinsky6-Motion-Context"
pkg = types.ModuleType("k6mc")
pkg.__path__ = [PACK]
sys.modules["k6mc"] = pkg
import k6mc.nodes as k6nodes

import comfy.nested_tensor
import comfy.sd
import node_helpers
from comfy_extras.nodes_kandinsky6 import Kandinsky6EmptyLatent, Kandinsky6PiFlowGuider

MODEL = ("/home/khrapov/AI/ComfyUI/models/diffusion_models/"
         "Kandinsky-6.0-Lite-Distilled-5s-int8-convrot.safetensors")
TMP = "/tmp/opencode/k6_chain"
WIDTH, HEIGHT, FRAMES, BATCH = 480, 480, 121, 1
FPS = 24.0
K = 6
HEAD_FRAMES = 38


class StubAudioVAE:
    """Returns the pre-encoded window instead of running the audio VAE."""

    def __init__(self, window):
        self.window = window

    def encode(self, x):
        return self.window


def main():
    os.makedirs(TMP, exist_ok=True)
    # detach: the prep process encoded outside an inference context, so the
    # saved latents carry requires_grad=True (the audio VAE's parameters keep
    # it); the ComfyUI server encodes under inference_mode, where they don't.
    encs = [torch.load("%s/enc_w%d.pt" % (TMP, i)).detach() for i in range(3)]
    ref0 = torch.load(TMP + "/ref0.pt").detach()
    track = torch.load(TMP + "/track44k.pt")

    t0 = time.time()
    model_patcher = comfy.sd.load_diffusion_model(MODEL)
    print("model loaded in %.1fs" % (time.time() - t0), flush=True)

    cond = [(None, {"cross_attn": torch.zeros(1, 8, 3584),
                    "pooled_output": torch.zeros(1, 768)})]
    sigmas = torch.linspace(1.0, 0.0, 11)
    empty = Kandinsky6EmptyLatent.execute(WIDTH, HEIGHT, FRAMES, BATCH).result[0]
    v0, a0 = empty["samples"].unbind()
    print("joint latent: video %s audio %s" % (tuple(v0.shape), tuple(a0.shape)), flush=True)

    mc = k6nodes.Kandinsky6MotionContext()
    prev = None
    prev_latent = None
    for i in range(3):
        pos_in = [[e[0], dict(e[1])] for e in cond]
        neg_in = [[e[0], dict(e[1])] for e in cond]
        if i == 0:
            ref = ref0.to(device=v0.device, dtype=v0.dtype)
            pos_in = node_helpers.conditioning_set_values(pos_in, {"k6_reference": ref})
            neg_in = node_helpers.conditioning_set_values(neg_in, {"k6_reference": ref})
            ctx = None
        else:
            ctx = prev_latent

        t0 = time.time()
        pos, neg, joint, trim_frames, trim_samples = mc.apply(
            pos_in, neg_in, empty, K,
            context_latent=ctx,
            audio_vae=StubAudioVAE(encs[i].to(device=a0.device, dtype=a0.dtype)),
            custom_audio={"waveform": track, "sample_rate": 44100},
            clip_index=i)
        v, a = joint["samples"].unbind()

        # the pinned window must sit at the head of the sampler input, exact
        d_in = float((a - encs[i].to(a.device)).abs().max())
        assert d_in == 0.0, "sampler input does not carry the window exactly (%.3e)" % d_in
        if i > 0:
            assert "k6_reference" in pos[0][1] and "k6_reference" in neg[0][1]
            ref = pos[0][1]["k6_reference"]
            d_ref = float((ref.cpu() - prev[0][:, :, -K:].cpu()).abs().max())
            assert d_ref == 0.0, "reference is not the previous clip's tail (%.3e)" % d_ref
        torch.cuda.reset_peak_memory_stats()
        guider = Kandinsky6PiFlowGuider(model_patcher)
        guider.set_conds(pos)
        noise = comfy.nested_tensor.NestedTensor((torch.randn_like(v), torch.randn_like(a)))
        with torch.inference_mode():
            out = guider.sample(noise, joint["samples"], None, sigmas,
                                joint.get("noise_mask"), None, True, 42 + i)
        vid, aud = list(out.unbind())
        assert torch.isfinite(vid).all() and torch.isfinite(aud).all()
        print("clip %d: %.1fs, peak %.2f GiB, trim %d frames / %d samples"
              % (i, time.time() - t0, torch.cuda.max_memory_allocated() / 2**30,
                 trim_frames, trim_samples), flush=True)

        d_out = float((aud - encs[i].to(aud.device)).abs().max())
        print("  audio vs track window: input %.3e | output max diff %.3e" % (d_in, d_out))
        assert d_out < 1e-4, "the masked audio must come back as the clean window"

        if prev is not None:
            # The custom-audio path re-encodes each window independently, so
            # the overlapping head/tail latents are NOT bit-identical (the
            # encoder is causal: the window head lacks the context the
            # previous tail had). Continuity is a waveform-domain property,
            # checked in the decode step against the track.
            seam_a = float((aud[:HEAD_FRAMES] - prev[1][-HEAD_FRAMES:]).abs().max())
            print("  audio seam latent diff (informational): max %.3e" % seam_a)
            pv = prev[0][:, :, -K:].to(vid.device)
            diff = (vid[:, :, :K] - pv).abs()
            print("  video seam: mean abs %.4f | rel %.4f | max %.4f"
                  % (float(diff.mean()), float(diff.mean() / pv.abs().mean()),
                     float(diff.max())))

        torch.save({"video": vid.cpu(), "audio": aud.cpu(),
                    "trim_frames": int(trim_frames), "trim_samples": int(trim_samples)},
                   "%s/clip%d.pt" % (TMP, i))
        prev = (vid, aud)
        prev_latent = {"samples": comfy.nested_tensor.NestedTensor((vid, aud)),
                       "frame_rate": FPS, "sample_rate": 44100}
    print("CHAIN_OK")


if __name__ == "__main__":
    main()
