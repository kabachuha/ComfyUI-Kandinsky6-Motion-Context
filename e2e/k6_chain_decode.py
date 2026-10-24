"""Decode the 3-clip chain, run the pack's Trim node, assemble the delivered
video, and check A/V drift plus audio fidelity against the track.

Runs in its own process (VAE only). Writes per-clip and full-chain mp4s and
seam contact sheets under /tmp/opencode/k6_chain.

Delivered windows in the track (samples at 44100):
  clip 0: [0, round(121/24*44100))
  clip 1: [223232, 223232 + round(100/24*44100))
  clip 2: [407552, 407552 + round(100/24*44100))
(the 223232 = 218*1024 boundary is the head of clip 1's window plus its
38*1024 pinned head; 407552 = 2*223232 - 38*1024 + 38*1024 the same way).
"""
import os
import subprocess
import sys
import types

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/khrapov/AI/ComfyUI")

PACK = "/home/khrapov/AI/ComfyUI/custom_nodes/ComfyUI-Kandinsky6-Motion-Context"
pkg = types.ModuleType("k6mc")
pkg.__path__ = [PACK]
sys.modules["k6mc"] = pkg
import k6mc.nodes as k6nodes

import comfy.sd
import comfy.utils
from comfy_extras.nodes_audio import vae_decode_audio

TMP = "/tmp/opencode/k6_chain"
FPS = 24.0
SR = 44100
WIDTH = HEIGHT = 480
AUDIO_VAE = "/home/khrapov/AI/ComfyUI/models/vae/kandinsky6_audio_vae_full.safetensors"
VIDEO_VAE = "/home/khrapov/AI/ComfyUI/models/vae/hunyuan_video_vae_bf16.safetensors"


def norm_corr(a, b):
    a = a.flatten().float() - a.mean()
    b = b.flatten().float() - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def mel_corr(mc, w1, w2):
    with torch.no_grad():
        m1 = mc(w1.reshape(1, -1).to(mc.device).clamp(-1., 1.))
        m2 = mc(w2.reshape(1, -1).to(mc.device).clamp(-1., 1.))
    t = min(m1.shape[-1], m2.shape[-1])
    return norm_corr(m1[0, :, :t], m2[0, :, :t])


def write_mp4(path, frames, wav, fps=FPS, sr=SR):
    # two raw inputs cannot share one stdin pipe (ffmpeg interleaves reads on
    # the same fd); stage both to files instead
    f = (frames[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    w = wav.flatten().float().cpu().numpy().astype(np.float32)
    raw_path = path + ".raw"
    wav_path = path + ".f32"
    f.tofile(raw_path)
    w.tofile(wav_path)
    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "%dx%d" % (f.shape[2], f.shape[1]), "-r", str(fps), "-i", raw_path,
               "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", wav_path,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
               "-c:a", "aac", "-shortest", path]
        subprocess.run(cmd, check=True)
    finally:
        os.remove(raw_path)
        os.remove(wav_path)


def contact_sheet(path, tail, head, n=21, cell=240):
    """Two rows: the previous clip's last n frames, the next clip's first n."""
    rows = []
    for part in (tail, head):
        row = []
        for t in range(part.shape[1]):
            im = Image.fromarray((part[0, t].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
            row.append(im.resize((cell, cell)))
        rows.append(row)
    sheet = Image.new("RGB", (cell * n, cell * 2))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            sheet.paste(im, (c * cell, r * cell))
    sheet.save(path)


def main():
    # clone: the sampler ran under inference_mode, so the saved latents are
    # inference tensors and the VAE's in-place output pass refuses them
    # outside InferenceMode.
    clips = []
    for i in range(3):
        d = torch.load("%s/clip%d.pt" % (TMP, i))
        clips.append((d["video"].clone(), d["audio"].clone(),
                      d["trim_frames"], d["trim_samples"]))

    # pass 1: video VAE only, so its 24 GB decode transient (full or tiled)
    # never shares the card with the audio VAE
    sd, md = comfy.utils.load_torch_file(VIDEO_VAE, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=md)
    frames_list = []
    for i, (vid, aud, tf, ts) in enumerate(clips):
        # inference_mode, as on the server: the tiled fallback's
        # tiled_scale_multidim is @torch.inference_mode and its output pass
        # (process_output) does in-place math, which is only legal inside it.
        with torch.inference_mode():
            frames = vae.decode(vid.to("cuda"))  # [B, T, H, W, C]
        frames = frames.clone().cpu()
        assert frames.ndim == 5 and frames.shape[1] == 121, tuple(frames.shape)
        frames_list.append(frames)
        print("clip %d: video decoded %s" % (i, tuple(frames.shape)), flush=True)
    del vae
    torch.cuda.empty_cache()

    # pass 2: audio VAE only
    sd, md = comfy.utils.load_torch_file(AUDIO_VAE, return_metadata=True)
    avae = comfy.sd.VAE(sd=sd, metadata=md)
    melc = avae.first_stage_model.mel_converter

    track = torch.load(TMP + "/track44k.pt").flatten().float()
    trim = k6nodes.Kandinsky6MotionContextTrim()

    all_frames, all_wav, delivered = [], [], []
    prev_frames = None
    for i, (vid, aud, tf, ts) in enumerate(clips):
        # the production audio decode (layout [B, C, L] + std*5 normalization),
        # under inference_mode as on the server (see the video pass above)
        with torch.inference_mode():
            out_audio_in = vae_decode_audio(avae, {"samples": aud.to("cuda"), "sample_rate": SR})
        out_audio_in["waveform"] = out_audio_in["waveform"].clone()
        frames = frames_list[i]
        out_frames, out_audio = trim.trim(
            frames, tf,
            audio=out_audio_in,
            trim_samples=ts, fps=FPS, match_tail=True)
        n_f, n_w = out_frames.shape[1], int(out_audio["waveform"].shape[-1])
        want_w = round(n_f / FPS * SR)
        assert n_w == want_w, "clip %d: audio %d != %d for %d frames" % (i, n_w, want_w, n_f)
        print("clip %d: delivered %d frames / %d samples (%.4fs / %.4fs)"
              % (i, n_f, n_w, n_f / FPS, n_w / SR), flush=True)

        write_mp4("%s/clip%d.mp4" % (TMP, i), out_frames, out_audio["waveform"])

        # fidelity of the delivered sound against its window of the track
        if i == 0:
            lo = 0
        else:
            lo = i * (218 * 1024 - 38 * 1024) + 38 * 1024
        hi = lo + n_w
        ref_w = track[lo:hi]
        assert ref_w.numel() == n_w
        got_w = out_audio["waveform"].flatten().float()
        print("  vs track[%d:%d]: mel corr %.4f | waveform corr %.4f"
              % (lo, hi, mel_corr(melc, got_w, ref_w), norm_corr(got_w, ref_w)))

        if prev_frames is not None:
            tail = prev_frames[:, -21:].cpu()
            head = out_frames[:, :21].cpu()
            contact_sheet("%s/seam_%d_%d.png" % (TMP, i - 1, i), tail, head)
            d = (out_frames[:, :21] - prev_frames[:, -21:]).abs()
            print("  decoded video seam: mean abs %.5f | rel %.4f"
                  % (float(d.mean()), float(d.mean() / prev_frames[:, -21:].abs().mean())))
        prev_frames = out_frames.cpu()
        delivered.append((lo, hi, n_w))
        all_frames.append(out_frames[:, :].cpu())
        all_wav.append(out_audio["waveform"])

    full_frames = torch.cat(all_frames, dim=1)
    full_wav = torch.cat(all_wav, dim=-1)
    total_f, total_w = full_frames.shape[1], int(full_wav.shape[-1])
    drift = abs(total_w - total_f * SR / FPS)
    print("full chain: %d frames (%.4fs) / %d samples (%.4fs) | drift %.1f samples (%.3f ms)"
          % (total_f, total_f / FPS, total_w, total_w / SR, drift, drift / SR * 1000))
    assert drift < 1.0, "delivered A/V drift exceeds half a sample"

    # the delivered windows must tile the track in order: each start is the
    # previous end (modulo the grid overhang the match tail trims)
    for a, b in zip(delivered, delivered[1:]):
        gap = b[0] - a[1]
        print("  track gap between clip %d and %d: %d samples (%.1f ms)"
              % (a[0], b[0], gap, gap / SR * 1000))
        assert 0 <= gap < 2000, "track windows out of order or gapped"

    write_mp4(TMP + "/chain_full.mp4", full_frames, full_wav)
    print("DECODE_OK")


if __name__ == "__main__":
    main()
