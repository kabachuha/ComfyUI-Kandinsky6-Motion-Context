"""Prepare the 3-clip chain e2e: encode the track windows and the I2VA image.

Runs in its own process (VAE only, no DiT), so neither VAE ever shares the
rollout process's GPU. Encodes the three custom-audio windows of the track
with the full audio VAE and the character reference image with the Hunyuan
video VAE, saving the latents for the main process.

Windows advance by the DELIVERED length (clip_samples - head*1024), so the
head of window N+1 is the tail of window N: the track plays once, in order.

Args: <out_dir> <track.mp3> <audio_vae.safetensors> <video_vae.safetensors> <image.png>
"""
import os
import sys

sys.path.insert(0, "/home/khrapov/AI/ComfyUI")

import numpy as np
import torch
from PIL import Image

import comfy.audio
import comfy.sd
import comfy.utils
from comfy_extras.nodes_audio import VAEEncodeAudio, load as load_audio_file

CLIP_SAMPLES = 218 * 1024   # a 121-frame clip's audio latent
HEAD_SAMPLES = 38 * 1024    # the audio frames covering 21 pinned px


def main():
    out_dir, track_path, audio_vae_path, video_vae_path, image_path = sys.argv[1:6]
    os.makedirs(out_dir, exist_ok=True)

    sd, metadata = comfy.utils.load_torch_file(audio_vae_path, return_metadata=True)
    avae = comfy.sd.VAE(sd=sd, metadata=metadata)
    wav, sr = load_audio_file(track_path)
    wav = wav.unsqueeze(0)  # [1, C, L]
    if sr != 44100:
        wav = comfy.audio.resample(wav, sr, 44100)
    total = int(wav.shape[-1])
    need = 2 * (CLIP_SAMPLES - HEAD_SAMPLES) + CLIP_SAMPLES
    assert total >= need, "track too short: %d < %d samples" % (total, need)
    print("track: %d samples at 44100 Hz (%.2f s)" % (total, total / 44100))
    torch.save(wav.float().cpu(), out_dir + "/track44k.pt")

    for i in range(3):
        start = i * (CLIP_SAMPLES - HEAD_SAMPLES)
        seg = wav[..., start:start + CLIP_SAMPLES]
        enc = VAEEncodeAudio.execute(
            avae, {"waveform": seg, "sample_rate": 44100}).result[0]["samples"]
        torch.save(enc.cpu(), "%s/enc_w%d.pt" % (out_dir, i))
        print("window %d: track[%d:%d] -> latent %s"
              % (i, start, start + CLIP_SAMPLES, tuple(enc.shape)))
    del avae
    torch.cuda.empty_cache()

    sd, metadata = comfy.utils.load_torch_file(video_vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    img = Image.open(image_path).convert("RGB")
    x = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0).permute(2, 0, 1)[None]
    resized = comfy.utils.common_upscale(x, 480, 480, "bilinear", "center")
    ref = vae.encode(resized.movedim(1, -1))  # the VAE wrapper takes channel-last
    if ref.ndim == 4:
        ref = ref.unsqueeze(2)
    torch.save(ref.float().cpu(), out_dir + "/ref0.pt")
    print("reference image latent: %s" % (tuple(ref.shape),))
    print("PREP_OK")


if __name__ == "__main__":
    main()
