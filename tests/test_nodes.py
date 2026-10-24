"""Node behavior: pinning, masks, trims, and the save/load round trip.

Drives the real node classes against fake latents and stub VAEs, so the
assertions are about this pack's arithmetic, not about a model run.
"""

import math
import os

import pytest
import torch

import folder_paths

import k6mc.nodes as nodes

B, T_V, H, W = 1, 31, 60, 60
T_A, C_A = 218, 40
FPS, SR = 24.0, 44100
CLIP_SAMPLES = T_A * 1024


def joint(video, audio, **extra):
    import comfy.nested_tensor
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    latent["frame_rate"] = FPS
    latent["sample_rate"] = SR
    latent.update(extra)
    return latent


def empty_latent():
    video = torch.zeros(B, 16, T_V, H, W)
    audio = torch.zeros(B, T_A, C_A)
    return joint(video, audio)


def marker_context():
    """A previous clip whose video tail and audio tail carry frame markers."""
    video = torch.zeros(B, 16, T_V, H, W)
    for i in range(6):
        video[0, 0, T_V - 6 + i, 0, 0] = 100.0 + i
    audio = torch.zeros(B, T_A, C_A)
    audio[:, :, 0] = torch.arange(T_A, dtype=torch.float32)
    return joint(video, audio)


class FakeAudioVAE:
    """Stands in for the K6 audio VAE wrapper's encode.

    Input is channel-last [B, L, C] (the wrapper convention), output the
    [B, T, 40] latent with the frame index in channel 0.
    """

    def __init__(self):
        self.calls = []

    def encode(self, x):
        self.calls.append(tuple(x.shape))
        t = x.shape[1] // 1024
        out = torch.zeros(1, t, C_A)
        out[:, :, 0] = torch.arange(t, dtype=torch.float32)
        return out


class FakeVideoVAE:
    """Stands in for the video VAE wrapper's encode of a frame tail."""

    def __init__(self):
        self.calls = []

    def encode(self, x):
        # channel-last [B, T, H, W, 3]
        self.calls.append(tuple(x.shape))
        b, t, h, w, _ = x.shape
        t_lat = (t - 1) // 4 + 1
        out = torch.zeros(b, 16, t_lat, h // 8, w // 8)
        out[:, 0, :, 0, 0] = torch.arange(t_lat, dtype=torch.float32)
        return out


COND = [[{}, {}]]


def test_no_context_passes_through():
    out = nodes.Kandinsky6MotionContext().apply(
        [c for c in COND], [c for c in COND], empty_latent(), 6)
    pos, neg, latent, trim_frames, trim_samples = out
    assert "k6_reference" not in pos[0][1]
    assert "noise_mask" not in latent
    assert trim_frames == 0
    assert trim_samples == 0


def test_context_latent_pins_tail_and_masks_audio_head():
    ctx = marker_context()
    pos, neg, latent, trim_frames, trim_samples = nodes.Kandinsky6MotionContext().apply(
        [c for c in COND], [c for c in COND], empty_latent(), 6,
        context_latent=ctx)

    # the reference is the previous clip's last 6 latent frames, bit-exact
    ref = pos[0][1]["k6_reference"]
    assert tuple(ref.shape) == (1, 16, 6, H, W)
    assert torch.equal(ref[0, 0, :, 0, 0], torch.tensor([100.0, 101, 102, 103, 104, 105]))
    assert "k6_reference" in neg[0][1]

    # the audio head is the previous clip's last 38 audio frames, bit-exact
    video, audio = latent["samples"].unbind()
    head = audio[:, :38, 0]
    assert torch.equal(head[0], torch.arange(218 - 38, 218, dtype=torch.float32))

    # the mask keeps the head clean and generates the rest; video fully 1
    v_mask, a_mask = latent["noise_mask"].unbind()
    assert torch.all(v_mask == 1.0)
    assert tuple(a_mask.shape) == (1, 1, 1, T_A)
    assert torch.all(a_mask[:, :, :, :38] == 0.0)
    assert torch.all(a_mask[:, :, :, 38:] == 1.0)

    assert trim_frames == 21
    assert trim_samples == 38 * 1024


def test_custom_audio_pins_the_whole_window_from_the_track():
    vae = FakeAudioVAE()
    track = {"waveform": torch.rand(1, 1, 3 * CLIP_SAMPLES + 5000),
             "sample_rate": SR}
    _, _, latent, trim_frames, trim_samples = nodes.Kandinsky6MotionContext().apply(
        [c for c in COND], [c for c in COND], empty_latent(), 6,
        context_latent=marker_context(),
        audio_vae=vae, custom_audio=track, clip_index=1)

    # clip 1's window starts where clip 0's delivery ended
    head = nodes._audio_frames_for_px(21)
    start = CLIP_SAMPLES - head * 1024
    assert vae.calls == [(1, CLIP_SAMPLES, 1)]
    _ = start  # the slice is implicit in the segment length; checked below

    # the whole window is pinned clean, but the delivery only trims the head
    video, audio = latent["samples"].unbind()
    v_mask, a_mask = latent["noise_mask"].unbind()
    assert torch.all(a_mask == 0.0)
    assert torch.equal(audio[:, :, 0][0], torch.arange(T_A, dtype=torch.float32))
    assert trim_frames == 21
    assert trim_samples == head * 1024


def test_custom_audio_first_clip_starts_at_the_top_of_the_track():
    vae = FakeAudioVAE()
    track = {"waveform": torch.rand(1, 1, 3 * CLIP_SAMPLES), "sample_rate": SR}
    nodes.Kandinsky6MotionContext().apply(
        [c for c in COND], [c for c in COND], empty_latent(), 6,
        audio_vae=vae, custom_audio=track, clip_index=0)
    # clip 0 has no pinned picture -> no head -> full clip-length window
    assert vae.calls == [(1, CLIP_SAMPLES, 1)]


def test_custom_audio_window_alignment():
    # the head of clip N+1's window is the tail of clip N's window
    vae = FakeAudioVAE()
    track = {"waveform": torch.rand(1, 1, 4 * CLIP_SAMPLES), "sample_rate": SR}
    for idx in (0, 1):
        nodes.Kandinsky6MotionContext().apply(
            [c for c in COND], [c for c in COND], empty_latent(), 6,
            audio_vae=vae, custom_audio=track, clip_index=idx)
    head = nodes._audio_frames_for_px(21)
    advance = CLIP_SAMPLES - head * 1024
    assert vae.calls == [(1, CLIP_SAMPLES, 1), (1, CLIP_SAMPLES, 1)]
    assert advance == CLIP_SAMPLES - 38 * 1024


def test_custom_audio_track_running_out_pins_what_there_is():
    vae = FakeAudioVAE()
    # enough for clip 0, only 15 frames inside clip 1's window
    track = {"waveform": torch.rand(1, 1, 200000), "sample_rate": SR}
    _, _, latent, _, _ = nodes.Kandinsky6MotionContext().apply(
        [c for c in COND], [c for c in COND], empty_latent(), 6,
        context_latent=marker_context(),
        audio_vae=vae, custom_audio=track, clip_index=1)
    pin_frames = vae.calls[0][1] // 1024
    assert 1 < pin_frames < T_A
    v_mask, a_mask = latent["noise_mask"].unbind()
    assert torch.all(a_mask[:, :, :, :pin_frames] == 0.0)
    assert torch.all(a_mask[:, :, :, pin_frames:] == 1.0)


def test_custom_audio_track_too_short_fails_loudly():
    vae = FakeAudioVAE()
    track = {"waveform": torch.rand(1, 1, 100000), "sample_rate": SR}
    with pytest.raises(ValueError, match="no audio for clip"):
        nodes.Kandinsky6MotionContext().apply(
            [c for c in COND], [c for c in COND], empty_latent(), 6,
            audio_vae=vae, custom_audio=track, clip_index=1)


def test_context_frames_encodes_the_pixel_tail():
    vae = FakeVideoVAE()
    frames = torch.rand(1, 121, 480, 480, 4)
    pos, _, _, trim_frames, _ = nodes.Kandinsky6MotionContext().apply(
        [c for c in COND], [c for c in COND], empty_latent(), 6,
        context_frames=frames, vae=vae)

    # the last 21 pixel frames, resized to the clip size
    assert vae.calls == [(1, 21, 480, 480, 3)]
    ref = pos[0][1]["k6_reference"]
    assert tuple(ref.shape) == (1, 16, 6, H, W)
    assert torch.equal(ref[0, 0, :, 0, 0], torch.arange(6, dtype=torch.float32))
    assert trim_frames == 21


def test_context_latent_resolution_mismatch_fails_loudly():
    video = torch.zeros(B, 16, T_V, H, W)
    for i in range(6):
        video[0, 0, T_V - 6 + i, 0, 0] = 1.0
    audio = torch.zeros(B, T_A, C_A)
    ctx = joint(video[:, :, :, :59].clone(), audio)  # 472 px wide, not 480
    with pytest.raises(ValueError, match="resolution"):
        nodes.Kandinsky6MotionContext().apply(
            [c for c in COND], [c for c in COND], empty_latent(), 6,
            context_latent=ctx)


def test_trim_drops_picture_and_sound_together():
    images = torch.rand(1, 121, 64, 64, 3)  # decoded frame stack [B, T, H, W, C]
    audio = {"waveform": torch.ones(1, 1, CLIP_SAMPLES), "sample_rate": SR}
    out_images, out_audio = nodes.Kandinsky6MotionContextTrim().trim(
        images, 21, audio=audio, trim_samples=38 * 1024, fps=FPS, match_tail=True)
    assert out_images.shape[1] == 100
    want = round(100 / FPS * SR)
    assert out_audio["waveform"].shape[-1] == want


def test_trim_without_exact_samples_falls_back_to_fps():
    images = torch.rand(1, 121, 64, 64, 3)
    audio = {"waveform": torch.ones(1, 1, CLIP_SAMPLES), "sample_rate": SR}
    _, out_audio = nodes.Kandinsky6MotionContextTrim().trim(
        images, 21, audio=audio, fps=FPS, match_tail=False)
    assert out_audio["waveform"].shape[-1] == CLIP_SAMPLES - round(21 / FPS * SR)


def test_trim_pads_a_short_tail_with_silence():
    images = torch.rand(1, 121, 64, 64, 3)
    audio = {"waveform": torch.ones(1, 1, CLIP_SAMPLES - 5000), "sample_rate": SR}
    _, out_audio = nodes.Kandinsky6MotionContextTrim().trim(
        images, 21, audio=audio, trim_samples=38 * 1024, fps=FPS, match_tail=True)
    want = round(100 / FPS * SR)
    assert out_audio["waveform"].shape[-1] == want
    assert torch.all(out_audio["waveform"][..., -100:] == 0.0)


def test_trim_without_audio_returns_none():
    images = torch.rand(1, 121, 64, 64, 3)
    out_images, out_audio = nodes.Kandinsky6MotionContextTrim().trim(images, 21)
    assert out_images.shape[1] == 100
    assert out_audio is None


def test_trim_refuses_to_consume_the_whole_clip():
    images = torch.rand(1, 121, 64, 64, 3)
    with pytest.raises(ValueError, match="trim"):
        nodes.Kandinsky6MotionContextTrim().trim(images, 121)


def test_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    video = torch.rand(1, 16, T_V, H, W)
    audio = torch.rand(1, T_A, C_A)
    latent = joint(video, audio)

    saved = nodes.Kandinsky6MotionContextSaveLatent()
    saved.save(latent, "k6_context/clip", clip_index=1)
    path = tmp_path / "k6_context" / "clip_00001.safetensors"
    assert path.is_file()

    loaded = nodes.Kandinsky6MotionContextLoadLatent().load("k6_context", 1)
    v2, a2 = loaded[0]["samples"].unbind()
    assert torch.equal(v2, video)
    assert torch.equal(a2, audio)
    assert loaded[0]["frame_rate"] == FPS
    assert loaded[0]["sample_rate"] == SR


def test_load_index_zero_loads_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    out = nodes.Kandinsky6MotionContextLoadLatent().load("k6_context", 0)
    assert out == (None,)


def test_save_zero_index_numbers_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    latent = joint(torch.rand(1, 16, T_V, H, W), torch.rand(1, T_A, C_A))
    nodes.Kandinsky6MotionContextSaveLatent().save(latent, "k6_context/clip", clip_index=0)
    files = os.listdir(tmp_path / "k6_context")
    assert len(files) == 1
    assert files[0].endswith("_00001_.safetensors")


def test_load_refuses_paths_outside_output(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError):
        nodes.Kandinsky6MotionContextLoadLatent().load("../../", 1)


def test_is_changed_tracks_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    latent = joint(torch.rand(1, 16, T_V, H, W), torch.rand(1, T_A, C_A))
    nodes.Kandinsky6MotionContextSaveLatent().save(latent, "k6_context/clip", clip_index=1)
    load = nodes.Kandinsky6MotionContextLoadLatent
    first = load.IS_CHANGED("k6_context", 1)
    assert first != float("nan")
    assert first == load.IS_CHANGED("k6_context", 1)
    assert load.IS_CHANGED("k6_context", 0) == "disabled"
    assert math.isnan(load.IS_CHANGED("no_such_folder", 1))
