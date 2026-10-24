"""Pin the tail of a previous Kandinsky 6 clip so the next one continues it.

Wire the Motion Context node between the conditioning and the guider:

    Kandinsky6EmptyLatent -> Kandinsky6ImageToVideoAudio (optional)
        -> K6 Motion Context
        -> Kandinsky6PiFlowGuider -> SamplerCustomAdvanced

The pinned picture is a run of whole latent frames from the previous
clip's tail, handed to the DiT as a multi-frame I2VA reference: the model
appends it as clean tail frames with their own RoPE positions, so it reads
real motion instead of guessing it from a single still. The pinned sound
goes on this clip's own timeline through the audio noise mask: it ends at
the join and reaches backward into the sound that has already played. The
pinned head sits at the start of the delivered timeline, where the Trim
node removes it, picture and sound together.

With custom_audio wired, the clip's sound is not continued from the
previous clip but taken from the supplied track: this clip's window of the
track is pinned clean over the audio stream, so the model only generates
the picture and lip-syncs it to the exact speech. The windows advance by
the delivered length, so the track plays once, in order, with no gap or
repeat at the join.

Nothing in ComfyUI is modified. The nodes check their assumptions against
the live K6 code before the first render, and if an update breaks one they
refuse to run and say what moved.
"""

import gc
import logging
import os
import re

import comfy.audio
import comfy.nested_tensor
import comfy.utils
import folder_paths
import node_helpers
import torch
from comfy.ldm.kandinsky6.core_contract import (
    GENERATION_DEFAULTS,
    LATENT_DEFAULTS,
)

from .k6_contract import ensure_k6_contract

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:  # ComfyUI always ships safetensors; belt and braces
    _st_load = _st_save = None

_LOG = logging.getLogger("k6_motion_context")

FPS = float(GENERATION_DEFAULTS["fps"])
VIDEO_CHANNELS = int(LATENT_DEFAULTS["video_channels"])
VIDEO_SPATIAL = int(LATENT_DEFAULTS["video_spatial_compression_factor"])
VIDEO_TEMPORAL = int(LATENT_DEFAULTS["video_temporal_compression_factor"])
AUDIO_CHANNELS = int(LATENT_DEFAULTS["audio_channels"])
AUDIO_DOWNSAMPLE = int(LATENT_DEFAULTS["audio_downsample_factor"])
AUDIO_SAMPLE_RATE = int(LATENT_DEFAULTS["audio_sample_rate"])


def _pixel_frames(k):
    """Pixel frames a run of k latent frames covers from a head position.

    The causal video VAE gives the first latent frame one pixel frame and
    each later one four, so k frames cover 4k - 3 pixels.
    """
    return (k - 1) * VIDEO_TEMPORAL + 1


def _audio_frames_for_px(px, fps=FPS, sr=AUDIO_SAMPLE_RATE):
    """Audio latent frames covering a window of `px` pixel frames."""
    return max(0, int(round(px / float(fps) * sr / AUDIO_DOWNSAMPLE)))


def _joint_streams(latent):
    """Unpack a K6 joint latent into (video, audio) streams."""
    samples = latent.get("samples")
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "k6_motion_context: expected a Kandinsky 6 joint latent "
            "(nested video/audio streams), got %r." % type(samples))
    streams = samples.unbind()
    if len(streams) != 2:
        raise ValueError(
            "k6_motion_context: expected exactly two latent streams, got %d."
            % len(streams))
    return streams


def _track_segment(track, start, length, clip_index):
    """Slice this clip's window out of the custom track, in whole latent
    frames, resampled to the audio VAE's rate."""
    wave = track["waveform"]  # [B, C, L]
    sr = int(track["sample_rate"])
    if sr != AUDIO_SAMPLE_RATE:
        wave = comfy.audio.resample(wave, sr, AUDIO_SAMPLE_RATE)
    seg = wave[..., start:start + length]
    have = int(seg.shape[-1])
    whole = have // AUDIO_DOWNSAMPLE * AUDIO_DOWNSAMPLE
    if whole < AUDIO_DOWNSAMPLE:
        raise ValueError(
            "k6_motion_context: the custom track has no audio for clip %d's "
            "window (it starts at sample %d, the track has %d samples at "
            "%d Hz). Extend the track or stop the chain here."
            % (clip_index, start, have, sr))
    return seg[..., :whole]


def _encode_audio_tail(audio_vae, wave, want_samples):
    """Encode the last `want_samples` of a waveform, in whole latent frames.

    Returns ([1, T, 40] latent, T). Takes what there is when the waveform
    is shorter than the window.
    """
    have = int(wave.shape[-1])
    take = min(want_samples, have)
    whole = take // AUDIO_DOWNSAMPLE * AUDIO_DOWNSAMPLE
    if whole < AUDIO_DOWNSAMPLE:
        raise ValueError(
            "k6_motion_context: the audio to pin is shorter than one latent "
            "frame (%d samples); there is nothing to pin." % have)
    if whole < take:
        _LOG.info("k6_motion_context: pinning %.1fms less audio than asked; "
                  "the tail is not a whole number of latent frames.",
                  (take - whole) / AUDIO_SAMPLE_RATE * 1000.0)
    tail = wave[..., have - whole:]
    return audio_vae.encode(tail[:1].movedim(1, -1))


class Kandinsky6MotionContext:
    """Pin a run of previous-clip frames and the sound at the join.

    See the module docstring for the wiring. With context_latent wired,
    both picture and sound are sliced straight out of the previous clip's
    latent; context_frames / context_audio are the decode-and-encode path
    for when only pixels are on hand. custom_audio pins this clip's window
    of a supplied track clean instead, so the picture lip-syncs to it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {
                    "tooltip": "This clip's positive conditioning. Any k6_reference it already carries (e.g. from Kandinsky 6 Image to Video+Audio) is kept when no context is wired and replaced by the pinned window when one is."}),
                "negative": ("CONDITIONING",),
                "empty_latent": ("LATENT", {
                    "tooltip": "This clip's joint video+audio latent (Kandinsky 6 Empty Latent, or the one an upstream K6 node passes through)."}),
                "context_length": ("INT", {
                    "default": 6, "min": 1, "max": 15,
                    "tooltip": "Latent frames of the previous clip's picture to pin: 6 covers 21 pixel frames (~0.9s) of real motion, 10 covers 37 (~1.5s). Longer windows pin more motion but come off the front of the delivered clip. Only meaningful with a context wired."}),
            },
            "optional": {
                "context_latent": ("LATENT", {
                    "tooltip": "Previous clip's SAMPLER OUTPUT latent (the same one you wire into the decode nodes and the Save node). Supplies both picture and sound, sliced straight out, skipping the decode and re-encode that cost a little quality at every link. Must be the same resolution as this clip."}),
                "context_frames": ("IMAGE", {
                    "tooltip": "Decoded frames of the previous clip. Used when no context_latent is wired; the last frames are resized and VAE-encoded, so wire the video VAE as well."}),
                "vae": ("VAE", {
                    "tooltip": "Video VAE. Needed only for the context_frames path."}),
                "audio_vae": ("VAE", {
                    "tooltip": "K6 audio VAE. Needed for custom_audio and context_audio."}),
                "custom_audio": ("AUDIO", {
                    "tooltip": "A supplied track (e.g. the speech for a talking person). This clip's window of it is pinned clean over the whole audio stream, so the model generates only the picture and lip-syncs to the exact sound. Windows advance by the delivered length, so the track plays once, in order, across the chain."}),
                "context_audio": ("AUDIO", {
                    "tooltip": "Decoded audio of the previous clip, for continuing generated sound when no context_latent is wired. Wire the audio VAE as well. Ignored when custom_audio or context_latent is wired."}),
                "audio_context_frames": ("INT", {
                    "default": 0, "min": 0, "max": 512,
                    "tooltip": "Audio latent frames to pin, independent of the picture window. 0 follows it. The window is END-aligned with the pinned video, so this only controls how far back the sound reaches. 1024 samples per frame at 44.1 kHz."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Which clip of the chain this is, for slicing custom_audio: clip 0 takes the start of the track. Set it to match the Load/Save indices."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "joint_latent", "trim_frames", "trim_samples")
    FUNCTION = "apply"
    CATEGORY = "model/kandinsky6"
    DESCRIPTION = ("Pin the tail of a previous Kandinsky 6 clip - picture and "
                   "sound - so the next clip continues it, or pin a window of "
                   "a custom audio track clean for lip-sync.")

    def apply(self, positive, negative, empty_latent, context_length,
              context_latent=None, context_frames=None, vae=None,
              audio_vae=None, custom_audio=None, context_audio=None,
              audio_context_frames=0, clip_index=0):
        ensure_k6_contract()

        video, audio = _joint_streams(empty_latent)
        if video.ndim != 5 or video.shape[1] != VIDEO_CHANNELS:
            raise ValueError(
                "k6_motion_context: expected a K6 video latent "
                "[B, %d, T, H, W], got shape %s."
                % (VIDEO_CHANNELS, tuple(video.shape)))
        if audio.ndim != 3 or audio.shape[-1] != AUDIO_CHANNELS:
            raise ValueError(
                "k6_motion_context: expected a K6 audio latent "
                "[B, T, %d], got shape %s."
                % (AUDIO_CHANNELS, tuple(audio.shape)))
        batch = int(video.shape[0])
        t_new = int(video.shape[2])
        t_audio = int(audio.shape[1])
        fps = float(empty_latent.get("frame_rate", FPS))
        sr = int(empty_latent.get("sample_rate", AUDIO_SAMPLE_RATE))
        clip_index = int(clip_index)
        if clip_index < 0:
            raise ValueError("k6_motion_context: clip_index must be >= 0.")

        # The picture window. Slicing it out of the previous clip's latent
        # keeps the blocks bit-identical to what the model produced; the
        # frames path is the decode-and-encode fallback.
        ref = None
        k = 0
        video_src = "off"
        if context_latent is not None:
            ctx_video, _ = _joint_streams(context_latent)
            if ctx_video.ndim != 5 or ctx_video.shape[1] != VIDEO_CHANNELS:
                raise ValueError(
                    "k6_motion_context: context_latent has no K6 video "
                    "stream; wire the sampler output of a K6 AV graph.")
            if tuple(ctx_video.shape[3:]) != tuple(video.shape[3:]):
                raise ValueError(
                    "k6_motion_context: context_latent is %dx%d but this clip "
                    "is %dx%d. A latent cannot be resized, so the previous "
                    "clip has to be regenerated at this resolution, or the "
                    "chain restarted here."
                    % (ctx_video.shape[4] * VIDEO_SPATIAL,
                       ctx_video.shape[3] * VIDEO_SPATIAL,
                       video.shape[4] * VIDEO_SPATIAL,
                       video.shape[3] * VIDEO_SPATIAL))
            k = min(int(context_length), int(ctx_video.shape[2]), t_new - 1)
            if k < 1:
                raise ValueError(
                    "k6_motion_context: context_latent has no frames to pin "
                    "(%d latent frames, this clip has %d)."
                    % (int(ctx_video.shape[2]), t_new))
            ref = ctx_video[:1, :, -k:].clone().to(device=video.device, dtype=video.dtype)
            video_src = "latent"
        elif context_frames is not None:
            if vae is None:
                raise ValueError(
                    "k6_motion_context: context_frames needs the video VAE "
                    "wired to encode them.")
            # a decoded clip is a frame stack [B, T, H, W, C]
            n_px = int(context_frames.shape[1])
            k = min(int(context_length), (n_px + 3) // VIDEO_TEMPORAL, t_new - 1)
            if k < 1:
                raise ValueError(
                    "k6_motion_context: context_frames has no frames to pin.")
            px = _pixel_frames(k)
            tail = context_frames[:1, -px:, :, :, :3].movedim(-1, 1)
            tail = comfy.utils.common_upscale(
                tail,
                int(video.shape[4]) * VIDEO_SPATIAL,
                int(video.shape[3]) * VIDEO_SPATIAL,
                "bilinear", "center")
            enc = vae.encode(tail.movedim(1, -1))
            if enc.ndim != 5:
                raise ValueError(
                    "k6_motion_context: the video VAE returned shape %s for "
                    "the context frames, expected [B, C, T, H, W]."
                    % (tuple(enc.shape),))
            if tuple(enc.shape[3:]) != tuple(video.shape[3:]):
                raise ValueError(
                    "k6_motion_context: the encoded context frames are %dx%d "
                    "but this clip is %dx%d; wire them at the same size."
                    % (enc.shape[4] * VIDEO_SPATIAL,
                       enc.shape[3] * VIDEO_SPATIAL,
                       video.shape[4] * VIDEO_SPATIAL,
                       video.shape[3] * VIDEO_SPATIAL))
            k = min(k, int(enc.shape[2]))
            ref = enc[:1, :, -k:].clone().to(device=video.device, dtype=video.dtype)
            video_src = "pixels"

        # How far back the sound reaches: the audio window end-aligned with
        # the pinned video, or an explicit count.
        head = int(audio_context_frames)
        if head <= 0:
            head = _audio_frames_for_px(_pixel_frames(k), fps, sr) if k > 0 else 0
        head = max(0, min(head, t_audio))

        # The pinned sound.
        pin = None  # [1, T_pin, 40]
        audio_src = "off"
        if custom_audio is not None:
            if audio_vae is None:
                raise ValueError(
                    "k6_motion_context: custom_audio needs the K6 audio VAE "
                    "wired.")
            # The window starts where the previous clip's delivered sound
            # ended: advance by the delivered length, not the clip length,
            # so the track plays once with no gap at the join.
            clip_samples = t_audio * AUDIO_DOWNSAMPLE
            start = clip_index * (clip_samples - head * AUDIO_DOWNSAMPLE)
            seg = _track_segment(custom_audio, start, clip_samples, clip_index)
            pin = audio_vae.encode(seg[:1].movedim(1, -1))
            audio_src = "custom track, clip %d" % clip_index
        elif context_latent is not None:
            _, ctx_audio = _joint_streams(context_latent)
            if ctx_audio.ndim != 3:
                raise ValueError(
                    "k6_motion_context: context_latent has no audio stream. "
                    "Wire the sampler output of a K6 AV graph.")
            if head > 0:
                head = min(head, int(ctx_audio.shape[1]))
                pin = ctx_audio[:1, -head:, :].clone()
                audio_src = "latent"
        elif context_audio is not None:
            if audio_vae is None:
                raise ValueError(
                    "k6_motion_context: context_audio needs the K6 audio VAE "
                    "wired, or wire context_latent instead.")
            if head > 0:
                wave = context_audio["waveform"]  # [B, C, L]
                if int(context_audio["sample_rate"]) != sr:
                    wave = comfy.audio.resample(wave, int(context_audio["sample_rate"]), sr)
                pin = _encode_audio_tail(audio_vae, wave, head * AUDIO_DOWNSAMPLE)
                head = min(head, int(pin.shape[1]))
                audio_src = "decoded"

        if pin is not None:
            pin = pin.to(device=audio.device, dtype=audio.dtype)

        # This clip's sampler input: the empty video, the pinned sound at
        # the head, and the mask that keeps the pinned sound clean.
        out_audio = torch.zeros_like(audio)
        out_latent = {"samples": None}
        if "frame_rate" in empty_latent:
            out_latent["frame_rate"] = empty_latent["frame_rate"]
        if "sample_rate" in empty_latent:
            out_latent["sample_rate"] = empty_latent["sample_rate"]
        if pin is not None:
            t_pin = int(pin.shape[1])
            out_audio[:, :t_pin] = pin[:1]
            mask = torch.ones(batch, 1, 1, t_audio, device=audio.device)
            mask[:, :, :, :t_pin] = 0.0
            out_latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (torch.ones_like(video), mask))
        out_latent["samples"] = comfy.nested_tensor.NestedTensor((video, out_audio))

        if ref is not None:
            values = {"k6_reference": ref}
            positive = node_helpers.conditioning_set_values(positive, values)
            negative = node_helpers.conditioning_set_values(negative, values)

        trim_frames = _pixel_frames(k) if ref is not None else 0
        # The delivery trims the HEAD that has already played, not the whole
        # pin: for a custom track the pin covers the entire window but the
        # listener already heard its first `head` frames in the previous clip.
        trim_samples = head * AUDIO_DOWNSAMPLE if pin is not None else 0
        _LOG.info(
            "k6_motion_context: %d-frame clip, picture pinned %d latent "
            "frames (%d px) from %s, audio pinned %d latent frames (%.3fs) "
            "from %s, trim %d frames / %d samples.",
            (t_new - 1) * VIDEO_TEMPORAL + 1, k if ref is not None else 0,
            trim_frames, video_src, int(pin.shape[1]) if pin is not None else 0,
            (int(pin.shape[1]) * AUDIO_DOWNSAMPLE / sr if pin is not None else 0.0),
            audio_src, trim_frames, trim_samples)
        return (positive, negative, out_latent, trim_frames, trim_samples)


class Kandinsky6MotionContextTrim:
    """Drop the pinned head off a decoded clip, picture and sound together.

    The pinned head occupies the start of the delivered timeline, so it has
    to come off before concatenating. Trimming only the images would leave
    the audio a full head longer than the video, and muxing those puts the
    whole soundtrack ahead of the picture. Wire trim_frames and
    trim_samples from the Motion Context node so the counts follow whatever
    it actually pinned.

    The tail gets the same treatment for a different reason: K6's audio
    latent runs at 44100/1024 Hz against 24 fps picture, so each clip
    ships a few milliseconds more sound than picture. match_tail trims a
    long tail or zero-pads a short one to exactly frames/fps, so the error
    does not accumulate down a chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Decoded audio for the same clip. Trimmed by the matching duration so sound stays locked to picture. Leave unwired for silent clips."}),
                "trim_samples": ("INT", {
                    "default": 0, "min": 0, "max": 10 ** 9,
                    "tooltip": "Exact audio samples to cut from the head, from the Motion Context node's trim_samples. 0 falls back to trim_frames / fps."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate used to convert the trim into an audio duration when trim_samples is 0. Must match what you feed Create Video."}),
                "match_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Make the audio duration equal frames/fps exactly, trimming a long tail or padding a short one with silence, so per-clip A/V error does not accumulate across a chain."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "model/kandinsky6"
    DESCRIPTION = ("Remove the leading pinned frames from a decoded Kandinsky "
                   "6 clip, trimming picture and sound by the same span.")

    def trim(self, images, trim_frames, audio=None, trim_samples=0,
             fps=24.0, match_tail=True):
        # the VAE Decode node collapses the batch and delivers
        # [T, H, W, C]; a raw vae.decode (the e2e harness) delivers
        # [B, T, H, W, C]
        if images.ndim not in (4, 5):
            raise ValueError(
                "k6_motion_context: trim expects the decoded frame stack "
                "[T, H, W, C] or [B, T, H, W, C], got %d dims." % images.ndim)
        frame_axis = images.ndim - 4  # 0 for [T, H, W, C], 1 for [B, T, H, W, C]
        n = max(0, int(trim_frames))
        total = int(images.shape[frame_axis])
        if n >= total:
            raise ValueError(
                "k6_motion_context: asked to trim %d frames from a %d frame "
                "clip." % (n, total))
        out_images = images.narrow(frame_axis, n, total - n) if n else images

        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            cut = int(trim_samples)
            if cut <= 0:
                cut = int(round(n / float(fps) * sr))
            length = int(waveform.shape[-1])
            if cut >= length:
                raise ValueError(
                    "k6_motion_context: trimming %d samples from %d samples "
                    "of audio would leave nothing. Check that the trim "
                    "outputs come from the Motion Context node of this clip."
                    % (cut, length))
            waveform = waveform[..., cut:]

            if match_tail:
                frames_left = total - n
                want = int(round(frames_left / float(fps) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    waveform = waveform[..., :want]
                    _LOG.info("k6_motion_context: tail trimmed %d samples "
                              "(%.2fms) so audio matches %d frames exactly.",
                              have - want, (have - want) / sr * 1000.0,
                              frames_left)
                elif have < want:
                    # The missing samples are sound the model never
                    # generated; silence is the only honest fill. Leaving it
                    # short instead drifts every later clip earlier, and that
                    # error compounds down the chain.
                    missing = want - have
                    waveform = torch.nn.functional.pad(waveform, (0, missing))
                    _LOG.info("k6_motion_context: tail padded %d zero samples "
                              "(%.2fms) so audio matches %d frames exactly.",
                              missing, missing / sr * 1000.0, frames_left)

            out_audio = {"waveform": waveform, "sample_rate": sr}
            _LOG.info(
                "k6_motion_context: %d frames / %.4fs picture, %.4fs sound, "
                "drift %.2fms.",
                total - n, (total - n) / float(fps),
                int(waveform.shape[-1]) / sr,
                abs((total - n) / float(fps) - int(waveform.shape[-1]) / sr)
                * 1000.0)
        elif n:
            _LOG.info("k6_motion_context: trimmed %d leading frames, %d "
                      "remain. No audio wired; if this clip has sound, mux it "
                      "through this node or it will run %.3fs ahead of the "
                      "picture.", n, total - n, n / float(fps))

        return (out_images, out_audio)


def _under_root(root, path, default=""):
    """Resolve `path` inside `root`. None if it would escape.

    Absolute paths are allowed only when they already live under root.
    Relative paths join onto root. Compared case-insensitively where the
    filesystem is (normcase), so D:\\Temp and D:\\TEMP are the same root.
    """
    root = os.path.realpath(root)
    p = (path or "").strip().strip('"').strip("'") or default
    if not p:
        return None
    resolved = os.path.realpath(p if os.path.isabs(p) else os.path.join(root, p))
    r, c = os.path.normcase(root), os.path.normcase(resolved)
    if c != r and not c.startswith(r.rstrip(os.sep) + os.sep):
        return None
    return resolved


def _under_output(path):
    """Resolve `path` inside ComfyUI's output folder. None if it would escape.

    This is what keeps latent_path from becoming an arbitrary file read or a
    delete-anywhere on Clear.
    """
    return _under_root(folder_paths.get_output_directory(), path, "k6_context")


LATENT_FOLDERS = ("output", "input")


def _resolve_input_latent(path):
    """A latent FILE inside ComfyUI's input folder (uploaded via /upload/image).

    Input mode never picks slots from a folder: the uploader names the exact
    file, so there is nothing to choose between.
    """
    c = _under_root(folder_paths.get_input_directory(), path)
    if not c:
        raise FileNotFoundError(
            "k6_motion_context: path must name a file inside the ComfyUI "
            "input folder.")
    if not os.path.isfile(c):
        raise FileNotFoundError(
            "k6_motion_context: no latent file %r in the ComfyUI input "
            "folder." % (path or "").strip())
    return c


def _resolve_latent_path(path, clip_index=0, folder="output"):
    """Turn the loader's path input into a concrete file under output/.

    Accepts a path relative to ComfyUI's output folder, an absolute path
    that already lives there, or a directory in either form. For a
    directory, clip_index must be a positive slot: clip 1 is
    *_00001.safetensors. Auto-mode files carry a trailing underscore
    (*_00001_.safetensors) and are never matched, because their numbers
    count runs and could hold a reject. clip_index 0 is handled by the
    Load node itself (no file, first clip).
    """
    if folder == "input":
        return _resolve_input_latent(path)
    if folder != "output":
        raise ValueError("k6_motion_context: folder must be one of %s, got %r."
                         % (", ".join(LATENT_FOLDERS), folder))
    c = _under_output(path)
    if not c:
        raise FileNotFoundError(
            "k6_motion_context: path must stay inside the ComfyUI output "
            "folder.")
    if os.path.isfile(c):
        return c
    if os.path.isdir(c):
        idx = int(clip_index)
        if idx <= 0:
            raise FileNotFoundError(
                "k6_motion_context: clip_index 0 does not load a file.")
        endings = ("_%05d.safetensors" % idx,)
        files = [os.path.join(c, f) for f in os.listdir(c)
                 if f.endswith(endings)]
        if not files:
            near = [f for f in os.listdir(c)
                    if f.endswith("_%05d_.safetensors" % idx)]
            hint = ""
            if near:
                hint = (" Found %s, which is an auto-numbered save "
                        "(trailing underscore = numbered by RUN, so "
                        "it may be a reject). If it really is clip "
                        "%d, rename it to drop the trailing "
                        "underscore: %s" %
                        (near[0],
                         idx,
                         near[0].replace("_%05d_" % idx,
                                         "_%05d" % idx)))
            raise FileNotFoundError(
                "k6_motion_context: no saved latent for clip %d "
                "(no *_%05d.safetensors in %s).%s"
                % (idx, idx, c, hint))
        return max(files, key=os.path.getmtime)
    raise FileNotFoundError(
        "k6_motion_context: %r is neither a file nor a folder under the "
        "ComfyUI output directory." % ((path or "").strip() or "k6_context"))


def _latent_folder(path):
    """Directory that holds this chain's slot files, or None.

    Refuses anything outside ComfyUI's output folder so Clear cannot
    delete numbered safetensors elsewhere on disk.
    """
    c = _under_output(path)
    if not c:
        return None
    if os.path.isdir(c):
        return c
    if os.path.isfile(c):
        return os.path.dirname(c)
    return None


def _clip_slot_exists(latent_path, clip_index=1):
    try:
        _resolve_latent_path(latent_path, int(clip_index))
        return True
    except FileNotFoundError:
        return False


_CHAIN_SLOT_FILE = re.compile(r"(?:_\d{5}|_\d{5}_)\.safetensors(?:\.tmp)?$")


def _clear_clip_slots(latent_path):
    """Delete numbered chain slots. Custom filenames are left alone."""
    folder = _latent_folder(latent_path)
    if not folder:
        return 0
    removed = 0
    for name in os.listdir(folder):
        if not _CHAIN_SLOT_FILE.search(name):
            continue
        fp = os.path.join(folder, name)
        try:
            os.remove(fp)
        except OSError:
            gc.collect()
            os.remove(fp)
        removed += 1
    if removed:
        _LOG.info("k6_motion_context: cleared %d chain slot(s) from %s",
                  removed, folder)
    return removed


def register_chain_routes():
    """POST slot_exists / clear_latents for the Chain node's start and Clear."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None or getattr(register_chain_routes, "_done", False):
        return

    from .csrf_guard import require_same_origin

    @server.routes.post("/k6_motion_context/slot_exists")
    @require_same_origin
    async def _slot_exists_route(request):
        data = await request.json()
        exists = _clip_slot_exists(data.get("latent_path") or "k6_context",
                                   int(data.get("clip_index") or 1))
        return web.json_response({"exists": exists})

    @server.routes.post("/k6_motion_context/clear_latents")
    @require_same_origin
    async def _clear_latents_route(request):
        data = await request.json()
        n = _clear_clip_slots(data.get("latent_path") or "k6_context")
        return web.json_response({"removed": n})

    register_chain_routes._done = True


def _write_safetensors(path, tensors):
    # safetensors load_file memory-maps on Windows. Overwriting a mapped
    # slot (a re-roll, or Load aimed at the file Save is about to replace)
    # fails with os error 1224. Write a sibling temp file and replace.
    tmp = path + ".tmp"
    try:
        _st_save(tensors, tmp,
                 metadata={"format": "k6_motion_context_av_v1"})
        try:
            os.replace(tmp, path)
        except OSError:
            gc.collect()
            os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


class Kandinsky6MotionContextSaveLatent:
    """Save a K6 AV latent to disk so the NEXT run can load it.

    Wiring the sampler's output straight into context_latent is a cycle:
    the sampler would be consuming its own result. The latent that motion
    context needs is the PREVIOUS clip's, which lives in the previous run,
    so it has to cross runs through disk, the same way the frames and
    audio already do. Stock Save/Load Latent cannot serialize the nested
    video/audio pair; this saves the two streams side by side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The sampler's output latent (the same one you wire into the decode nodes)."}),
                "filename_prefix": ("STRING", {
                    "default": "k6_context/clip",
                    "tooltip": "Saved under the ComfyUI output folder. The default keeps all chain latents in one folder so the Load node can always pick the right slot."}),
                "clip_index": ("INT", {
                    "default": 1, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "Which clip of the chain THIS is. Saves to that clip's fixed slot, so a re-roll overwrites its own reject instead of stacking new files. First clip: 1 here and 0 on the Load node. Clip 2: 2 here and 1 on the Load node. 0 = a new numbered file every run (numbers count runs, not clips)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "model/kandinsky6"
    DESCRIPTION = ("Save the sampler's Kandinsky 6 AV latent so the next "
                   "run's Motion Context node can pin from it via the "
                   "matching Load node.")

    def save(self, latent, filename_prefix, clip_index=0):
        if _st_save is None:
            raise RuntimeError("k6_motion_context: safetensors is not "
                               "available; cannot save latents.")
        video, audio = _joint_streams(latent)
        video = video.cpu().contiguous()
        audio = audio.cpu().contiguous()
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        if int(clip_index) > 0:
            # fixed slot with the natural name: clip 2 -> *_00002. A
            # re-roll of this clip overwrites its own save, so rejects
            # never accumulate or get loaded later.
            path = os.path.join(folder, "%s_%05d.safetensors"
                                % (filename, int(clip_index)))
        else:
            path = os.path.join(folder, "%s_%05d_.safetensors"
                                % (filename, counter))
        _write_safetensors(path, {"video": video, "audio": audio})
        _LOG.info("k6_motion_context: saved AV latent to %s (video %s, "
                  "audio %s)", path, tuple(video.shape), tuple(audio.shape))
        # report the file like stock Save Latent's ui.latents, so an HTTP
        # client can find it in /history and fetch it through /view
        subfolder = os.path.relpath(os.path.dirname(path),
                                    folder_paths.get_output_directory())
        subfolder = "" if subfolder == "." else subfolder.replace(os.sep, "/")
        return {"ui": {"latents": [{"filename": os.path.basename(path),
                                    "subfolder": subfolder,
                                    "type": "output"}]},
                "result": (path,)}


class Kandinsky6MotionContextLoadLatent:
    """Load a saved K6 AV latent for the context_latent input.

    clip_index means exactly what it says: set it to the clip you want to
    CONTINUE FROM, and that clip's slot is loaded. Generating clip 2 from
    clip 1: Load node 1, Save node 2. Use K6 Motion Context Chain to
    advance the pair and queue: Approve walks forward, Run/Re-roll stays
    on the current slot (use it instead of ComfyUI's Run), Chain is
    Approve on a loop, Reset sets Load 0 / Save 1, Clear latents deletes
    numbered slots. All three nodes must sit in the same canvas group.

    At 0 there is no previous clip: the loader returns nothing and Motion
    Context passes the conditioning through. First clip of a chain is
    Load 0 / Save 1, then Load 1 / Save 2, and so on.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {
                    "default": "k6_context",
                    "tooltip": "A saved latent file, or a folder (relative paths resolve against the ComfyUI output directory). Pointing at a specific FILE always loads that file when clip_index is greater than 0, ignoring the index. clip_index 0 never reads a file."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "The clip to CONTINUE FROM: that clip's slot is loaded. First clip: 0 here and 1 on the Save node (nothing is loaded). Clip 2 from clip 1: 1 here and 2 on the Save node."}),
            },
            "optional": {
                # Optional and last on purpose. ComfyUI stores widget
                # values by position, so new widgets go at the end.
                "folder": (list(LATENT_FOLDERS), {
                    "default": "output",
                    "tooltip": "Where latent_path lives. output: a file or chain folder under the ComfyUI output directory (the normal chain). input: one file under the ComfyUI input directory, e.g. uploaded over HTTP by an external tool. clip_index 0 still loads nothing; above 0 it loads that file."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "model/kandinsky6"
    DESCRIPTION = ("Load a latent saved by K6 Motion Context Save Latent, as "
                   "a standard Kandinsky 6 joint latent.")

    @classmethod
    def IS_CHANGED(cls, latent_path, clip_index=0, folder="output"):
        # the path string stays constant while the file behind it changes
        # (an overwritten slot), so cache on the resolved file identity
        # instead, otherwise ComfyUI would happily serve a stale latent
        # forever. Index 0 never reads a file.
        if int(clip_index) <= 0:
            return "disabled"
        try:
            p = _resolve_latent_path(latent_path, clip_index, folder)
            return "%s:%d:owned" % (p, os.stat(p).st_mtime_ns)
        except Exception:
            return float("NaN")  # unresolvable: never cache

    def load(self, latent_path, clip_index=0, folder="output"):
        if int(clip_index) <= 0:
            return (None,)
        if _st_load is None:
            raise RuntimeError("k6_motion_context: safetensors is not "
                               "available; cannot load latents.")
        path = _resolve_latent_path(latent_path, clip_index, folder)
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "k6_motion_context: %s is not a k6_motion_context latent "
                "(missing video/audio streams). Was it saved by the stock "
                "Save Latent node instead?" % path)
        # copy off the mmap so a later Save can overwrite this slot on Windows
        video = data.pop("video").contiguous().clone()
        audio = data.pop("audio").contiguous().clone()
        _LOG.info("k6_motion_context: loaded AV latent from %s", path)
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio)),
                 "frame_rate": FPS,
                 "sample_rate": AUDIO_SAMPLE_RATE},)


class Kandinsky6MotionContextChain:
    """Approve, Run/Re-roll, auto-chain, reset indices, or clear slots.

    Load, Save, and this node must sit in the same canvas group or the
    buttons do nothing. Chain is Approve on a loop. At Load 0 / Save 1
    with no clip 1 on disk it generates that first clip instead of
    advancing into a missing file. segments > 0 stops Chain after that
    many clips; 0 keeps going until Stop. Reset only sets 0/1. Clear
    latents deletes numbered chain slots, not custom filenames.
    Execute is a no-op; the buttons and two HTTP routes do the work.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments": ("INT", {
                    "default": 0, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "How many clips Chain generates before it stops. 0 keeps going until you click Stop."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    OUTPUT_NODE = True
    CATEGORY = "model/kandinsky6"
    DESCRIPTION = ("Approve advances Load/Save and runs the next clip. "
                   "Run/Re-roll queues the current slot (use this instead "
                   "of ComfyUI's Run). Chain is Approve on a loop; at "
                   "Load 0 / Save 1 with no clip 1 saved it generates "
                   "clip 1 first. segments is how many clips Chain runs "
                   "before stopping (0 = until Stop). Reset sets Load 0 / "
                   "Save 1. Clear latents deletes numbered chain slots. "
                   "Load, Save, and this node must sit in the same canvas "
                   "group or the buttons do nothing.")

    def noop(self, segments=0):
        return ()


NODE_CLASS_MAPPINGS = {
    "Kandinsky6MotionContext": Kandinsky6MotionContext,
    "Kandinsky6MotionContextTrim": Kandinsky6MotionContextTrim,
    "Kandinsky6MotionContextSaveLatent": Kandinsky6MotionContextSaveLatent,
    "Kandinsky6MotionContextLoadLatent": Kandinsky6MotionContextLoadLatent,
    "Kandinsky6MotionContextChain": Kandinsky6MotionContextChain,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Kandinsky6MotionContext": "K6 Motion Context",
    "Kandinsky6MotionContextTrim": "K6 Motion Context Trim",
    "Kandinsky6MotionContextSaveLatent": "K6 Motion Context Save Latent",
    "Kandinsky6MotionContextLoadLatent": "K6 Motion Context Load Latent",
    "Kandinsky6MotionContextChain": "K6 Motion Context Chain",
}
