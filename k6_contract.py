"""Check that this ComfyUI still has the Kandinsky 6 support this pack builds on.

Nothing here modifies ComfyUI. The pack's arithmetic is written against the
K6 release contract (24 fps picture on a 4x causal latent, 44.1 kHz sound
on a 1024x time-first latent) and against two model behaviors that stock
ComfyUI does not exercise:

  multi-frame tail  the DiT accepts a k6_reference with more than one frame
                    and gives the tail its own run of RoPE positions, so a
                    window of previous-clip frames can be pinned
  audio mask        the PiFlow guider keeps the masked part of the audio
                    stream clean, which is how a custom track is pinned

Both are checked before the first render, and if an update breaks an
assumption the node refuses to run and says what moved. A loud failure
beats a bad join you do not notice.
"""

import inspect
import logging

_LOG = logging.getLogger("k6_motion_context")

_checked = None  # None not yet run, True passed, str the failure


def _check():
    from comfy.ldm.kandinsky6.core_contract import (
        GENERATION_DEFAULTS,
        LATENT_DEFAULTS,
    )

    # The window and trim arithmetic in nodes.py is written against these.
    expect = {
        "fps": (float(GENERATION_DEFAULTS["fps"]), 24.0),
        "video channels": (int(LATENT_DEFAULTS["video_channels"]), 16),
        "video temporal factor": (int(LATENT_DEFAULTS["video_temporal_compression_factor"]), 4),
        "audio channels": (int(LATENT_DEFAULTS["audio_channels"]), 40),
        "audio downsample factor": (int(LATENT_DEFAULTS["audio_downsample_factor"]), 1024),
        "audio sample rate": (int(LATENT_DEFAULTS["audio_sample_rate"]), 44100),
    }
    for name, (got, want) in expect.items():
        if got != want:
            raise RuntimeError(
                "the Kandinsky 6 contract moved: %s is %r, this pack needs %r. "
                "The window and trim arithmetic in nodes.py would place the "
                "join at the wrong instant." % (name, got, want))

    # The DiT must append a multi-frame reference tail: type ids and mask
    # marked over the whole tail, RoPE built from the marked frame count.
    import comfy.ldm.kandinsky6.model as model

    adapter_src = inspect.getsource(model.Kandinsky6NativeAVDiT._forward)
    if "-ref_frames:" not in adapter_src:
        raise RuntimeError(
            "the Kandinsky 6 DiT in this ComfyUI does not mark a multi-frame "
            "reference tail. Clip chaining needs the multi-frame I2VA tail; "
            "update ComfyUI.")
    rope_src = inspect.getsource(model.Kandinsky6._forward)
    if "ref_frames" not in rope_src:
        raise RuntimeError(
            "the Kandinsky 6 RoPE no longer sizes the reference tail from the "
            "marked frame count. A pinned window would land at the wrong "
            "positions; update ComfyUI.")

    # The PiFlow guider must drive the per-frame audio mask.
    import comfy_extras.nodes_kandinsky6 as nk6

    guider_src = inspect.getsource(nk6.Kandinsky6PiFlowGuider.inner_sample)
    if "audio_mask" not in guider_src:
        raise RuntimeError(
            "the Kandinsky 6 PiFlow guider in this ComfyUI no longer keeps a "
            "masked part of the audio stream clean, so a custom track or a "
            "pinned tail could not be pinned; update ComfyUI.")


def ensure_k6_contract():
    """Run the checks once per process; raise a loud error if they fail."""
    global _checked
    if _checked is True:
        return
    if _checked is not None:
        raise RuntimeError("k6_motion_context: " + _checked)
    try:
        _check()
    except Exception as e:
        _checked = str(e)
        raise
    _checked = True
    _LOG.info("k6_motion_context: K6 contract checks passed.")
