"""Window and trim arithmetic: latent frames, pixel frames, audio frames,
and the custom-track window starts."""

import k6mc.nodes as nodes

CLIP_SAMPLES = 218 * nodes.AUDIO_DOWNSAMPLE  # a 121-frame clip


def test_pixel_frames_cover_the_causal_grid():
    # first latent frame = 1 pixel, each later one = 4
    assert nodes._pixel_frames(1) == 1
    assert nodes._pixel_frames(6) == 21
    assert nodes._pixel_frames(10) == 37
    assert nodes._pixel_frames(15) == 57


def test_audio_frames_for_px_rounds_to_the_nearest_frame():
    # 21 px at 24 fps is 0.875 s = 37.78 audio frames -> 38
    assert nodes._audio_frames_for_px(21) == 38
    assert nodes._audio_frames_for_px(1) == 2


def test_custom_track_windows_advance_by_the_delivered_length():
    head = nodes._audio_frames_for_px(nodes._pixel_frames(6))
    assert head == 38

    advance = CLIP_SAMPLES - head * nodes.AUDIO_DOWNSAMPLE
    starts = [i * advance for i in range(3)]

    # the head of clip N+1 is the tail of clip N's window: no gap, no repeat
    for n in range(2):
        assert starts[n + 1] + head * nodes.AUDIO_DOWNSAMPLE == starts[n] + CLIP_SAMPLES

    # the first clip starts at the top of the track
    assert starts[0] == 0


def test_first_clip_window_starts_at_zero_with_no_head():
    # no pinned picture -> no head -> the window does not overlap backward
    head = 0
    start = 1 * (CLIP_SAMPLES - head * nodes.AUDIO_DOWNSAMPLE)
    assert start == CLIP_SAMPLES
