"""Kandinsky 6 Motion Context.

Clip chaining for Kandinsky 6: pin the tail of the previous clip, picture
and sound, so the next clip genuinely continues it, with an optional
custom speech track pinned clean on the audio stream for lip-sync.

This registers the nodes and nothing else. Nothing in ComfyUI is modified.
There is one dependency on behavior rather than structure: the K6 DiT must
accept a multi-frame k6_reference tail (its own run of RoPE positions,
marked by the visual token-type embeddings and the mask channel) and the
PiFlow guider must keep the masked part of the audio stream clean.
k6_contract.py checks both against the live code before the first render
and refuses if either ever stops holding, so an upstream change produces a
clear error rather than a silently wrong join.

Those checks run on first use, not at import, so having the pack installed
changes nothing until you actually chain a clip.

Requires the Kandinsky 6 ComfyUI support: the K6 nodes, the PiFlow guider
with the audio mask, and the multi-frame I2VA reference tail.
"""

import logging

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    register_chain_routes,
)

register_chain_routes()

logging.getLogger("k6_motion_context").info(
    "k6_motion_context: nodes registered. ComfyUI is not modified; the "
    "K6 contract checks run on the first use of a Motion Context node.")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
