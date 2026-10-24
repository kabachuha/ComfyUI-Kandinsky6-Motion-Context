"""The contract checks pass against this ComfyUI and fail loudly when they
cannot verify an assumption."""

import k6mc.k6_contract as contract
import k6mc.nodes as nodes


def test_contract_passes_here():
    contract._checked = None
    contract.ensure_k6_contract()
    assert contract._checked is True


def test_apply_runs_the_checks():
    contract._checked = None
    import torch
    import comfy.nested_tensor

    latent = {
        "samples": comfy.nested_tensor.NestedTensor((
            torch.zeros(1, 16, 31, 60, 60),
            torch.zeros(1, 218, 40),
        )),
        "frame_rate": 24.0,
        "sample_rate": 44100,
    }
    nodes.Kandinsky6MotionContext().apply([[{}], [{}]], [[{}], [{}]], latent, 6)
    assert contract._checked is True
