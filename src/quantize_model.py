import copy

import torch
import torch.nn as nn


_VALID_BITS = {2, 4, 8, 16}


def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
    """
    Fake-quantize every parameter in the model to a signed INTn grid.

    Walks model.parameters() and rounds every weight, bias, and internal
    state tensor (including snn.Leaky beta and membrane parameters) to the
    nearest value representable in a signed `bits`-bit integer. Values remain
    stored as float32 at runtime — this simulates the precision loss of INTn
    storage without requiring hardware support for non-INT8 widths.

    Covers ALL layers unconditionally:
        Conv2d weights / biases
        BatchNorm scales / biases
        Linear weights / biases
        snn.Leaky beta, threshold, and any other registered parameters

    Parameters
    ----------
    model : trained nn.Module, on any device
    bits  : target bit width — one of {2, 4, 8, 16}
            must be a positive multiple of 2

    Returns
    -------
    A deep-copied model on CPU with all parameters fake-quantized.
    The original model is not modified.

    Raises
    ------
    ValueError : if bits is not in {2, 4, 8, 16}
    """
    if bits not in _VALID_BITS:
        raise ValueError(
            f"bits={bits} is not supported. Must be one of {sorted(_VALID_BITS)}."
        )

    q_max = 2 ** (bits - 1) - 1   # e.g. INT8 → 127
    q_min = -(2 ** (bits - 1))    # e.g. INT8 → -128

    q_model = copy.deepcopy(model).cpu().eval()

    with torch.no_grad():
        for param in q_model.parameters():
            abs_max = param.abs().max().clamp(min=1e-8)
            scale   = abs_max / q_max
            param.copy_(
                torch.clamp(torch.round(param / scale), q_min, q_max) * scale
            )

    return q_model