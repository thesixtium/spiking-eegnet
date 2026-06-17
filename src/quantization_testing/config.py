# quantization/config.py
"""
Configuration dataclass for the fixed-point quantization sweep.

All parameters that govern the sweep are stored here so that callers
never need to thread individual knobs through multiple function signatures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class QuantConfig:
    """
    Parameters controlling the fixed-point sweep.

    Attributes
    ----------
    signed : bool
        If True, use two's-complement signed representation.
        If False, use unsigned (clamp min to 0). Default: True.

    overflow : Literal["saturate", "wrap"]
        Behaviour when a quantized value exceeds the representable range.
        "saturate"  → clip to [min_val, max_val]  (hardware register saturation)
        "wrap"      → modular wraparound             (hardware overflow behaviour)
        Default: "saturate"

    rounding : Literal["nearest", "floor", "stochastic"]
        Rounding mode applied before clamping.
        "nearest"     → round half-up (standard)
        "floor"       → truncation (towards -∞)
        "stochastic"  → probabilistic rounding (useful for training; not used here
                        but included for completeness)
        Default: "nearest"

    safety_margin_bits : int
        Extra integer bits added on top of the minimum required to prevent
        overflow on observed data/weight magnitudes.
        Default: 1

    frac_bits_start : int
        Highest number of fractional bits to evaluate (most precise).
        Default: 64

    frac_bits_end : int
        Lowest number of fractional bits to evaluate (least precise).
        Default: 0

    frac_bits_step : int
        Step size when iterating from frac_bits_start down to frac_bits_end.
        Default: 1  (evaluate every integer value)

    n_steps_eval : int
        Number of SNN time-steps used during quantized evaluation.
        Should match the value used during standard (float) evaluation.
        Default: 20
    """

    signed: bool = True
    overflow: Literal["saturate", "wrap"] = "saturate"
    rounding: Literal["nearest", "floor", "stochastic"] = "nearest"
    safety_margin_bits: int = 1
    frac_bits_start: int = 64
    frac_bits_end: int = 0
    frac_bits_step: int = 1
    n_steps_eval: int = 20

    @property
    def frac_bits_range(self) -> list[int]:
        """Return the list of fractional-bit values to sweep, high → low."""
        return list(range(self.frac_bits_start, self.frac_bits_end - 1, -self.frac_bits_step))
