# quantization/quantizer.py
"""
Core fixed-point quantizer.

Implements true Q<integer_bits>.<fractional_bits> arithmetic that matches
how values would be stored and computed on an FPGA DSP slice:

  1. Scale the real-valued tensor into the integer domain.
  2. Round to the nearest representable integer (configurable rounding).
  3. Clamp / wrap to the representable range (configurable overflow mode).
  4. Scale back to real values for continued floating-point bookkeeping.

The tensor returned by ``quantize`` is a float32 tensor whose values are
exactly representable in the chosen fixed-point format — i.e. every value
is an integer multiple of ``2**(-frac_bits)``.

Assumptions
-----------
* All tensors are on whatever device they were already on (CPU or CUDA).
  The quantizer is device-agnostic.
* float32 precision is sufficient for bookkeeping up to ~Q8.64 (log2 of
  float32 mantissa ≈ 24 bits; headroom beyond that degrades gracefully).
* SNN spike tensors (binary 0/1) pass through quantization unchanged
  because 0 and 1 are always exactly representable.
* BatchNorm running statistics (mean, variance, weight, bias) are quantized
  as ordinary parameters — no special treatment needed.
"""

from __future__ import annotations
import math
import torch
from .config import QuantConfig


class FixedPointQuantizer:
    """
    Stateless fixed-point quantizer for a single ``(integer_bits, frac_bits)``
    format.

    Parameters
    ----------
    integer_bits : int
        Number of bits used for the integer part (including sign bit when
        ``cfg.signed=True``).
    frac_bits : int
        Number of fractional bits (bits after the binary point).
    cfg : QuantConfig
        Sweep-level configuration (signed, overflow mode, rounding).
    """

    def __init__(self, integer_bits: int, frac_bits: int, cfg: QuantConfig) -> None:
        self.integer_bits = integer_bits
        self.frac_bits = frac_bits
        self.cfg = cfg

        # Compute representable range -------------------------------------------
        total_bits = integer_bits + frac_bits
        if cfg.signed:
            self._q_min = float(-(2 ** (total_bits - 1)))
            self._q_max = float((2 ** (total_bits - 1)) - 1)
        else:
            self._q_min = 0.0
            self._q_max = float((2 ** total_bits) - 1)

        # Scale factor: multiply real value → integer domain
        self._scale: float = float(2.0 ** frac_bits)
        # Minimum representable step in real domain
        self._resolution: float = 2.0 ** (-frac_bits) if frac_bits > 0 else 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize ``x`` into the Q<integer_bits>.<frac_bits> format.

        Steps
        -----
        1. Scale into integer domain.
        2. Round (nearest / floor / stochastic).
        3. Clamp or wrap to representable range.
        4. Scale back to real domain.

        Parameters
        ----------
        x : torch.Tensor  (any shape, any device, float32)

        Returns
        -------
        torch.Tensor  same shape and device as ``x``, float32.
            Values are exact multiples of ``2**(-frac_bits)``.
        """
        # Step 1: scale to integer domain
        x_scaled = x * self._scale

        # Step 2: round
        x_rounded = self._round(x_scaled)

        # Step 3: overflow handling
        if self.cfg.overflow == "saturate":
            x_clamped = x_rounded.clamp(self._q_min, self._q_max)
        else:  # "wrap"
            span = self._q_max - self._q_min + 1
            x_clamped = ((x_rounded - self._q_min) % span) + self._q_min

        # Step 4: scale back to real domain
        return x_clamped / self._scale

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _round(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured rounding mode."""
        if self.cfg.rounding == "nearest":
            return x.round()
        elif self.cfg.rounding == "floor":
            return x.floor()
        elif self.cfg.rounding == "stochastic":
            # Stochastic rounding: floor + Bernoulli(fractional_part)
            frac = x - x.floor()
            noise = torch.bernoulli(frac)
            return x.floor() + noise
        else:
            raise ValueError(f"Unknown rounding mode: {self.cfg.rounding!r}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sign = "S" if self.cfg.signed else "U"
        return (
            f"FixedPointQuantizer("
            f"{sign}Q{self.integer_bits}.{self.frac_bits}, "
            f"range=[{self._q_min}/{self._scale:.0f}, {self._q_max}/{self._scale:.0f}], "
            f"overflow={self.cfg.overflow!r}, "
            f"rounding={self.cfg.rounding!r})"
        )


# ---------------------------------------------------------------------------
# Integer-bit auto-selection
# ---------------------------------------------------------------------------

def determine_integer_bits(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    cfg: QuantConfig,
    device: torch.device,
    max_batches: int = 50,
) -> int:
    """
    Scan training data and model parameters to determine the minimum number of
    integer bits that will prevent overflow for any observed value.

    Algorithm
    ---------
    1. Collect the maximum absolute value seen in ``max_batches`` of input data.
    2. Collect the maximum absolute value across all model weights and biases.
    3. Compute ``ceil(log2(max_abs + 1))`` as the minimum integer width.
    4. Add ``cfg.safety_margin_bits`` for headroom.
    5. Add 1 for the sign bit when ``cfg.signed`` is True.

    Parameters
    ----------
    train_loader : DataLoader   Training data loader (used for input analysis only).
    model        : nn.Module    Trained model (parameters are analysed, not modified).
    cfg          : QuantConfig  Provides ``safety_margin_bits`` and ``signed``.
    device       : torch.device Where to move input tensors for analysis.
    max_batches  : int          Cap on how many batches to scan (speed/accuracy trade-off).

    Returns
    -------
    int  — Selected integer-bit width (including sign bit when signed=True).
    """
    max_input_abs = 0.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            max_input_abs = max(max_input_abs, xb.abs().max().item())

    # Scan model parameters (weights, biases, BN running stats)
    max_param_abs = 0.0
    for name, param in model.named_parameters():
        max_param_abs = max(max_param_abs, param.data.abs().max().item())
    for name, buf in model.named_buffers():
        # Buffers include BN running_mean, running_var, num_batches_tracked
        val = buf.abs().max().item()
        max_param_abs = max(max_param_abs, val)

    max_abs = max(max_input_abs, max_param_abs)

    # Minimum bits to represent max_abs without overflow
    if max_abs <= 0:
        min_bits = 1
    else:
        min_bits = math.ceil(math.log2(max_abs + 1))

    integer_bits = min_bits + cfg.safety_margin_bits
    if cfg.signed:
        integer_bits += 1  # sign bit

    print(f"\n[QuantAnalysis] Maximum observed value  : {max_abs:.6f}")
    print(f"[QuantAnalysis]   - max input abs       : {max_input_abs:.6f}")
    print(f"[QuantAnalysis]   - max param abs       : {max_param_abs:.6f}")
    print(f"[QuantAnalysis] Minimum integer bits    : {min_bits}")
    print(f"[QuantAnalysis] Safety margin           : {cfg.safety_margin_bits} bit(s)")
    print(f"[QuantAnalysis] Sign bit                : {'1 (signed)' if cfg.signed else '0 (unsigned)'}")
    print(f"[QuantAnalysis] Selected integer bits   : {integer_bits}")

    return integer_bits
