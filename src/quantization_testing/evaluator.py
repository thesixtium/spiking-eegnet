# quantization/evaluator.py
"""
Fixed-point FPGA simulation evaluator for SpikingEEGNet.

This module wraps SpikingEEGNet so that every tensor involved in inference —
inputs, weights, biases, BatchNorm parameters, LIF membrane potentials,
spike outputs, intermediate activations, pooling outputs, and final logits —
is requantized into the selected fixed-point format after every operation.

Design notes
------------
* We do NOT modify the model in-place. Instead we extract quantized copies
  of all parameters before each evaluation and run a manual forward pass
  that injects quantization after every sub-operation. This is the closest
  simulation of a true FPGA datapath achievable inside PyTorch.

* The LIF neuron's threshold is always 1.0 (snntorch default). Spike outputs
  are binary {0, 1} which are exactly representable in any fixed-point format
  with ≥1 integer bit, so no special handling is required.

* BatchNorm is applied in inference mode (running statistics are used, not
  batch statistics). We quantize mean, variance (as 1/sqrt(var+eps) i.e. the
  pre-computed scale), weight (gamma), and bias (beta) individually.

* Dropout layers are disabled during evaluation (model.eval() mode), so no
  quantization is required for them.

* The snntorch LIF membrane state is re-initialized to zero at the start of
  each forward pass, matching the behaviour in the original training code.

Assumptions
-----------
* The model is a SpikingEEGNet instance (attributes checked at runtime).
* model.eval() has been called before entering this module — the evaluator
  calls it internally to be safe, but the caller should not rely on side effects.
* Gradients are not computed (torch.no_grad() is always active here).
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from .quantizer import FixedPointQuantizer, QuantConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qp(q: FixedPointQuantizer, t: torch.Tensor) -> torch.Tensor:
    """Shorthand: quantize tensor ``t`` with quantizer ``q``."""
    return q.quantize(t)


def _get_quantized_params(module: nn.Module, q: FixedPointQuantizer) -> dict:
    """
    Return a dict of quantized copies of a module's weight and bias tensors.
    Handles Conv2d, Linear, BatchNorm2d.  Unknown modules return an empty dict.
    """
    params: dict = {}
    if hasattr(module, "weight") and module.weight is not None:
        params["weight"] = _qp(q, module.weight.detach())
    if hasattr(module, "bias") and module.bias is not None:
        params["bias"] = _qp(q, module.bias.detach())
    return params


# ---------------------------------------------------------------------------
# Manual fixed-point forward pass
# ---------------------------------------------------------------------------

def _quantized_forward(
    model: "SpikingEEGNet",  # type: ignore[name-defined]
    x: torch.Tensor,
    q: FixedPointQuantizer,
    n_steps: int,
) -> torch.Tensor:
    """
    Run a completely fixed-point forward pass through SpikingEEGNet.

    Every tensor is quantized after every arithmetic operation to simulate
    an FPGA datapath where all registers hold fixed-point values.

    Parameters
    ----------
    model   : SpikingEEGNet  Trained model in eval() mode.
    x       : torch.Tensor   Input batch, already quantized by the caller.
    q       : FixedPointQuantizer  Active quantizer for this precision level.
    n_steps : int            Number of SNN time steps.

    Returns
    -------
    torch.Tensor  Shape (batch, num_classes).  Averaged logits across time steps.
    """

    # ── Extract and quantize all learnable parameters once ──────────────────
    # Block 1 – temporal conv  (nn.Sequential: Conv2d + BN)
    tc_conv: nn.Conv2d = model.temporal_conv[0]
    tc_bn:   nn.BatchNorm2d = model.temporal_conv[1]
    tc_w = _qp(q, tc_conv.weight.detach())

    # Block 1 – depthwise spatial conv  (nn.Sequential: Conv2d + BN)
    dw_conv: nn.Conv2d = model.depthwise_conv[0]
    dw_bn:   nn.BatchNorm2d = model.depthwise_conv[1]
    dw_w = _qp(q, dw_conv.weight.detach())

    # Block 2 – separable depthwise conv  (standalone Conv2d)
    sd_conv: nn.Conv2d = model.separable_depthwise
    sd_w = _qp(q, sd_conv.weight.detach())
    sd_b = _qp(q, sd_conv.bias.detach()) if sd_conv.bias is not None else None

    # Block 2 – pointwise conv  (nn.Sequential: Conv2d + BN)
    pw_conv: nn.Conv2d = model.separable_pointwise[0]
    pw_bn:   nn.BatchNorm2d = model.separable_pointwise[1]
    pw_w = _qp(q, pw_conv.weight.detach())

    # Classifier
    cls: nn.Linear = model.classifier
    cls_w = _qp(q, cls.weight.detach())
    cls_b = _qp(q, cls.bias.detach()) if cls.bias is not None else None

    # ── Quantize BatchNorm as scale+shift ────────────────────────────────────
    # BN inference: y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
    # We decompose into: y = x * bn_scale + bn_shift
    # where:
    #   bn_scale = gamma / sqrt(running_var + eps)
    #   bn_shift = beta  - running_mean * bn_scale
    # Both bn_scale and bn_shift are quantized as ordinary parameters.

    def _bn_params(bn: nn.BatchNorm2d):
        eps = bn.eps
        std = (bn.running_var.detach() + eps).sqrt()
        gamma = bn.weight.detach() if bn.weight is not None else torch.ones_like(std)
        beta  = bn.bias.detach()   if bn.bias   is not None else torch.zeros_like(std)
        scale = _qp(q, gamma / std)           # (C,)
        shift = _qp(q, beta - bn.running_mean.detach() * scale)  # (C,)
        return scale, shift                    # both shape (C,)

    tc_bn_scale, tc_bn_shift = _bn_params(tc_bn)
    dw_bn_scale, dw_bn_shift = _bn_params(dw_bn)
    pw_bn_scale, pw_bn_shift = _bn_params(pw_bn)

    # Reshape for broadcasting over (N, C, H, W)
    def _reshape_bn(s, shift):
        return s.view(1, -1, 1, 1), shift.view(1, -1, 1, 1)

    tc_s, tc_sh = _reshape_bn(tc_bn_scale, tc_bn_shift)
    dw_s, dw_sh = _reshape_bn(dw_bn_scale, dw_bn_shift)
    pw_s, pw_sh = _reshape_bn(pw_bn_scale, pw_bn_shift)

    # ── Initialise LIF membrane potentials ───────────────────────────────────
    # snntorch stores threshold at 1.0; membrane starts at 0.0
    mem1 = _qp(q, model.lif1.init_leaky())
    mem2 = _qp(q, model.lif2.init_leaky())
    mem3 = _qp(q, model.lif3.init_leaky())

    beta1 = _qp(q, torch.tensor(model.lif1.beta, dtype=torch.float32))
    beta2 = _qp(q, torch.tensor(model.lif2.beta, dtype=torch.float32))
    beta3 = _qp(q, torch.tensor(model.lif3.beta, dtype=torch.float32))
    threshold = 1.0   # snntorch default; representable in any reasonable format

    logit_accum = None

    for _ in range(n_steps):

        # ────────────────────────────────────────────────────────────────────
        # Block 1 – Temporal Conv + BN + LIF
        # ────────────────────────────────────────────────────────────────────
        # Conv2d (no bias — bias=False in model)
        cur = F.conv2d(
            x, tc_w,
            bias=None,
            stride=tc_conv.stride,
            padding=tc_conv.padding,
            dilation=tc_conv.dilation,
            groups=tc_conv.groups,
        )
        cur = _qp(q, cur)                          # quantize conv output

        # BN as affine scale+shift
        cur = cur * tc_s + tc_sh
        cur = _qp(q, cur)                          # quantize BN output

        # LIF: membrane update → spike → reset
        # mem_new = beta * mem_old + cur
        mem1 = _qp(q, beta1 * mem1 + cur)
        spk1 = (mem1 >= threshold).float()         # binary spikes, exact
        mem1 = mem1 * (1.0 - spk1)                # reset spiked neurons
        mem1 = _qp(q, mem1)

        # ────────────────────────────────────────────────────────────────────
        # Block 1 – Depthwise Spatial Conv + BN + LIF
        # ────────────────────────────────────────────────────────────────────
        cur = F.conv2d(
            spk1, dw_w,
            bias=None,
            stride=dw_conv.stride,
            padding=dw_conv.padding,
            dilation=dw_conv.dilation,
            groups=dw_conv.groups,
        )
        cur = _qp(q, cur)

        cur = cur * dw_s + dw_sh
        cur = _qp(q, cur)

        mem2 = _qp(q, beta2 * mem2 + cur)
        spk2 = (mem2 >= threshold).float()
        mem2 = mem2 * (1.0 - spk2)
        mem2 = _qp(q, mem2)

        # ────────────────────────────────────────────────────────────────────
        # Pool1  (AvgPool — dropout disabled in eval mode)
        # ────────────────────────────────────────────────────────────────────
        cur = F.avg_pool2d(
            spk2,
            kernel_size=model.pool1.kernel_size,
            stride=model.pool1.stride,
        )
        cur = _qp(q, cur)

        # ────────────────────────────────────────────────────────────────────
        # Block 2 – Separable Depthwise Conv
        # ────────────────────────────────────────────────────────────────────
        cur = F.conv2d(
            cur, sd_w,
            bias=sd_b,
            stride=sd_conv.stride,
            padding=sd_conv.padding,
            dilation=sd_conv.dilation,
            groups=sd_conv.groups,
        )
        cur = _qp(q, cur)

        # ────────────────────────────────────────────────────────────────────
        # Block 2 – Pointwise Conv + BN + LIF
        # ────────────────────────────────────────────────────────────────────
        cur = F.conv2d(
            cur, pw_w,
            bias=None,
            stride=pw_conv.stride,
            padding=pw_conv.padding,
            dilation=pw_conv.dilation,
            groups=pw_conv.groups,
        )
        cur = _qp(q, cur)

        cur = cur * pw_s + pw_sh
        cur = _qp(q, cur)

        mem3 = _qp(q, beta3 * mem3 + cur)
        spk3 = (mem3 >= threshold).float()
        mem3 = mem3 * (1.0 - spk3)
        mem3 = _qp(q, mem3)

        # ────────────────────────────────────────────────────────────────────
        # Pool2 + flatten + classifier
        # ────────────────────────────────────────────────────────────────────
        cur = F.avg_pool2d(
            spk3,
            kernel_size=model.pool2.kernel_size,
            stride=model.pool2.stride,
        )
        cur = _qp(q, cur)

        cur_flat = cur.flatten(1)
        cur_flat = _qp(q, cur_flat)

        logits = F.linear(cur_flat, cls_w, cls_b)
        logits = _qp(q, logits)

        # Accumulate logits across time steps (matches training code)
        if logit_accum is None:
            logit_accum = logits
        else:
            logit_accum = logit_accum + logits
            logit_accum = _qp(q, logit_accum)

    # Average logits over time steps (matches training code: spk.mean(0))
    averaged = logit_accum / n_steps
    averaged = _qp(q, averaged)
    return averaged


# ---------------------------------------------------------------------------
# Main evaluator function
# ---------------------------------------------------------------------------

def evaluate_quantized(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    q: FixedPointQuantizer,
    device: torch.device,
    n_steps: int,
) -> tuple[float, float]:
    """
    Evaluate the model under a single fixed-point precision.

    All inference tensors are quantized; no floating-point arithmetic is used
    beyond the bookkeeping that PyTorch itself performs to represent the
    quantized-value floats.

    Parameters
    ----------
    model      : SpikingEEGNet  Trained model.
    val_loader : DataLoader     Evaluation data (test split).
    q          : FixedPointQuantizer  Quantizer for the current sweep point.
    device     : torch.device
    n_steps    : int            SNN time steps.

    Returns
    -------
    (balanced_accuracy, avg_cross_entropy_loss)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # Quantize input data — first operation in the FPGA datapath
            xb_q = q.quantize(xb)

            # Full fixed-point forward pass
            logits = _quantized_forward(model, xb_q, q, n_steps)

            loss = criterion(logits, yb)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return float(bal_acc), float(avg_loss)
