# SpikingEEGNet Fixed-Point Quantization — Complete Reference

## Table of Contents
1. [What this package does](#1-what-this-package-does)
2. [Fixed-point number format primer](#2-fixed-point-number-format-primer)
3. [What is and is not a "weight"](#3-what-is-and-is-not-a-weight)
4. [Exactly what gets quantized (and when)](#4-exactly-what-gets-quantized-and-when)
5. [What does NOT get quantized (and why)](#5-what-does-not-get-quantized-and-why)
6. [The integer-bit selection algorithm](#6-the-integer-bit-selection-algorithm)
7. [The quantization sweep](#7-the-quantization-sweep)
8. [BatchNorm: the special case](#8-batchnorm-the-special-case)
9. [LIF neurons: what is state vs. what is a weight](#9-lif-neurons-what-is-state-vs-what-is-a-weight)
10. [Outputs and how to read them](#10-outputs-and-how-to-read-them)
11. [Common failure modes and fixes](#11-common-failure-modes-and-fixes)
12. [File map](#12-file-map)

---

## 1. What this package does

This package simulates what would happen if you deployed SpikingEEGNet on an FPGA where all
arithmetic is done in **fixed-point** rather than float32. The goal is to find the minimum
number of fractional bits you can get away with before accuracy degrades meaningfully —
that number directly determines the word length of your FPGA datapath.

The simulation works by running a fully manual forward pass in PyTorch where **every tensor
is snapped to the nearest representable fixed-point value after every single operation**.
No floating-point accumulation is allowed to persist between steps. This is the closest you
can get to a true FPGA simulation without writing RTL.

---

## 2. Fixed-point number format primer

A fixed-point number in `Q<I>.<F>` format has:
- `I` integer bits (includes the sign bit when signed)
- `F` fractional bits

The resolution (smallest representable step) is `2^(-F)`.
The representable range for signed `Q<I>.<F>` is:

```
[-2^(I+F-1) / 2^F,  (2^(I+F-1) - 1) / 2^F]
```

**Example:** `Q8.8` (16-bit total, signed)
- Resolution: `1/256 ≈ 0.0039`
- Range: `[-128.0, 127.996]`

**Example:** `Q4.12` (16-bit total, signed)
- Resolution: `1/4096 ≈ 0.000244`
- Range: `[-8.0, 7.9998]`

In the sweep, `I` (integer bits) is fixed for the entire run — it is computed once from
the observed data range. `F` (fractional bits) sweeps from `frac_bits_start` down to
`frac_bits_end`, trading precision for hardware cost at each step.

The implementation in `quantizer.py` does this in four steps:
1. Multiply by `2^F` → move into integer domain
2. Round (nearest, floor, or stochastic)
3. Clamp to `[q_min, q_max]` (saturate) or wrap modularly (wrap)
4. Divide by `2^F` → back to real domain as a float32 that holds an exact fixed-point value

---

## 3. What is and is not a "weight"

This is the most important thing to understand for the integer-bit analysis.

### PyTorch's two storage types

**Parameters** (`model.named_parameters()`):  
Things that are updated by the optimizer during training. These are always "weights" in
the traditional sense.

| Tensor | What it is |
|--------|-----------|
| `temporal_conv.0.weight` | Conv2d kernel, shape `(F, 1, 1, K)` |
| `depthwise_conv.0.weight` | Depthwise conv kernel, shape `(F*D, 1, C, 1)` |
| `separable_depthwise.weight` | Separable depthwise kernel |
| `separable_pointwise.0.weight` | Pointwise conv kernel |
| `depthwise_conv.1.weight` | BatchNorm **gamma** (scale), shape `(C,)` |
| `depthwise_conv.1.bias` | BatchNorm **beta** (shift), shape `(C,)` |
| `temporal_conv.1.weight` | BatchNorm gamma |
| `temporal_conv.1.bias` | BatchNorm beta |
| `separable_pointwise.1.weight` | BatchNorm gamma |
| `separable_pointwise.1.bias` | BatchNorm beta |
| `classifier.weight` | Linear layer weight, shape `(n_classes, features)` |
| `classifier.bias` | Linear layer bias |

**Buffers** (`model.named_buffers()`):  
Tensors that live on the model but are NOT updated by the optimizer. Some are real
inference-time values; others are training scaffolding or runtime state that doesn't
belong in a fixed-point range analysis.

| Tensor | What it is | Quantized? | Included in int-bit scan? |
|--------|-----------|-----------|--------------------------|
| `temporal_conv.1.running_mean` | BN running mean (inference) | Yes (via `_bn_params`) | ✅ Yes |
| `temporal_conv.1.running_var` | BN running variance (inference) | **No** — converted to `1/sqrt(var+eps)` first | ❌ No — raw value misleading |
| `depthwise_conv.1.running_mean` | Same as above | Yes (via `_bn_params`) | ✅ Yes |
| `depthwise_conv.1.running_var` | Same | No | ❌ No |
| `*.num_batches_tracked` | Training counter, scalar int | No | ❌ No |
| `lif1.mem` | **LIF membrane potential** — runtime state | Yes, re-initialised to 0 each forward pass | ❌ No — see §9 |
| `lif2.mem` | Same | Yes | ❌ No |
| `lif3.mem` | Same | Yes | ❌ No |
| `lif*.threshold` | Spike threshold, always 1.0 | Hardcoded in evaluator | ❌ No — always 1.0 |
| `lif*.beta` | Leak factor (≤1.0) | Yes, as scalar | ❌ No — always ≤1.0 |
| `lif*.graded_spikes_factor` | Always 1.0 | No | ❌ No |

---

## 4. Exactly what gets quantized (and when)

The evaluator (`evaluator.py`) implements a complete fixed-point datapath. Here is the
order of operations and what gets quantized at each step.

### Before the time-step loop (parameter extraction)

All learnable parameters are quantized **once** and cached:

```
tc_w  = quantize(temporal_conv[0].weight)
dw_w  = quantize(depthwise_conv[0].weight)
sd_w  = quantize(separable_depthwise.weight)
sd_b  = quantize(separable_depthwise.bias)          # if not None
pw_w  = quantize(separable_pointwise[0].weight)
cls_w = quantize(classifier.weight)
cls_b = quantize(classifier.bias)
```

BatchNorm is **fused** into a scale+shift pair (see §8):

```
bn_scale = quantize(gamma / sqrt(running_var + eps))
bn_shift = quantize(beta - running_mean * bn_scale)
```

LIF decay factors:

```
beta1 = quantize(tensor(lif1.beta))
beta2 = quantize(tensor(lif2.beta))
beta3 = quantize(tensor(lif3.beta))
```

LIF membrane state is initialised and immediately quantized:

```
mem1 = quantize(lif1.init_leaky())   # → quantized 0.0
mem2 = quantize(lif2.init_leaky())
mem3 = quantize(lif3.init_leaky())
```

### Input quantization

```
xb_q = quantize(xb)    # first thing in the FPGA datapath
```

### Inside each time step

Every intermediate result is quantized after every operation:

```
# Block 1 — Temporal Conv
cur = conv2d(xb_q, tc_w)       → quantize
cur = cur * tc_scale + tc_shift → quantize   (BN)
mem1 = quantize(beta1 * mem1 + cur)
spk1 = (mem1 >= 1.0).float()               (exact: binary)
mem1 = quantize(mem1 * (1 - spk1))         (reset)

# Block 1 — Depthwise Spatial Conv
cur = conv2d(spk1, dw_w)       → quantize
cur = cur * dw_scale + dw_shift → quantize
mem2 = quantize(beta2 * mem2 + cur)
spk2 = (mem2 >= 1.0).float()
mem2 = quantize(mem2 * (1 - spk2))

# Pool 1
cur = avg_pool2d(spk2)         → quantize

# Block 2 — Separable Depthwise
cur = conv2d(cur, sd_w, sd_b)  → quantize

# Block 2 — Pointwise Conv
cur = conv2d(cur, pw_w)        → quantize
cur = cur * pw_scale + pw_shift → quantize
mem3 = quantize(beta3 * mem3 + cur)
spk3 = (mem3 >= 1.0).float()
mem3 = quantize(mem3 * (1 - spk3))

# Pool 2 + classifier
cur = avg_pool2d(spk3)         → quantize
cur_flat = flatten(cur)        → quantize
logits = linear(cur_flat, cls_w, cls_b) → quantize

# Logit accumulation
logit_accum = logit_accum + logits  → quantize
```

After all time steps: `averaged = logit_accum / n_steps → quantize`

---

## 5. What does NOT get quantized (and why)

| Thing | Why not quantized |
|-------|------------------|
| Spike outputs `spk1/2/3` | Binary {0, 1} — exactly representable in any format with ≥1 integer bit. Quantizing them would change nothing. |
| The threshold comparison `mem >= 1.0` | The threshold is a hardcoded float constant in the evaluator, not a stored tensor. On real FPGA hardware this becomes a single comparator. |
| The logit averaging `/ n_steps` | Dividing by a power-of-two constant is a bit-shift on FPGA, not a multiply-accumulate. The result is quantized afterward. |
| `num_batches_tracked` | Training counter, not used at inference time at all. |
| `running_var` raw value | Converted to `bn_scale` before use; the raw value is never in the datapath. |
| LIF `membrane` buffer at scan time | The value in this buffer after training is leftover state from the last training batch. It gets zeroed at the start of every forward pass and is therefore irrelevant to range analysis. |

---

## 6. The integer-bit selection algorithm

`determine_integer_bits()` in `quantizer.py` figures out how many integer bits `I` are
needed so that no observed value causes overflow.

### Algorithm

1. Scan `max_batches` of training data → `max_input_abs`
2. Scan all **parameters** → `max_param_abs`
3. Scan **included buffers** (running_mean only; see §3) → update `max_param_abs`
4. `max_abs = max(max_input_abs, max_param_abs)`
5. `min_bits = ceil(log2(max_abs + 1))`
6. `integer_bits = min_bits + safety_margin_bits + (1 if signed)`

### Why the +1 for sign?

In two's complement, a signed `N`-bit integer represents `[-2^(N-1), 2^(N-1)-1]`.
The sign bit "costs" one bit of positive range. If you need to represent +60 you need
`ceil(log2(61)) = 6` magnitude bits plus 1 sign bit = 7 bits total.

### Excluded buffers and their effect

The original code scanned everything in `named_buffers()`, including the LIF membrane
tensors (`lif*.mem`). After training, those membranes can hold values up to ~60 (as
seen in the outlier report). This caused:

```
ceil(log2(60.88)) = 6  →  + 1 safety + 1 sign = 8 bits
```

But if the membrane scan happened to catch a larger value (or if `running_var` was
large), the count inflated to 18. The fix is to exclude non-weight buffers from the
scan entirely. With only real weights included, the largest value is typically
the BN gamma (~1.13), giving:

```
ceil(log2(2.13)) = 2  →  + 1 safety + 1 sign = 4 integer bits
```

---

## 7. The quantization sweep

`sweep.py` orchestrates everything:

```
Step 0: analyse_model()         → model_analysis/ directory
Step 1: determine_integer_bits() → fixes I for the whole sweep
Step 2: float32 baseline        → reference accuracy and loss
Step 3: for F in [frac_bits_start .. frac_bits_end]:
            build FixedPointQuantizer(I, F)
            run evaluate_quantized()
            record (F, acc, loss, time)
Step 4: write quantization_results.csv + float_baseline.csv
Step 5: generate_all_plots()
```

The sweep goes **high F to low F** (most precise to least precise). This means the
first rows of your CSV should be near float32 accuracy, degrading as you go down.
The "cliff" — where accuracy drops sharply — is the minimum viable F for deployment.

---

## 8. BatchNorm: the special case

Standard BatchNorm inference is:

```
y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
```

An FPGA cannot efficiently compute `sqrt` or division at runtime in a DSP slice.
The evaluator pre-fuses this into a single affine transform:

```
bn_scale = gamma / sqrt(running_var + eps)      # computed once, in float
bn_shift = beta - running_mean * bn_scale        # computed once, in float
y = x * bn_scale + bn_shift                      # pure multiply-add, FPGA-friendly
```

Both `bn_scale` and `bn_shift` are then **quantized** before being stored.  
The raw `running_var` is **never quantized** and is **never in the datapath** —  
it only exists to compute `bn_scale`.

This is why `running_var` is excluded from the integer-bit scan: its raw value  
(e.g. 2.33 for `temporal_conv.1`) is irrelevant. What matters is  
`1 / sqrt(2.33 + 1e-5) ≈ 0.655`, which is well within a small integer range.

---

## 9. LIF neurons: what is state vs. what is a weight

A Leaky Integrate-and-Fire neuron has two distinct categories of tensors:

### Learned parameters (quantized, included in int-bit scan)
None for standard snntorch LIF — `beta` is set at construction, not learned by default.

### Scalar constants (quantized in evaluator, excluded from int-bit scan)
| Name | Value | Why excluded from scan |
|------|-------|----------------------|
| `beta` | ≤1.0 (e.g. 0.95) | By definition ≤1.0, can never drive integer-bit requirement |
| `threshold` | 1.0 always | Hardcoded constant in evaluator, not a stored weight for quantization purposes |
| `graded_spikes_factor` | 1.0 always | Only relevant for graded-spike mode (not used here) |

### Runtime state (quantized in evaluator, EXCLUDED from int-bit scan)
| Name | What it is | Why excluded |
|------|-----------|-------------|
| `mem` | Membrane potential accumulator | Reinitialised to `quantize(0.0)` at the start of every forward pass via `init_leaky()`. The value stored in this buffer after training is leftover state from the last training batch — it is **not representative** of inference-time values and should not influence the fixed-point format choice. |

The membrane **during inference** will stay bounded by the threshold (1.0) in a healthy
trained model because spikes reset it. If your membrane values are exploding during the
quantized forward pass, that is a sign of quantization noise accumulating across time
steps, not a signal to increase integer bits.

---

## 10. Outputs and how to read them

### `quantization/quantization_results.csv`
One row per fractional-bit value. Key columns:

| Column | What it means |
|--------|--------------|
| `fractional_bits` | F in Q<I>.<F> |
| `integer_bits` | I (fixed for the whole sweep) |
| `format` | Human-readable label, e.g. `Q4.12` |
| `balanced_accuracy` | Balanced accuracy on test subject (LOSO) |
| `loss` | Cross-entropy loss |
| `eval_time_s` | Wall-clock seconds for this F value |

### `quantization/float_baseline.csv`
Single-row reference. Compare every `balanced_accuracy` in the main CSV against this.

### `quantization/model_analysis/layer_weight_stats.csv`
One row per tensor (parameters + included buffers). Key columns:

| Column | What it means |
|--------|--------------|
| `abs_max` | Maximum absolute value — the number that drives integer-bit selection |
| `std` | Standard deviation — large std with small mean → many outliers possible |
| `outlier_fraction` | Fraction of values more than 3σ from mean |
| `q01` / `q99` | 1st and 99th percentile — shows where the bulk of values live vs. extremes |

### `quantization/model_analysis/weight_outlier_report.txt`
Plain-text diagnostic. The "Likely drivers" section at the bottom names the specific
tensors causing a large integer-bit requirement. Read this first when the sweep
produces an unexpectedly large `integer_bits`.

### Plots

| File | What to look for |
|------|-----------------|
| `accuracy_vs_fractional_bits.png` | The "cliff" — where accuracy drops sharply below the float baseline |
| `loss_vs_fractional_bits.png` | Loss usually rises more smoothly; useful for spotting gradual degradation |
| `accuracy_drop_vs_fractional_bits.png` | The minimum F where the curve crosses the 5pp threshold line is your deployment target |
| `weight_distributions.png` | Red histograms = layers with large weights. Bimodal or heavy-tailed = outlier risk |
| `weight_heatmaps.png` | Bright rows = specific output filters with large kernels. If only a few rows are bright, consider per-channel quantization |
| `parameter_magnitude_bar.png` | Instantly shows which single tensor is driving your integer-bit count |
| `outlier_fraction_bar.png` | Red bars (>5%) = layers where a per-layer quantizer would help more than a global one |

---

## 11. Common failure modes and fixes

### "My integer_bits is way too large"
Run the debug prints added to `determine_integer_bits`. Look for `[DEBUG buf]` lines
with large values. Common culprits:
- `lif*.mem` — leftover membrane state from training. Fixed by the buffer exclusion list.
- `running_var` — raw variance before the sqrt. Fixed by exclusion.
- A genuine outlier weight (check `parameter_magnitude_bar.png`).

### "Accuracy collapses even at F=32"
The integer bits are too small — values are saturating or wrapping. Increase
`safety_margin_bits` in `QuantConfig`, or check if a specific layer has weights
larger than what the current `integer_bits` can represent.

### "Accuracy is fine all the way to F=4 then suddenly collapses"
Normal — this is the cliff. The minimum viable F for your deployment is somewhere
in the `F=4..8` range. Check `accuracy_drop_vs_fractional_bits.png` for the exact
crossing of the 5pp line.

### "Accuracy never recovers to float baseline even at F=64"
The `_quantized_forward` function and the original `model.forward` are doing something
differently. Check that:
- `n_steps` matches `N_STEPS_EVAL` from training
- The BN fused scale+shift is mathematically equivalent to standard BN (verify with
  a single batch at high F)
- The logit accumulation and averaging matches the training code

### "BatchNorm scale/shift values are large"
If `temporal_conv.1.running_var` is very large (>10), your BN layers may not have
converged properly. The fused `bn_scale = gamma / sqrt(var + eps)` will be very small,
which is fine for quantization, but the corresponding `bn_shift` may be large.
Check the `layer_weight_stats.csv` for `bn_scale` and `bn_shift` values directly by
adding them to the scan in `_collect_tensor_stats`.

---

## 12. File map

```
src/
├── quantization_testing/
│   ├── __init__.py
│   ├── config.py          QuantConfig dataclass — all sweep knobs live here
│   ├── quantizer.py       FixedPointQuantizer + determine_integer_bits()
│   ├── evaluator.py       Manual fixed-point forward pass for SpikingEEGNet
│   ├── sweep.py           Orchestrator: analysis → baseline → sweep → plots
│   ├── plotting.py        Publication-quality matplotlib figures
│   ├── model_analysis.py  Architecture summary, weight stats, outlier report
│   └── run_quantization.py  CLI entry point (train or load → sweep)
│
results/
└── BNCI2014_001/
    ├── model.pt                          Saved float32 checkpoint
    └── quantization/
        ├── quantization_results.csv      Main sweep results (one row per F)
        ├── float_baseline.csv            Reference float32 accuracy + loss
        ├── accuracy_vs_fractional_bits.png
        ├── loss_vs_fractional_bits.png
        ├── accuracy_drop_vs_fractional_bits.png
        └── model_analysis/
            ├── model_architecture.txt    Full module tree with shapes
            ├── layer_weight_stats.csv    Per-tensor statistics
            ├── weight_outlier_report.txt Top-N outliers + BN table + culprits
            ├── weight_distributions.png  Per-layer histograms
            ├── weight_heatmaps.png       Conv2d |weight| heatmaps
            ├── parameter_magnitude_bar.png  Ranked |max| across all tensors
            ├── outlier_fraction_bar.png  Per-layer outlier fraction
            └── layer_stats_table.png     Stats table as image
```