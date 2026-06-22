import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingEEGNet(nn.Module):
    """
    Spiking EEGNet with all tunable hyperparameters exposed for genetic search.

    Fixed architectural constraints (not search parameters):
      - Temporal conv kernel:            (1, temporal_kernel_size)   — must not cross channels
      - Depthwise spatial conv kernel:   (num_channels, 1)           — must span all channels, time=1
      - Separable depthwise conv kernel: (1, separable_kernel_size)  — must not cross channels
      - Pointwise conv kernel:           (1, 1)                      — by definition

    Search spaces
    -------------
    temporal_filters  int  uniform [4, 32]
        low=4:  Fewer than 4 filters cannot represent the diversity of EEG frequency bands
                (delta/theta/alpha/beta/gamma = 5 bands minimum).
        high=32: Beyond 32, parameter count explodes through the depthwise layer
                 (32 * depth_multiplier filters) with diminishing returns on small EEG datasets
                 that typically have O(100-1000) trials.

    depth_multiplier  int  uniform [1, 4]
        low=1:  Must be at least 1 or the depthwise layer produces no output.
                D=1 means one spatial filter per temporal filter — minimal but valid.
        high=4: D=4 gives 4 spatial hypotheses per temporal filter. Beyond this,
                spatial filters start to be redundant given that EEG has a small
                number of meaningful source configurations.

    pointwise_filters  int  uniform [8, 64]
        low=8:  Fewer than 8 features going into the classifier loses too much
                information; this is already a heavy bottleneck on the upstream
                temporal_filters * depth_multiplier features.
        high=64: Beyond 64, the Linear classifier input grows large relative to
                 typical EEG trial counts, increasing overfitting risk sharply.

    temporal_kernel_div  int  uniform [2, 8]
        low=2:  div=2 → kernel = num_samples//2, the original EEGNet value.
                Going below 2 would make the kernel longer than half the signal,
                which causes excessive padding artifacts and near-full overlap
                between adjacent filter positions.
        high=8: div=8 → kernel = num_samples//8, e.g. 16 samples at 128 Hz = 125 ms.
                Shorter than ~100 ms captures only high-gamma / noise, which is
                rarely the informative band for standard BCI paradigms.

    separable_kernel_size  int  uniform [4, 32]
        low=4:  Fewer than 4 samples (~31 ms at 128 Hz after 4x pool) is too short
                to capture even one cycle of beta-band oscillations (~13 Hz),
                which are a primary BCI feature.
        high=32: 32 samples after pool1 corresponds to ~250 ms of original signal
                 (at 128 Hz with pool1=4). Longer than ~250 ms in this compressed
                 representation starts to re-learn what the temporal conv already
                 captured, adding redundancy rather than complementary features.

    pool1_size  int  uniform [2, 8]
        low=2:  Minimum meaningful downsampling; preserves most temporal resolution
                for the separable conv but provides at least some translation
                invariance and compute reduction.
        high=8: 8x downsampling after Block 1 leaves num_samples//8 time steps.
                For a 128-sample input this gives 16 steps — still workable.
                Beyond 8x the representation becomes too coarse for the separable
                conv to find structure in.

    pool2_size  int  uniform [2, 8]
        low=2:  Same reasoning as pool1 — minimum useful downsampling.
        high=8: Combined with pool1, total downsampling is pool1*pool2. The
                assertion time_after_pool2 >= 1 catches the hard failure, but
                practically pool1=8 + pool2=8 = 64x reduction leaves only 2
                time steps for 128-sample input, which is marginal. The search
                will naturally avoid this via fitness pressure.

    dropout  float  uniform [0.1, 0.75]
        low=0.1: Below 0.1 dropout provides negligible regularisation — effectively
                 equivalent to no dropout, removing the parameter from the search.
        high=0.75: Above 0.75, too many activations are zeroed per forward pass;
                   the sparse binary spike signals from LIF neurons are already
                   low-density, so heavy dropout compounds sparsity and stalls
                   learning.

    beta  float  uniform [0.5, 0.99]
        low=0.5: At beta=0.5 the membrane loses half its charge each time step —
                 very short memory (~1-2 steps). Going below 0.5 makes the LIF
                 neuron nearly memoryless, equivalent to a simple threshold unit
                 with no temporal integration benefit.
        high=0.99: At beta=0.99 the membrane decays by only 1% per step — very
                   long memory. Beyond 0.99 the potential accumulates without
                   decaying meaningfully, causing runaway activation or requiring
                   many steps to reset, which destabilises training.

    spike_grad_slope  float  uniform [5.0, 100.0]
        low=5.0: Below 5 the fast_sigmoid surrogate is so wide and flat that
                 gradients are near-zero everywhere, effectively blocking backprop
                 through spike events — equivalent to not training the spiking
                 layers at all.
        high=100.0: Above 100 the surrogate approaches the true step function; the
                    gradient is nonzero only in an extremely narrow band around the
                    threshold, causing near-zero gradients almost everywhere and
                    making training as difficult as without a surrogate.
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        num_samples: int,
        # Filter counts
        temporal_filters: int = 8,           # uniform int [4, 32]
        depth_multiplier: int = 2,           # uniform int [1, 4]
        pointwise_filters: int = 16,         # uniform int [8, 64]
        # Filter sizes
        temporal_kernel_div: int = 2,        # uniform int [2, 8] → kernel = num_samples // div
        separable_kernel_size: int = 16,     # uniform int [4, 32]
        # Pooling
        pool1_size: int = 4,                 # uniform int [2, 8]
        pool2_size: int = 4,                 # uniform int [2, 8]
        # Regularisation
        dropout: float = 0.5,               # uniform float [0.1, 0.75]
        # SNN dynamics
        beta: float = 0.95,                 # uniform float [0.5, 0.99]
        spike_grad_slope: float = 25.0,     # uniform float [5.0, 100.0]
    ):
        super().__init__()
        self.num_classes = num_classes

        # --- Validate and derive temporal kernel size ---
        temporal_kernel_div = max(2, min(temporal_kernel_div, 8))
        temporal_kernel_size = num_samples // temporal_kernel_div
        # Kernel must be at least 1; padding = kernel // 2 for near-same output size
        assert temporal_kernel_size >= 1, (
            f"temporal_kernel_size={temporal_kernel_size} is too small. "
            f"Increase num_samples or reduce temporal_kernel_div."
        )

        # --- Validate separable kernel fits after pool1 ---
        time_after_pool1 = num_samples // pool1_size
        assert separable_kernel_size <= time_after_pool1, (
            f"separable_kernel_size={separable_kernel_size} >= "
            f"time_after_pool1={time_after_pool1}. "
            f"Reduce separable_kernel_size or pool1_size."
        )

        # --- Validate pooling doesn't collapse the time dimension to zero ---
        time_after_pool2 = time_after_pool1 // pool2_size
        assert time_after_pool2 >= 1, (
            f"pool1_size={pool1_size} * pool2_size={pool2_size} = "
            f"{pool1_size * pool2_size} collapses time dimension to zero "
            f"for num_samples={num_samples}."
        )

        spike_grad = surrogate.fast_sigmoid(slope=spike_grad_slope)

        # Block 1 — temporal
        # kernel (1, temporal_kernel_size): height=1 so filter never crosses channels — fixed constraint
        # padding (0, temporal_kernel_size // 2) keeps time dimension ~ unchanged
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, temporal_filters,
                      kernel_size=(1, temporal_kernel_size),
                      padding=(0, temporal_kernel_size // 2), bias=False),
            nn.BatchNorm2d(temporal_filters),
        )
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Block 1 — depthwise spatial
        # kernel (num_channels, 1): height spans all channels, width=1 so never crosses time — fixed constraint
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(temporal_filters, temporal_filters * depth_multiplier,
                      kernel_size=(num_channels, 1),
                      groups=temporal_filters, bias=False),
            nn.BatchNorm2d(temporal_filters * depth_multiplier),
        )
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1_size))
        self.drop1 = nn.Dropout(dropout)

        # Block 2 — separable
        # kernel (1, separable_kernel_size): height=1 so filter never crosses channels — fixed constraint
        self.separable_depthwise = nn.Conv2d(
            temporal_filters * depth_multiplier,
            temporal_filters * depth_multiplier,
            kernel_size=(1, separable_kernel_size),
            padding=(0, separable_kernel_size // 2),
            groups=temporal_filters * depth_multiplier, bias=False,
        )
        # kernel (1, 1): pointwise by definition — fixed constraint
        self.separable_pointwise = nn.Sequential(
            nn.Conv2d(temporal_filters * depth_multiplier, pointwise_filters,
                      kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(pointwise_filters),
        )
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2_size))
        self.drop2 = nn.Dropout(dropout)

        flat_size = self._get_flat_size(num_channels, num_samples)
        self.classifier = nn.Linear(flat_size, num_classes, bias=True)

    def _get_flat_size(self, num_channels, num_samples):
        with torch.no_grad():
            x = torch.zeros(1, 1, num_channels, num_samples)
            x = self.temporal_conv(x)
            x = self.depthwise_conv(x)
            x = self.pool1(x)
            x = self.separable_depthwise(x)
            x = self.separable_pointwise(x)
            x = self.pool2(x)
            return x.flatten(1).shape[1]

    def get_mem_flat_size(self):
        """Flat size of mem3 after pool2, used for membrane-readout classifier sizing."""
        with torch.no_grad():
            # mem3 has same spatial shape as the conv output before pool2
            # We can derive it from the classifier input size since both
            # go through the same pool2 → flatten path.
            # The classifier is already built on this size, so just return
            # classifier.in_features.
            return self.classifier.in_features

    def forward(self, x: torch.Tensor, num_steps: int = 1):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_out_steps = []
        mem_out_steps = []

        for _ in range(num_steps):
            cur = self.temporal_conv(x)
            spk1, mem1 = self.lif1(cur, mem1)

            cur = self.depthwise_conv(spk1)
            spk2, mem2 = self.lif2(cur, mem2)

            cur = self.pool1(spk2)
            cur = self.drop1(cur)

            cur = self.separable_depthwise(cur)
            cur = self.separable_pointwise(cur)
            spk3, mem3 = self.lif3(cur, mem3)

            cur = self.pool2(spk3)
            cur = self.drop2(cur)

            out = self.classifier(cur.flatten(1))
            spk_out_steps.append(out)
            mem_out_steps.append(mem3)

        # spk: (num_steps, batch, num_classes)
        # mem: (num_steps, batch, pointwise_filters, 1, time_after_pool2)
        return torch.stack(spk_out_steps), torch.stack(mem_out_steps)