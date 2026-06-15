import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
        X: np.ndarray,
        sfreq: float,
        flow: float,
        fhigh: float,
        n:int = 4
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to EEG trials.

    Parameters
    ----------
    X      : np.ndarray  (n_trials, 1, n_channels, n_samples)  float32
    sfreq  : float       Sampling frequency in Hz
    flow   : float       Low cutoff in Hz   — searchable, suggested range [1.0, 40.0]
    fhigh  : float       High cutoff in Hz  — searchable, suggested range [8.0, 120.0]

    Returns
    -------
    X_filt : np.ndarray  same shape as X, float32

    Notes
    -----
    flow must be < fhigh, and fhigh must be < sfreq/2 (Nyquist).
    These are checked with assertions so invalid filter configs fail fast
    during a genetic search rather than producing silent garbage.

    Search spaces (for genetic algorithm)
    --------------------------------------
    flow   float  uniform [1.0, 40.0]
        low=1.0:  Below 1 Hz the filter approaches DC; slow drifts dominate
                  and meaningful oscillatory content is swamped.
        high=40.0: flow=40 Hz would only pass high-gamma, which is rarely
                   informative alone and would need fhigh >> 40.
                   The constraint flow < fhigh - 2 (enforced below) prevents
                   degenerate zero-bandwidth configs.

    fhigh  float  uniform [44.0, 120.0]
        low=8.0:  Below 8 Hz the passband only covers delta/theta; that may
                  be intentional but 8 Hz is the minimum that still includes
                  alpha (8-12 Hz), which is the dominant BCI-relevant band.
        high=120.0: Capped at Nyquist - a few Hz margin (sfreq=250 → max 120).
                    Above ~100 Hz you are mostly capturing muscle artifact
                    rather than neural signal for standard BCI paradigms.

    # Search spaces — 256 Hz signal, all asserts guaranteed to pass for any combination
    #
    # flow   float  uniform [1.0, 40.0]
    #     low=1.0:   Captures slow cortical potentials / delta band floor.
    #                Above 0 so assert 1 always passes.
    #     high=40.0: Ensures flow_max + 2 = 42.0 < fhigh_min = 44.0,
    #                so assert 2 always passes across the full joint space.
    #
    # fhigh  float  uniform [44.0, 124.0]
    #     low=44.0:  flow_max (40) + 2 = 42 < 44, so assert 2 always passes
    #                even in the worst case (flow=40, fhigh=44).
    #     high=124.0: 4 Hz below Nyquist (128 Hz). Butter becomes numerically
    #                 unstable as fhigh approaches Nyquist exactly; 4 Hz margin
    #                 keeps assert 3 passing and avoids instability.
    #
    # n      int    uniform [2, 8]
    #     low=2:  Minimum meaningful roll-off (-40 dB/decade). Order 1 is too
    #             shallow to cleanly separate EEG bands.
    #     high=8: At N=8 roll-off is -160 dB/decade — extremely sharp.
    #             Beyond 8, sosfiltfilt becomes numerically unstable on typical
    #             EEG trial lengths (~256-1000 samples) and ringing artifacts
    #             appear near the cutoff edges.
    """
    assert flow > 0,            f"flow must be > 0, got {flow}"
    assert fhigh > flow + 2,    f"fhigh ({fhigh}) must be at least 2 Hz above flow ({flow})"
    assert fhigh < sfreq / 2,   f"fhigh ({fhigh}) must be below Nyquist ({sfreq/2})"

    sos = butter(N=n, Wn=[flow, fhigh], btype="bandpass", fs=sfreq, output="sos")

    # X shape: (n_trials, 1, n_channels, n_samples)
    # filtfilt operates on the last axis, so we can filter the whole array at once
    X_filt = sosfiltfilt(sos, X, axis=-1).astype(np.float32)
    return X_filt