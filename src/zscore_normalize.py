import numpy as np


def zscore_normalize(X: np.ndarray, axis=(1,2,3), eps: float = 1e-8) -> np.ndarray:
    """
    Apply per-epoch z-score normalization to EEG trials.

    Each epoch is normalized independently: subtract its mean and divide by
    its standard deviation, computed across all channels and samples within
    that epoch. This removes epoch-level amplitude shifts and scale differences
    before bandpass filtering.

    Parameters
    ----------
    X   : np.ndarray  (n_trials, 1, n_channels, n_samples)  float32
    eps : float       Small constant added to std to prevent division by zero
                      (relevant for synthetic/constant-signal epochs in tests)

    Returns
    -------
    X_norm : np.ndarray  same shape as X, float32

    Notes
    -----
    Normalization axes are (1, 2, 3) — i.e. the entire epoch volume per trial.
    This matches the standard BCI preprocessing convention where a single trial
    is treated as one unit of signal, rather than normalizing per-channel (which
    would destroy spatial amplitude relationships across channels).
    SWITCH TO (1,3) FOR PER CHANNEL NORMALIZATION

    Call this before bandpass_filter(). Filtering after normalization is correct
    because the bandpass operates on the temporal axis only; the filter's
    frequency response is unaffected by the amplitude scaling applied here.
    """
    # Mean and std over the epoch volume: axes 1 (dummy), 2 (channels), 3 (samples)
    # keepdims=True so broadcasting back to (n_trials, 1, n_channels, n_samples) is exact
    mean = X.mean(axis=axis, keepdims=True)
    std  = X.std(axis=axis, keepdims=True)
    return ((X - mean) / (std + eps)).astype(np.float32)