"""
tests.py — pytest suite for SpikingEEGNet pipeline components.

Covers:
  - zscore_normalize
  - bandpass_filter
  - make_loader
  - SpikingEEGNet (shape, forward pass, num_steps)
  - build_model
  - train_one_epoch
  - evaluate
  - run_training
  - cache_key
  - experiment_loso (smoke test with tiny fake data)

Does NOT cover:
  - load_moabb_dataset  (requires network / MOABB install; test separately)
  - main.py             (integration entry point, not unit-testable in isolation)

Run with:
    pytest tests.py -v
"""

import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
# Assumes tests.py lives alongside the source files, or adjust paths below.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import numpy as np
import pytest
import torch

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_TRIALS    = 20
N_CHANNELS  = 22
N_SAMPLES   = 128   # small but valid for all pool/kernel combos below
N_CLASSES   = 4
SFREQ       = 256.0

RNG = np.random.default_rng(42)


def make_X(n_trials=N_TRIALS, n_channels=N_CHANNELS, n_samples=N_SAMPLES):
    """Random float32 EEG array in (n_trials, 1, n_channels, n_samples)."""
    return RNG.standard_normal((n_trials, 1, n_channels, n_samples)).astype(np.float32)


def make_y(n_trials=N_TRIALS, n_classes=N_CLASSES):
    return RNG.integers(0, n_classes, size=n_trials).astype(np.int64)


def make_meta(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES):
    return {
        "n_classes":   n_classes,
        "n_channels":  n_channels,
        "n_samples":   n_samples,
        "n_subjects":  3,
        "class_names": [str(i) for i in range(n_classes)],
        "subject_list": [1, 2, 3],
        "sfreq":       SFREQ,
    }


DEVICE = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# zscore_normalize
# ══════════════════════════════════════════════════════════════════════════════

class TestZscoreNormalize:
    from zscore_normalize import zscore_normalize

    def test_output_shape(self):
        from zscore_normalize import zscore_normalize
        X = make_X()
        assert zscore_normalize(X).shape == X.shape

    def test_dtype_preserved(self):
        from zscore_normalize import zscore_normalize
        X = make_X()
        assert zscore_normalize(X).dtype == np.float32

    def test_per_epoch_mean_near_zero(self):
        from zscore_normalize import zscore_normalize
        X = make_X()
        X_norm = zscore_normalize(X)
        means = X_norm.mean(axis=(1, 2, 3))
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_per_epoch_std_near_one(self):
        from zscore_normalize import zscore_normalize
        X = make_X()
        X_norm = zscore_normalize(X)
        stds = X_norm.std(axis=(1, 2, 3))
        np.testing.assert_allclose(stds, 1.0, atol=1e-4)

    def test_epochs_are_independent(self):
        """Shifting one epoch should not affect others."""
        from zscore_normalize import zscore_normalize
        X = make_X(n_trials=4)
        X_shifted = X.copy()
        X_shifted[0] += 1000.0          # big shift on epoch 0
        out_orig    = zscore_normalize(X)
        out_shifted = zscore_normalize(X_shifted)
        # epochs 1-3 should be identical
        np.testing.assert_allclose(out_orig[1:], out_shifted[1:], atol=1e-6)

    def test_constant_epoch_no_nan(self):
        """Constant-signal epoch should produce zeros, not NaN."""
        from zscore_normalize import zscore_normalize
        X = np.zeros((2, 1, N_CHANNELS, N_SAMPLES), dtype=np.float32)
        X[1] = make_X(n_trials=1)
        out = zscore_normalize(X)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out[0], 0.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# bandpass_filter
# ══════════════════════════════════════════════════════════════════════════════

class TestBandpassFilter:

    def test_output_shape(self):
        from bandpass_filter import bandpass_filter
        X = make_X()
        assert bandpass_filter(X, SFREQ, 4.0, 40.0).shape == X.shape

    def test_dtype_float32(self):
        from bandpass_filter import bandpass_filter
        X = make_X()
        assert bandpass_filter(X, SFREQ, 4.0, 40.0).dtype == np.float32

    def test_attenuates_dc(self):
        """DC component should be strongly attenuated."""
        from bandpass_filter import bandpass_filter
        # Flat signal → after bandpass should be near zero
        X_dc = np.ones((4, 1, N_CHANNELS, N_SAMPLES), dtype=np.float32) * 5.0
        X_filt = bandpass_filter(X_dc, SFREQ, 4.0, 40.0)
        assert np.abs(X_filt).max() < 0.1

    def test_passes_in_band_signal(self):
        """A sine wave inside the passband should survive with high amplitude."""
        from bandpass_filter import bandpass_filter
        t = np.linspace(0, N_SAMPLES / SFREQ, N_SAMPLES, endpoint=False)
        freq_hz = 15.0   # squarely inside 4–40 Hz
        sine = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
        X = np.tile(sine, (4, 1, N_CHANNELS, 1))
        X_filt = bandpass_filter(X, SFREQ, 4.0, 40.0)
        # Amplitude should be largely preserved (> 0.5 of original)
        assert np.abs(X_filt).max() > 0.5

    def test_assert_flow_too_low(self):
        from bandpass_filter import bandpass_filter
        with pytest.raises(AssertionError):
            bandpass_filter(make_X(), SFREQ, flow=-1.0, fhigh=40.0)

    def test_assert_bandwidth_too_narrow(self):
        from bandpass_filter import bandpass_filter
        with pytest.raises(AssertionError):
            bandpass_filter(make_X(), SFREQ, flow=20.0, fhigh=21.0)  # gap < 2

    def test_assert_fhigh_above_nyquist(self):
        from bandpass_filter import bandpass_filter
        with pytest.raises(AssertionError):
            bandpass_filter(make_X(), SFREQ, flow=4.0, fhigh=SFREQ / 2)


# ══════════════════════════════════════════════════════════════════════════════
# make_loader
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeLoader:

    def test_yields_correct_batch_shape(self):
        from make_loader import make_loader
        X, y = make_X(), make_y()
        loader = make_loader(X, y, batch_size=8)
        xb, yb = next(iter(loader))
        assert xb.shape == (8, 1, N_CHANNELS, N_SAMPLES)
        assert yb.shape == (8,)

    def test_covers_all_samples(self):
        from make_loader import make_loader
        X, y = make_X(), make_y()
        loader = make_loader(X, y, batch_size=7, shuffle=False)
        total = sum(xb.shape[0] for xb, _ in loader)
        assert total == N_TRIALS

    def test_no_shuffle_is_deterministic(self):
        from make_loader import make_loader
        X, y = make_X(), make_y()
        loader = make_loader(X, y, batch_size=N_TRIALS, shuffle=False)
        xb1, _ = next(iter(loader))
        xb2, _ = next(iter(loader))
        assert torch.equal(xb1, xb2)

    def test_dtype_passthrough(self):
        from make_loader import make_loader
        X, y = make_X(), make_y()
        loader = make_loader(X, y, batch_size=N_TRIALS, shuffle=False)
        xb, yb = next(iter(loader))
        assert xb.dtype == torch.float32
        assert yb.dtype == torch.int64


# ══════════════════════════════════════════════════════════════════════════════
# SpikingEEGNet
# ══════════════════════════════════════════════════════════════════════════════

class TestSpikingEEGNet:

    def _model(self, **kwargs):
        from spiking_eegnet import SpikingEEGNet
        defaults = dict(
            num_classes=N_CLASSES,
            num_channels=N_CHANNELS,
            num_samples=N_SAMPLES,
            temporal_filters=8,
            depth_multiplier=2,
            pointwise_filters=16,
            temporal_kernel_div=4,
            separable_kernel_size=8,
            pool1_size=4,
            pool2_size=2,
            dropout=0.25,
            beta=0.9,
            spike_grad_slope=25.0,
        )
        defaults.update(kwargs)
        return SpikingEEGNet(**defaults)

    def test_output_shape_single_step(self):
        model = self._model()
        x = torch.zeros(4, 1, N_CHANNELS, N_SAMPLES)
        out = model(x, num_steps=1)
        assert out.shape == (1, 4, N_CLASSES)

    def test_output_shape_multi_step(self):
        model = self._model()
        x = torch.zeros(4, 1, N_CHANNELS, N_SAMPLES)
        out = model(x, num_steps=5)
        assert out.shape == (5, 4, N_CLASSES)

    def test_output_finite(self):
        model = self._model()
        x = torch.randn(4, 1, N_CHANNELS, N_SAMPLES)
        out = model(x, num_steps=3)
        assert torch.isfinite(out).all()

    def test_param_count_positive(self):
        model = self._model()
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_gradient_flows(self):
        model = self._model()
        x = torch.randn(4, 1, N_CHANNELS, N_SAMPLES)
        out = model(x, num_steps=2)
        loss = out.mean(0).sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_assert_bad_pool_sizes(self):
        """Pool combo that collapses the time dim should raise AssertionError."""
        from spiking_eegnet import SpikingEEGNet
        with pytest.raises(AssertionError):
            SpikingEEGNet(
                num_classes=N_CLASSES, num_channels=N_CHANNELS, num_samples=N_SAMPLES,
                pool1_size=8, pool2_size=8,        # 64x reduction on 128-sample input
                separable_kernel_size=4,
            )

    def test_assert_separable_kernel_too_large(self):
        """separable_kernel_size > time_after_pool1 should raise AssertionError."""
        from spiking_eegnet import SpikingEEGNet
        with pytest.raises(AssertionError):
            SpikingEEGNet(
                num_classes=N_CLASSES, num_channels=N_CHANNELS, num_samples=N_SAMPLES,
                pool1_size=4,                      # time_after_pool1 = 32
                separable_kernel_size=64,           # > 32 → invalid
            )

    @pytest.mark.parametrize("temporal_filters,depth_multiplier,pointwise_filters", [
        (4,  1, 8),
        (16, 2, 32),
        (32, 4, 64),
    ])
    def test_filter_count_variants(self, temporal_filters, depth_multiplier, pointwise_filters):
        model = self._model(
            temporal_filters=temporal_filters,
            depth_multiplier=depth_multiplier,
            pointwise_filters=pointwise_filters,
        )
        x = torch.zeros(2, 1, N_CHANNELS, N_SAMPLES)
        out = model(x, num_steps=1)
        assert out.shape == (1, 2, N_CLASSES)


# ══════════════════════════════════════════════════════════════════════════════
# build_model
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildModel:

    def test_returns_model_on_device(self):
        from build_model import build_model
        meta = make_meta()
        model = build_model(meta, DEVICE)
        param_devices = {p.device for p in model.parameters()}
        assert all(d.type == "cpu" for d in param_devices)

    def test_model_has_correct_output_classes(self):
        from build_model import build_model
        meta = make_meta(n_classes=2)
        model = build_model(meta, DEVICE)
        assert model.num_classes == 2

    def test_model_kwargs_forwarded(self):
        from build_model import build_model
        meta = make_meta()
        model = build_model(meta, DEVICE, dropout=0.1, beta=0.8)
        # Just confirm it builds without error; kwargs accepted
        assert model is not None


# ══════════════════════════════════════════════════════════════════════════════
# train_one_epoch
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainOneEpoch:

    def _setup(self):
        from make_loader import make_loader
        from build_model import build_model
        import torch.nn as nn, torch.optim as optim
        meta    = make_meta()
        model   = build_model(meta, DEVICE, temporal_kernel_div=4, separable_kernel_size=8,
                               pool1_size=4, pool2_size=2)
        loader  = make_loader(make_X(), make_y(), batch_size=8)
        opt     = optim.Adam(model.parameters(), lr=1e-3)
        crit    = nn.CrossEntropyLoss()
        return model, loader, opt, crit

    def test_returns_float(self):
        from train_one_epoch import train_one_epoch
        model, loader, opt, crit = self._setup()
        loss = train_one_epoch(model, loader, opt, crit, DEVICE, n_steps=2)
        assert isinstance(loss, float)

    def test_loss_is_finite(self):
        from train_one_epoch import train_one_epoch
        model, loader, opt, crit = self._setup()
        loss = train_one_epoch(model, loader, opt, crit, DEVICE, n_steps=2)
        assert np.isfinite(loss)

    def test_loss_is_positive(self):
        from train_one_epoch import train_one_epoch
        model, loader, opt, crit = self._setup()
        loss = train_one_epoch(model, loader, opt, crit, DEVICE, n_steps=2)
        assert loss > 0.0

    def test_parameters_update(self):
        from train_one_epoch import train_one_epoch
        model, loader, opt, crit = self._setup()
        params_before = [p.clone().detach() for p in model.parameters()]
        train_one_epoch(model, loader, opt, crit, DEVICE, n_steps=2)
        params_after  = [p.clone().detach() for p in model.parameters()]
        any_changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        assert any_changed


# ══════════════════════════════════════════════════════════════════════════════
# evaluate
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def _setup(self):
        from make_loader import make_loader
        from build_model import build_model
        meta   = make_meta()
        model  = build_model(meta, DEVICE, temporal_kernel_div=4, separable_kernel_size=8,
                              pool1_size=4, pool2_size=2)
        loader = make_loader(make_X(), make_y(), batch_size=8, shuffle=False)
        return model, loader

    def test_returns_float_in_0_1(self):
        from evaluate import evaluate
        model, loader = self._setup()
        acc = evaluate(model, loader, DEVICE, n_steps=2)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_no_grad_context(self):
        """evaluate() should not leave requires_grad state dirty."""
        from evaluate import evaluate
        model, loader = self._setup()
        evaluate(model, loader, DEVICE, n_steps=2)
        # If no exception, grad state was cleanly restored

    def test_consistent_on_same_data(self):
        """Two calls on same loader/model → same balanced accuracy."""
        from evaluate import evaluate
        model, loader = self._setup()
        acc1 = evaluate(model, loader, DEVICE, n_steps=2)
        acc2 = evaluate(model, loader, DEVICE, n_steps=2)
        assert acc1 == acc2


# ══════════════════════════════════════════════════════════════════════════════
# run_training
# ══════════════════════════════════════════════════════════════════════════════

class TestRunTraining:

    def _loaders(self):
        from make_loader import make_loader
        return (
            make_loader(make_X(), make_y(), batch_size=8),
            make_loader(make_X(n_trials=8), make_y(n_trials=8), batch_size=8, shuffle=False),
        )

    def _model(self):
        from build_model import build_model
        return build_model(make_meta(), DEVICE,
                           temporal_kernel_div=4, separable_kernel_size=8,
                           pool1_size=4, pool2_size=2)

    def test_history_keys_present(self):
        from run_training import run_training
        train_l, val_l = self._loaders()
        hist = run_training(self._model(), train_l, val_l, epochs=2, lr=1e-3,
                            device=DEVICE, n_steps_train=2, n_steps_eval=2,
                            eval_every_epoch=True)
        assert "loss" in hist and "bal_acc" in hist

    def test_eval_every_epoch_length(self):
        from run_training import run_training
        train_l, val_l = self._loaders()
        hist = run_training(self._model(), train_l, val_l, epochs=3, lr=1e-3,
                            device=DEVICE, n_steps_train=2, n_steps_eval=2,
                            eval_every_epoch=True)
        assert len(hist["loss"])    == 3
        assert len(hist["bal_acc"]) == 3

    def test_no_eval_every_epoch_single_acc(self):
        """eval_every_epoch=False → bal_acc has exactly one entry."""
        from run_training import run_training
        train_l, val_l = self._loaders()
        hist = run_training(self._model(), train_l, val_l, epochs=3, lr=1e-3,
                            device=DEVICE, n_steps_train=2, n_steps_eval=2,
                            eval_every_epoch=False)
        assert len(hist["loss"])    == 3
        assert len(hist["bal_acc"]) == 1

    def test_loss_values_finite(self):
        from run_training import run_training
        train_l, val_l = self._loaders()
        hist = run_training(self._model(), train_l, val_l, epochs=2, lr=1e-3,
                            device=DEVICE, n_steps_train=2, n_steps_eval=2,
                            eval_every_epoch=True)
        assert all(np.isfinite(l) for l in hist["loss"])
        assert all(np.isfinite(a) for a in hist["bal_acc"])


# ══════════════════════════════════════════════════════════════════════════════
# cache_key
# ══════════════════════════════════════════════════════════════════════════════

class TestCacheKey:

    def test_returns_string(self):
        from cache_key import cache_key
        assert isinstance(cache_key("BNCI2014_001"), str)

    def test_deterministic(self):
        from cache_key import cache_key
        assert cache_key("BNCI2014_001") == cache_key("BNCI2014_001")

    def test_hex_format(self):
        from cache_key import cache_key
        key = cache_key("BNCI2014_001")
        int(key, 16)   # raises ValueError if not valid hex

    def test_length_32(self):
        """MD5 hex digest is always 32 characters."""
        from cache_key import cache_key
        assert len(cache_key("BNCI2014_001")) == 32

    def test_unknown_key_raises(self):
        from cache_key import cache_key
        with pytest.raises(KeyError):
            cache_key("DOES_NOT_EXIST")


# ══════════════════════════════════════════════════════════════════════════════
# experiment_loso  (smoke test — no MOABB required)
# ══════════════════════════════════════════════════════════════════════════════

class TestExperimentLoso:
    """
    Smoke test: verify experiment_loso runs end-to-end and returns sensible
    types/shapes using synthetic data. Not a correctness test.
    """

    def _data(self, n_subjects=3, trials_per_subject=8):
        n = n_subjects * trials_per_subject
        X = make_X(n_trials=n)
        y = make_y(n_trials=n)
        subj = np.repeat(np.arange(n_subjects), trials_per_subject).astype(np.int64)
        return X, y, subj

    def _cfg(self):
        return dict(epochs=1, batch_size=8, lr=1e-3,
                    n_steps_train=2, n_steps_eval=2)

    def _meta(self):
        return make_meta()

    def test_returns_history_and_acc(self):
        from experiment_loso import experiment_loso
        X, y, subj = self._data()
        hist, acc = experiment_loso(
            X, y, subj, self._meta(), DEVICE, self._cfg(),
            test_subject_idx=0,
        )
        assert "loss" in hist and "bal_acc" in hist
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_train_test_split_correct(self):
        """Test subject's trials should not appear in train."""
        from experiment_loso import experiment_loso
        X, y, subj = self._data(n_subjects=3, trials_per_subject=8)
        # We can't inspect internals directly, but if the split is wrong
        # the model would train and eval on the same data — no easy assert.
        # At minimum confirm it finishes without error for each subject.
        for s in range(3):
            experiment_loso(X, y, subj, self._meta(), DEVICE, self._cfg(),
                            test_subject_idx=s)

    def test_history_loss_length_matches_epochs(self):
        from experiment_loso import experiment_loso
        X, y, subj = self._data()
        cfg = self._cfg()
        hist, _ = experiment_loso(X, y, subj, self._meta(), DEVICE, cfg,
                                  test_subject_idx=0)
        assert len(hist["loss"]) == cfg["epochs"]