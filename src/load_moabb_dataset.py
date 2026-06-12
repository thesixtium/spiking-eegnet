import json
from pathlib import Path
from typing import Optional

import moabb
import numpy as np

from constants import CACHE_DIR, DATASET_REGISTRY
from cache_key import cache_key

moabb.set_log_level("warning")


def load_moabb_dataset(dataset_key: str, cache_dir: Optional[Path] = None):
    """
    Load a MOABB dataset and return raw (unfiltered) arrays.

    No bandpass filtering is applied here. Call bandpass_filter() on the
    returned X before training. This keeps the cache filter-agnostic so
    fmin/fmax can be searched without re-downloading.

    Returns
    -------
    X            : np.ndarray  (n_trials, 1, n_channels, n_samples)  float32
    y            : np.ndarray  (n_trials,)                           int64
    subject_ids  : np.ndarray  (n_trials,)                           int (0-indexed)
    meta         : dict  {n_classes, n_channels, n_samples, n_subjects,
                          class_names, subject_list, sfreq}
    """
    cache_root = Path(cache_dir) if cache_dir else CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)

    key        = cache_key(dataset_key)
    cache_path = cache_root / f"{dataset_key}_{key}.npz"
    meta_path  = cache_root / f"{dataset_key}_{key}.json"

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_path.exists() and meta_path.exists():
        print(f"[{dataset_key}] loading from cache: {cache_path}")
        arrays = np.load(cache_path)
        X, y, subject_ids = arrays["X"], arrays["y"], arrays["subject_ids"]
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[{dataset_key}] cache hit: {X.shape}, classes={meta['class_names']}, "
              f"subjects={meta['n_subjects']}")
        return X, y, subject_ids, meta

    # ── Cache miss ────────────────────────────────────────────────────────────
    print(f"[{dataset_key}] no cache found, running paradigm.get_data() ...")
    cfg     = DATASET_REGISTRY[dataset_key]
    dataset = cfg.dataset_cls()

    paradigm_kwargs = dict(
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        # Wide passband — effectively no filtering at the load stage.
        # All meaningful EEG content (delta through high-gamma) is preserved.
        # The caller is responsible for applying their own bandpass via bandpass_filter().
        fmin=0.1,
        fmax=120.0,
    )
    if cfg.resample  is not None: paradigm_kwargs["resample"]  = cfg.resample
    if cfg.events    is not None: paradigm_kwargs["events"]    = cfg.events
    if cfg.n_classes is not None: paradigm_kwargs["n_classes"] = cfg.n_classes

    paradigm           = cfg.paradigm_cls(**paradigm_kwargs)
    X_raw, labels, metadata = paradigm.get_data(dataset=dataset)

    class_names = sorted(set(labels))
    label_map   = {c: i for i, c in enumerate(class_names)}
    y           = np.array([label_map[l] for l in labels], dtype=np.int64)

    X = X_raw[:, np.newaxis, :, :].astype(np.float32)

    subjects     = metadata["subject"].values
    subject_list = sorted(set(subjects))
    subj_map     = {s: i for i, s in enumerate(subject_list)}
    subject_ids  = np.array([subj_map[s] for s in subjects], dtype=np.int64)

    sfreq = cfg.resample if cfg.resample is not None else float(X_raw.info["sfreq"])

    def _to_py(v):
        if isinstance(v, np.integer):  return int(v)
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, (list, np.ndarray)): return [_to_py(i) for i in v]
        return v

    meta = {
        "n_classes":    int(len(class_names)),
        "n_channels":   int(X.shape[2]),
        "n_samples":    int(X.shape[3]),
        "class_names":  [_to_py(c) for c in class_names],
        "subject_list": [_to_py(s) for s in subject_list],
        "n_subjects":   int(len(subject_list)),
        "sfreq":        float(sfreq),
    }

    np.savez_compressed(cache_path, X=X, y=y, subject_ids=subject_ids)

    def _to_serialisable(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, list):        return [_to_serialisable(i) for i in obj]
        return obj

    with open(meta_path, "w") as f:
        json.dump({k: _to_serialisable(v) for k, v in meta.items()}, f, indent=2)
    print(f"[{dataset_key}] cached to {cache_path}")
    return X, y, subject_ids, meta