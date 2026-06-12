from constants import DATASET_REGISTRY
import json
import hashlib

def cache_key(dataset_key: str) -> str:
    """
    Stable MD5 key over paradigm parameters that affect the raw cached arrays.
    fmin/fmax are excluded — bandpass filtering happens after loading.
    """
    cfg = DATASET_REGISTRY[dataset_key]
    fingerprint = json.dumps({
        "dataset_key":   dataset_key,
        "dataset_cls":   cfg.dataset_cls.__name__,
        "paradigm_cls":  cfg.paradigm_cls.__name__,
        "tmin":          cfg.tmin,
        "tmax":          cfg.tmax,
        "resample":      cfg.resample,
        "events":        cfg.events,
        "n_classes":     cfg.n_classes,
    }, sort_keys=True)
    return hashlib.md5(fingerprint.encode()).hexdigest()