from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

CACHE_DIR = Path(
    __import__("os").environ.get("SPIKING_EEGNET_CACHE", Path.home() / ".cache" / "spiking_eegnet")
)

@dataclass
class DatasetConfig:
    dataset_cls: type
    paradigm_cls: type
    tmin: float = 0.0
    tmax: float = 4.0
    resample: Optional[float] = 128.0
    events: Optional[list] = None
    n_classes: Optional[int] = None

DATASET_REGISTRY = {
    "BNCI2014_001": DatasetConfig(
        dataset_cls=BNCI2014_001,
        paradigm_cls=MotorImagery,
        tmin=0.0, tmax=4.0,
        resample=250.0,
        events=["left_hand", "right_hand", "feet", "tongue"],
        n_classes=4,
    ),
}