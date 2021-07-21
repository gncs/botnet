from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Bohr
    energy: Optional[float] = None  # Hartree
    forces: Optional[Forces] = None  # Hartree/Bohr


Configurations = List[Configuration]


def get_split_sizes(size: int, first_fraction: float) -> Tuple[int, int]:
    assert 0.0 < first_fraction < 1.0
    first_size = int(first_fraction * size)
    return first_size, size - first_size


def split_train_valid_configs(configs: Configurations, valid_fraction: float) -> Tuple[Configurations, Configurations]:
    _, train_size = get_split_sizes(len(configs), first_fraction=valid_fraction)
    return configs[:train_size], configs[train_size:]
