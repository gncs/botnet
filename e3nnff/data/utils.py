from dataclasses import dataclass
from typing import Optional, List

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
