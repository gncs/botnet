import json
import logging
import os
import sys
from typing import Sequence, Iterable, Tuple, Union, Optional, Dict, Any

import numpy as np
import scipy.constants
import torch

from .torch_tools import to_numpy

_bohr_per_angstrom = scipy.constants.angstrom / scipy.constants.value('Bohr radius')
_angstrom_per_bohr = 1 / _bohr_per_angstrom
_joule_per_hartree = scipy.constants.value('Hartree energy')
_hartree_per_joule = 1 / _joule_per_hartree
_hartree_per_kjpmol = _hartree_per_joule * 1_000 / scipy.constants.Avogadro
_kj_per_kcal = scipy.constants.calorie
_hartree_per_ev = 1 / scipy.constants.value('Hartree energy in eV')


def ev_to_hartree(x):
    return x * _hartree_per_ev


def kcalpmol_to_hartree(x):
    return x * _kj_per_kcal * _hartree_per_kjpmol


def angstrom_to_bohr(x):
    return x * _bohr_per_angstrom


def kcalpmol_per_angstrom_to_hartree_per_bohr(x):
    return kcalpmol_to_hartree(x) * _angstrom_per_bohr


def get_tag(name: str, seed: int) -> str:
    return f'{name}_run-{seed}'


def get_split_sizes(size: int, first_fraction: float) -> Tuple[int, int]:
    assert 0.0 < first_fraction < 1.0
    first_size = int(first_fraction * size)
    return first_size, size - first_size


def setup_logger(level: Union[int, str] = logging.INFO, tag: Optional[str] = None, directory: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + '.log')
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f'AtomicNumberTable: {tuple(s for s in self.zs)}'

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(atomic_numbers: np.ndarray, z_table: AtomicNumberTable) -> np.ndarray:
    to_index_fn = np.vectorize(lambda z: z_table.z_to_index(z))
    return to_index_fn(atomic_numbers)


def get_optimizer(name: str, learning_rate: float, parameters: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
    if name == 'adam':
        amsgrad = False
    elif name == 'amsgrad':
        amsgrad = True
    else:
        raise RuntimeError(f"Unknown optimizer '{name}'")

    return torch.optim.Adam(parameters, lr=learning_rate, amsgrad=amsgrad)


class UniversalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return to_numpy(obj)
        return json.JSONEncoder.default(self, obj)


class ProgressLogger:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.filename = tag + '.txt'
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        logging.debug(f'Saving info: {self.path}')
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode='a') as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write('\n')
