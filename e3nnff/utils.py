import logging
import os
import sys
from typing import Sequence, Iterable
from typing import Tuple, Union, Optional, Dict

import ase.data
import ase.io
import numpy as np
import scipy.constants
import torch

TensorDict = Dict[str, torch.Tensor]


# noinspection PyPep8Naming
def kcal_to_kJ(x):
    return x * scipy.constants.calorie


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

    def index_to_z(self, index: Union[int, np.int64]) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_symbol_table_from_symbols(symbols: Iterable[str]) -> AtomicNumberTable:
    atomic_numbers = list(set(ase.data.atomic_numbers[symbol] for symbol in symbols))
    return AtomicNumberTable(zs=sorted(atomic_numbers))


def get_symbol_table_from_atoms(atoms: Iterable[ase.Atom]) -> AtomicNumberTable:
    symbols = []
    for atom in atoms:
        if atom.symbol not in symbols:
            symbols.append(atom.symbol)
    return get_symbol_table_from_symbols(symbols)


def atomic_numbers_to_indices(atomic_numbers: np.ndarray, table: AtomicNumberTable) -> np.ndarray:
    to_index_fn = np.vectorize(lambda z: table.z_to_index(z))
    return to_index_fn(atomic_numbers)
