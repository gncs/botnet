import logging
from typing import Tuple

import ase.data
import ase.io
import numpy as np

from e3nnff.data import Configuration, Configurations


def config_from_atoms(atoms: ase.Atoms) -> Configuration:
    energy = float(atoms.info.get('E', None))

    forces = None
    if atoms.has('forces'):
        forces = atoms.get_forces()

    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])

    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions, energy=energy, forces=forces)


def load_xyz(path: str, formatting: str = 'extxyz') -> Tuple[Configurations, Configurations, Configurations]:
    logging.info(f"Loading dataset from '{path}' (format={formatting})")
    atoms_list = ase.io.read(path, ':', format=formatting)
    configs = [config_from_atoms(atoms) for atoms in atoms_list]
    train_valid_test_configs = (configs, configs, configs)
    logging.info(f'Number of configurations: {[len(configs) for configs in train_valid_test_configs]}')
    return train_valid_test_configs
