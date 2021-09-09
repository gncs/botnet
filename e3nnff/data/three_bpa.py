import io
import logging
import os
import tarfile
from typing import Dict

import ase
import ase.data
import ase.io
import numpy as np

from .utils import Configuration, Configurations

# "Linear Atomic Cluster Expansion Force Fields for Organic Molecules: beyond RMSE"
# Kovacs, D. P.; Oord, C. van der; Kucera, J.; Allen, A.; Cole, D.; Ortner, C.; Csanyi, G. 2021.
# https://doi.org/10.33774/chemrxiv-2021-7qlf5-v3

# Atomic energies (in eV)
atomic_energies = {
    1: -13.587222780835477,
    6: -1029.4889999855063,
    7: -1484.9814568572233,
    8: -2041.9816003861047,
}


def config_from_atoms(atoms: ase.Atoms) -> Configuration:
    energy = atoms.info.get('energy', None)  # eV
    forces = atoms.arrays.get('forces', None)  # eV / Ang
    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions, energy=energy, forces=forces)


def unpack_configs(path: str) -> Dict[str, Configurations]:
    extracted_data: Dict[str, Configurations] = {}
    with tarfile.open(name=path, mode='r|gz') as tar_file:
        for file in tar_file:
            basename = os.path.basename(file.name)
            root, ext = os.path.splitext(basename)
            if file.isfile() and ext == '.xyz':
                extracted_file = tar_file.extractfile(file)
                if extracted_file:
                    content = io.StringIO(extracted_file.read().decode('ascii'))
                    configs = [config_from_atoms(config) for config in ase.io.read(content, format='extxyz', index=':')]
                    extracted_data[root] = configs
                else:
                    raise RuntimeError(f'Cannot read file: {file.name}')

    return extracted_data


def load(directory: str) -> Dict[str, Configurations]:
    filename = 'dataset_3BPA.tar.gz'

    # Prepare
    logging.info('Loading 3BPA')
    path = os.path.join(directory, filename)
    return unpack_configs(path=path)
