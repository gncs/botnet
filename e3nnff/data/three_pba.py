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
    energy = float(atoms.info.get('energy'))  # eV
    forces = atoms.get_forces()  # eV / Ang
    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions, energy=energy, forces=forces)


def unpack_configs(path: str) -> Dict[str, Configurations]:
    logging.info('Unpacking archive')

    subsets = {'test_300K', 'test_600K', 'test_1200K', 'train_300K', 'train_mixed'}
    file_subset_dict = {f'{subset}.xyz': subset for subset in subsets}

    extracted_data: Dict[str, Configurations] = {}

    # Extract files
    with tarfile.open(name=path, mode='r|gz') as tar_file:
        # Find files
        for file in tar_file:
            basename = os.path.basename(file.name)
            if basename in file_subset_dict.keys():
                extracted_file = tar_file.extractfile(file)
                if extracted_file:
                    content = io.StringIO(extracted_file.read().decode('ascii'))
                    configs = [config_from_atoms(config) for config in ase.io.read(content, format='extxyz', index=':')]
                    extracted_data[file_subset_dict[basename]] = configs
                else:
                    raise RuntimeError(f'Cannot read file: {file.name}')

            if len(extracted_data) == len(file_subset_dict):
                break

    return extracted_data


def load(directory: str) -> Dict[str, Configurations]:
    filename = 'configs_final.tar.gz'

    # Prepare
    logging.info('Loading 3BPA')
    path = os.path.join(directory, filename)
    return unpack_configs(path=path)
