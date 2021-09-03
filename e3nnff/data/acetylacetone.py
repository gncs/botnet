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

# Atomic energies (in eV)
atomic_energies = {
    1: -13.568422178253735,
    6: -1026.8538996116154,
    8: -2037.796869412825,
}


def config_from_atoms(atoms: ase.Atoms) -> Configuration:
    energy = float(atoms.info['energy'])

    forces = None
    if atoms.has('forces'):
        forces = atoms.get_forces()

    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])

    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions, energy=energy, forces=forces)


def unpack_configs(path: str) -> Dict[str, Configurations]:
    logging.info('Unpacking archive')

    subsets = {'train_300K', 'train_600K', 'test_MD_300K', 'test_MD_600K', 'test_dihedral', 'test_H_transfer'}
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
    filename = 'acac_dataset.tar.gz'

    logging.info('Loading acetylacetone dataset')
    path = os.path.join(directory, filename)
    return unpack_configs(path=path)
