import logging
import os
import tarfile
from typing import Tuple

import ase.data
import ase.db

from .utils import Configurations, Configuration, fetch_archive

# The data is partitioned as used in the SchNet paper [6] (arXiv:1706.08566):
#
# reference.db       - 80% of steps of 80% of MD trajectories
# test_within.db     - remaining 20% unseen steps of reference trajectories
# test_other.db      - remaining 20% unseen MD trajectories
#
# The data is stored in ASE sqlite format with the total energy in eV under the key total energy and
# the atomic_forces under the key atomic_forces in eV/Ang.

# Atomic energies (same as QM9) (in eV)
atomic_energies = {
    1: -0.500273,
    6: -37.846772,
    7: -54.583861,
    8: -75.064579,
    9: -99.718730,
}


def parse_db(path: str) -> Configurations:
    configs = []
    with ase.db.connect(path) as conn:
        for row in conn.select():
            configs.append(
                Configuration(
                    atomic_numbers=row.numbers,
                    positions=row.positions,
                    energy=row['total_energy'],  # eV
                    forces=row.data['atomic_forces'],  # eV/Ang
                ))

    return configs


def unpack_archive(path: str, working_dir: str) -> None:
    with tarfile.open(name=path, mode='r:gz') as archive:
        archive.extractall(path=working_dir)


def load(directory: str, force_download=False) -> Tuple[Configurations, Configurations, Configurations]:
    url = 'http://quantum-machine.org/datasets/iso17.tar.gz'
    archive_name = 'iso17.tar.gz'
    folder_name = 'iso17'
    filenames = ('reference.db', 'test_within.db', 'test_other.db')

    # Download
    logging.info('Loading ISO17 dataset')
    archive_path = os.path.join(directory, archive_name)
    os.makedirs(name=directory, exist_ok=True)
    fetch_archive(path=archive_path, url=url, force_download=force_download)

    # Unpack
    extracted_directory = os.path.join(directory, folder_name)
    if os.path.exists(extracted_directory):
        logging.info(f'Directory {extracted_directory} exists')
    else:
        unpack_archive(path=archive_path, working_dir=directory)

    # Process dataset
    logging.info(f'Parsing ISO17 dataset files: {filenames}')

    configs_tuple = tuple(parse_db(path=os.path.join(extracted_directory, filename1)) for filename1 in filenames)
    assert len(configs_tuple) == 3

    return configs_tuple  # type: ignore
