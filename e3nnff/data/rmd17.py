import io
import logging
import os
import tarfile
import urllib.request
import zipfile
from typing import List, Sequence, Dict, Any, Tuple

import numpy as np

from e3nnff.tools import get_split_sizes
from .utils import Configuration, Configurations

# "On the role of gradients for machine learning of molecular energies and forces"
# Anders S. Christensen, O. Anatole von Lilienfeld
# https://arxiv.org/abs/2007.09593


def download_url(url: str, save_path: str) -> None:
    with urllib.request.urlopen(url) as download_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(download_file.read())


def fetch_archive(path: str, url: str, force_download=False) -> None:
    if not os.path.exists(path) and not force_download:
        logging.info(f'Downloading {url} to {path}')
        download_url(url=url, save_path=path)
    else:
        logging.info(f'File {path} exists')


def kcalpmol_to_ev(x):
    return x * 0.04336411531


def extract_configs(data: Dict[str, np.ndarray], indices: Sequence[int]) -> List[Configuration]:
    # yapf: disable
    return [
        Configuration(
            atomic_numbers=np.array(data['nuclear_charges'], dtype=int),
            positions=np.array(coords, dtype=float),  # Ang
            forces=kcalpmol_to_ev(np.array(forces, dtype=float)),  # kcal/mol/Ang -> eV/Ang
            energy=kcalpmol_to_ev(float(energy)),  # kcal/mol -> Ang
        ) for coords, forces, energy
        in zip(data['coords'][indices], data['forces'][indices], data['energies'][indices])
    ]
    # yapf: enable


subsets = {
    'aspirin', 'azobenzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'paracetamol', 'salicylic', 'toluene', 'uracil'
}
splits = list(range(1, 6))

# Atomic energies (in eV)
# Calculated with ORCA
atomic_energies = {
    1: -13.568422383046626,
    6: -1025.2770951782686,
    7: -1479.0665594928669,
    8: -2035.5709809589698,
}


def unpack_configs(
    path: str,
    subset: str,
    split: int,
) -> Tuple[Configurations, Configurations]:
    logging.info('Unpacking archive')

    assert subset in subsets
    assert split in splits

    archived_files = {
        f'rmd17/splits/index_train_{split:02}.csv': 'train_split',
        f'rmd17/splits/index_test_{split:02}.csv': 'test_split',
        f'rmd17/npz_data/rmd17_{subset}.npz': 'npz',
    }

    extracted_data: Dict[str, Any] = {}

    # Extract files
    zip_archive = zipfile.ZipFile(path, 'r')
    with zip_archive.open('rmd17.tar.bz2') as tar_bz_file:
        with tarfile.open(fileobj=tar_bz_file, mode='r|bz2') as tar_file:
            # Find files
            for file in tar_file:
                if file.name in archived_files.keys():
                    extracted_file = tar_file.extractfile(file)
                    if extracted_file:
                        extracted_data[archived_files[file.name]] = extracted_file.read()
                    else:
                        raise RuntimeError(f'Cannot read file: {file.name}')

                if len(extracted_data) == len(archived_files):
                    break

            # Process extracted splits
            for k in ['train_split', 'test_split']:
                extracted_data[k] = [int(i) for i in extracted_data[k].decode('ascii').splitlines()]

            # Generate configurations
            with np.load(io.BytesIO(extracted_data['npz']), mmap_mode='r') as np_load:
                train_configs = extract_configs(data=np_load, indices=extracted_data['train_split'])
                test_configs = extract_configs(data=np_load, indices=extracted_data['test_split'])

            return train_configs, test_configs


def split_train_valid_configs(configs: Configurations, valid_fraction: float) -> Tuple[Configurations, Configurations]:
    _, train_size = get_split_sizes(len(configs), first_fraction=valid_fraction)
    return configs[:train_size], configs[train_size:]


def load(
    directory: str,
    subset: str,
    split: int,
    valid_fraction=0.2,
    max_size_train: int = None,
    max_size_test: int = None,
    force_download=False,
) -> Tuple[Configurations, Configurations, Configurations]:
    filename = '12672038.zip'
    url = 'https://ndownloader.figshare.com/articles/12672038/versions/3'

    # Prepare
    logging.info(f'Loading rMD17, subset={subset}, split={split}')
    path = os.path.join(directory, filename)
    os.makedirs(name=directory, exist_ok=True)
    fetch_archive(path=path, url=url, force_download=force_download)

    # Process dataset
    train_valid_configs, test_configs = unpack_configs(path=path, subset=subset, split=split)

    # Slice
    train_valid_configs = train_valid_configs[:max_size_train]
    test_configs = test_configs[:max_size_test]

    train_configs, valid_configs = split_train_valid_configs(train_valid_configs, valid_fraction=valid_fraction)
    configs = (train_configs, valid_configs, test_configs)

    logging.info(f'Number of configurations: ' f'{[len(configs) for configs in configs]}')
    return configs
