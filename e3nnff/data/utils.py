import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom


Configurations = List[Configuration]


def get_split_sizes(size: int, first_fraction: float) -> Tuple[int, int]:
    assert 0.0 < first_fraction < 1.0
    first_size = int(first_fraction * size)
    return first_size, size - first_size


def split_train_valid_configs(configs: Configurations, valid_fraction: float) -> Tuple[Configurations, Configurations]:
    _, train_size = get_split_sizes(len(configs), first_fraction=valid_fraction)
    return configs[:train_size], configs[train_size:]


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
