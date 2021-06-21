import dataclasses
import json
import logging
import os
import re
import sys
from typing import Sequence, Iterable, List, Tuple, Union, Optional, Dict, Any

import numpy as np
import scipy.constants
import torch

from .torch_tools import to_numpy


# noinspection PyPep8Naming
def kcal_to_kJ(x):
    return x * scipy.constants.calorie


# noinspection PyPep8Naming
def eV_to_kJ_per_mol(x):
    return x * scipy.constants.electron_volt * scipy.constants.Avogadro / 1000


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


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def get_optimizer(name: str, learning_rate: float, parameters: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
    if name == 'adam':
        amsgrad = False
    elif name == 'amsgrad':
        amsgrad = True
    else:
        raise RuntimeError(f"Unknown optimizer '{name}'")

    return torch.optim.Adam(parameters, lr=learning_rate, amsgrad=amsgrad)


@dataclasses.dataclass
class ModelPathInfo:
    path: str
    tag: str
    steps: int


class ModelIO:
    def __init__(self, directory: str, tag: str, keep: bool = False) -> None:
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.old_path: Optional[str] = None

        self._steps_string = '_step-'
        self._suffix = '.model'

    def _get_model_filename(self, steps: int) -> str:
        return self.tag + self._steps_string + str(steps) + self._suffix

    def _list_file_paths(self) -> List[str]:
        all_paths = [os.path.join(self.directory, f) for f in os.listdir(self.directory)]
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_model_path(self, path: str) -> Optional[ModelPathInfo]:
        filename = os.path.basename(path)
        regex = re.compile(rf'(?P<tag>.+){self._steps_string}(?P<steps>\d+){self._suffix}')
        match = regex.match(filename)
        if not match:
            return None

        return ModelPathInfo(
            path=path,
            tag=match.group('tag'),
            steps=int(match.group('steps')),
        )

    def save(self, module: torch.nn.Module, steps: int) -> None:
        if not self.keep and self.old_path:
            logging.debug(f'Deleting old model: {self.old_path}')
            os.remove(self.old_path)

        filename = self._get_model_filename(steps)
        path = os.path.join(self.directory, filename)
        logging.debug(f'Saving model: {path}')
        os.makedirs(self.directory, exist_ok=True)
        torch.save(obj=module, f=path)
        self.old_path = path

    def load(self, path: str, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, int]:
        model_info = self._parse_model_path(path)

        if model_info is None:
            raise RuntimeError(f"Cannot find model '{path}'")

        logging.info(f'Loading model: {model_info.path}')
        model = torch.load(f=model_info.path, map_location=device)

        return model, model_info.steps

    def load_latest(self, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, int]:
        all_file_paths = self._list_file_paths()
        model_info_list = [self._parse_model_path(path) for path in all_file_paths]
        selected_model_info_list = [info for info in model_info_list if info and info.tag == self.tag]

        if len(selected_model_info_list) == 0:
            raise RuntimeError(f"Cannot find model with tag '{self.tag}' in '{self.directory}'")

        latest_model_info = max(selected_model_info_list, key=lambda info: info.steps)

        logging.info(f'Loading model: {latest_model_info.path}')
        model = torch.load(f=latest_model_info.path, map_location=device)

        return model, latest_model_info.steps


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
