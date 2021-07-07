import dataclasses
import logging
import os
import re
from typing import List, Tuple, Optional, Dict

import torch

from .torch_tools import TensorDict


@dataclasses.dataclass
class CheckpointPathInfo:
    path: str
    tag: str
    epochs: int


Checkpoint = Dict[str, TensorDict]


class CheckpointBuilder:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def create_checkpoint(self) -> Checkpoint:
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler
        }

    def load_checkpoint(self, checkpoint: Checkpoint, strict: bool) -> None:
        self.model.load_state_dict(checkpoint['model'], strict=strict)  # type: ignore
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


class CheckpointIO:
    def __init__(self, directory: str, tag: str, keep: bool = False) -> None:
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.old_path: Optional[str] = None

        self._epochs_string = '_step-'
        self._suffix = '.pt'

    def _get_checkpoint_filename(self, epochs: int) -> str:
        return self.tag + self._epochs_string + str(epochs) + self._suffix

    def _list_file_paths(self) -> List[str]:
        all_paths = [os.path.join(self.directory, f) for f in os.listdir(self.directory)]
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_checkpoint_path(self, path: str) -> Optional[CheckpointPathInfo]:
        filename = os.path.basename(path)
        regex = re.compile(rf'(?P<tag>.+){self._epochs_string}(?P<epochs>\d+){self._suffix}')
        match = regex.match(filename)
        if not match:
            return None

        return CheckpointPathInfo(
            path=path,
            tag=match.group('tag'),
            epochs=int(match.group('epochs')),
        )

    def _get_latest_checkpoint_path(self) -> str:
        all_file_paths = self._list_file_paths()
        checkpoint_info_list = [self._parse_checkpoint_path(path) for path in all_file_paths]
        selected_checkpoint_info_list = [info for info in checkpoint_info_list if info and info.tag == self.tag]

        if len(selected_checkpoint_info_list) == 0:
            raise RuntimeError(f"Cannot find checkpoint with tag '{self.tag}' in '{self.directory}'")

        latest_checkpoint_info = max(selected_checkpoint_info_list, key=lambda info: info.epochs)
        return latest_checkpoint_info.path

    def save(self, checkpoint: Checkpoint, epochs: int) -> None:
        if not self.keep and self.old_path:
            logging.debug(f'Deleting old checkpoint file: {self.old_path}')
            os.remove(self.old_path)

        filename = self._get_checkpoint_filename(epochs)
        path = os.path.join(self.directory, filename)
        logging.debug(f'Saving checkpoint: {path}')
        os.makedirs(self.directory, exist_ok=True)
        torch.save(obj=checkpoint, f=path)
        self.old_path = path

    def load_latest(self) -> Tuple[Checkpoint, int]:
        return self.load(self._get_latest_checkpoint_path())

    def load(self, path: str) -> Tuple[Checkpoint, int]:
        checkpoint_info = self._parse_checkpoint_path(path)

        if checkpoint_info is None:
            raise RuntimeError(f"Cannot find path '{path}'")

        logging.info(f'Loading checkpoint: {checkpoint_info.path}')
        return torch.load(f=checkpoint_info.path), checkpoint_info.epochs


class CheckpointHandler:
    def __init__(self, builder: CheckpointBuilder, io: CheckpointIO) -> None:
        self.builder = builder
        self.io = io

    def save(self, epochs: int) -> None:
        checkpoint = self.builder.create_checkpoint()
        self.io.save(checkpoint, epochs)

    def load_latest(self, strict=False):
        checkpoint, epochs = self.io.load_latest()
        self.builder.load_checkpoint(checkpoint, strict=strict)
        return epochs

    def load(self, path: str, strict=False) -> int:
        checkpoint, epochs = self.io.load(path)
        self.builder.load_checkpoint(checkpoint, strict=strict)
        return epochs
