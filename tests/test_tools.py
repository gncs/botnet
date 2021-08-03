import tempfile

import numpy as np
import torch
import torch.nn.functional
from torch import nn, optim

from e3nnff.data import Configuration
from e3nnff.tools import AtomicNumberTable, atomic_numbers_to_indices, CheckpointState, CheckpointHandler


class TestAtomicNumberTable:
    def test_conversion(self):
        table = AtomicNumberTable(zs=[1, 8])
        array = np.array([8, 8, 1])
        indices = atomic_numbers_to_indices(array, z_table=table)
        expected = np.array([1, 1, 0], dtype=int)
        assert np.allclose(expected, indices)


class TestConversions:
    def test_conversion(self):
        config = Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array([
                [0.0, -2.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            forces=np.array([
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.1, 0.3],
            ]),
            energy=-1.5,
        )
        assert config


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))


class TestStateIO:
    def test_save_load(self):
        model = MyModel()
        initial_lr = 0.001
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

        directory = tempfile.TemporaryDirectory()

        handler = CheckpointHandler(directory=directory.name, tag='test', keep=True)
        handler.save(state=CheckpointState(model, optimizer, scheduler), epochs=50)

        optimizer.step()
        scheduler.step()
        assert not np.isclose(optimizer.param_groups[0]['lr'], initial_lr)

        handler.load_latest(state=CheckpointState(model, optimizer, scheduler))
        assert np.isclose(optimizer.param_groups[0]['lr'], initial_lr)
