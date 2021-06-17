import numpy as np
import pytest
import torch

from e3nnff.keys import ATOMIC_NUMBERS_KEY

from e3nnff.data import Configuration
from e3nnff.tensor_tools import config_to_tensor_dict, atomic_numbers_to_one_hot
from e3nnff.utils import AtomicNumberTable


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

        tensor_dict = config_to_tensor_dict(config)

        with pytest.raises(Exception):
            table = AtomicNumberTable(zs=[1, 6])
            atomic_numbers_to_one_hot(tensor_dict[ATOMIC_NUMBERS_KEY], table=table)

        table = AtomicNumberTable(zs=[1, 8])
        one_hot = atomic_numbers_to_one_hot(tensor_dict[ATOMIC_NUMBERS_KEY], table=table)
        expected = torch.tensor([[0., 1.], [1., 0.], [1., 0.]])
        assert torch.allclose(input=one_hot, other=expected)
