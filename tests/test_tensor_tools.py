import numpy as np

from e3nnff.data import Configuration
from e3nnff.tools import AtomicNumberTable, atomic_numbers_to_indices


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
