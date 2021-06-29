import numpy as np
import pytest
from e3nn import o3

from e3nnff.data import Configuration
from e3nnff.tools import AtomicNumberTable, atomic_numbers_to_indices, get_num_e0_channels, ev_to_hartree, \
    kcalpmol_to_hartree, angstrom_to_bohr, kcalpmol_per_angstrom_to_hartree_per_bohr


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


class TestE3NNTools:
    def test_e0_channels(self):
        irreps = o3.Irreps('4x1e + 7x0e + 2x2o')
        e0_channels = get_num_e0_channels(irreps)
        assert e0_channels == 7

        irreps = o3.Irreps('4x1e + 7x0o + 2x2o')
        with pytest.raises(RuntimeError):
            get_num_e0_channels(irreps)


class TestUnits:
    def test_standard_units(self):
        assert np.isclose(ev_to_hartree(1.0), 0.036749)
        assert np.isclose(kcalpmol_to_hartree(1.0), 0.0015936)
        assert np.isclose(angstrom_to_bohr(1.0), 1.88973)
        assert np.isclose(kcalpmol_per_angstrom_to_hartree_per_bohr(1.0), 0.0015936 / 1.88973)
