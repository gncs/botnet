import numpy as np

from e3nnff.atomic_data import AtomicData, get_data_loader
from e3nnff.data import Configuration
from e3nnff.models.bo import BondOrderModel
from e3nnff.utils import AtomicNumberTable

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

table = AtomicNumberTable([1, 8])


class TestModels:
    def test_bo_model(self):
        atomic_energies = np.array([1.0, 3.0], dtype=float)
        model = BondOrderModel(
            r_max=2.0,
            num_bessel=7,
            num_polynomial_cutoff=5,
            max_ell=4,
            num_channels_input=len(table),
            num_channels_hidden=11,
            num_interactions=3,
            atomic_energies=atomic_energies,
        )

        data = AtomicData.from_config(config, table=table, cutoff=3.0)

        data_loader = get_data_loader([data, data], batch_size=2)
        batch = next(iter(data_loader))

        energy, aux = model(batch)
        assert energy.shape == (2,)
