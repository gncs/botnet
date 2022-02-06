import numpy as np
import torch
import torch_geometric
import torch_scatter

from botnet.data import Configuration, AtomicData
from botnet.modules import PolynomialCutoff, AtomicEnergiesBlock, BesselBasis
from botnet.tools import AtomicNumberTable, to_numpy

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


def test_bessel_basis():
    d = torch.linspace(start=0.5, end=5.5, steps=10)
    bessel_basis = BesselBasis(r_max=6.0, num_basis=5)
    output = bessel_basis(d.unsqueeze(-1))
    assert output.shape == (10, 5)


def test_polynomial_cutoff():
    d = torch.linspace(start=0.5, end=5.5, steps=10)
    cutoff_fn = PolynomialCutoff(r_max=5.0)
    output = cutoff_fn(d)
    assert output.shape == (10, )


def test_atomic_energies():
    energies_block = AtomicEnergiesBlock(atomic_energies=np.array([1.0, 3.0]))

    data = AtomicData.from_config(config, z_table=table, cutoff=3.0)
    data_loader = torch_geometric.data.DataLoader(
        dataset=[data, data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))

    energies = energies_block(batch.node_attrs)
    out = torch_scatter.scatter(src=energies, index=batch.batch, dim=-1, reduce='sum')
    out = to_numpy(out)
    assert np.allclose(out, np.array([5., 5.]))
