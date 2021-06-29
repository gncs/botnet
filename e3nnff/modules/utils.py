from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter_sum

from e3nnff.modules import AtomicEnergiesBlock
from e3nnff.tools import to_numpy


def compute_mean_std_atomic_inter_energy(
    data_loader: DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_inter_energies_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs)
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_inter_energies = (batch.energy - graph_e0s) / graph_sizes
        atom_inter_energies_list.append(atom_inter_energies)

    atom_inter_energies = torch.cat(atom_inter_energies_list)  # [n_graphs]
    mean = to_numpy(torch.mean(atom_inter_energies)).item()
    std = to_numpy(torch.std(atom_inter_energies)).item()

    return mean, std
