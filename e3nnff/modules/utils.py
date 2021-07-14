from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter_sum

from e3nnff.tools import to_numpy
from .blocks import AtomicEnergiesBlock


def compute_mean_std_atomic_inter_energy(
    data_loader: DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs)
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es = (batch.energy - graph_e0s) / graph_sizes  # [n_graphs]
        avg_atom_inter_es_list.append(avg_atom_inter_es)

    all_avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(all_avg_atom_inter_es)).item()
    std = to_numpy(torch.std(all_avg_atom_inter_es)).item()

    return mean, std
