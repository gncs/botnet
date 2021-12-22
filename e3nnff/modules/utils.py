from typing import Tuple

import numpy as np
import torch
import torch.nn
import torch.utils.data
from torch_scatter import scatter_sum

from e3nnff.tools import to_numpy
from .blocks import AtomicEnergiesBlock


def get_edge_vectors_and_lengths(
        positions: torch.Tensor,  # [n_nodes, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
        shifts: torch.Tensor,  # [n_edges, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender, receiver = edge_index
    # From ase.neighborlist:
    # D = positions[j]-positions[i]+S.dot(cell)
    # where shifts = S.dot(cell)
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    return vectors, lengths


def compute_forces(energy: torch.Tensor, positions: torch.Tensor, training=True) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=energy,  # [n_graphs, ]
        inputs=positions,  # [n_nodes, 3]
        grad_outputs=torch.ones_like(energy),
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        only_inputs=True,  # Diff only w.r.t. inputs
        allow_unused=False,
    )[0]  # [n_nodes, 3]

    return -1 * gradient


def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs)
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append((batch.energy - graph_e0s) / graph_sizes)  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()

    return mean, std


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs)
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append((batch.energy - graph_e0s) / graph_sizes)  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    return mean, rms
