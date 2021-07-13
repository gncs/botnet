from typing import Tuple

import torch.nn


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
        only_inputs=True,  # Diff only w.r.t. inputs
        retain_graph=training,  # Make sure the graph is not destroyed
        create_graph=True,
        allow_unused=False,
    )[0]  # [n_nodes, 3]

    return -1 * gradient
