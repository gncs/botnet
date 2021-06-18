from typing import Tuple

import ase
import ase.geometry
import ase.neighborlist
import numpy as np


def get_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # No PBC for now
    pbc = (False, False, False)
    cell = np.identity(3, dtype=float)

    first_index, second_index, shifts = ase.neighborlist.primitive_neighbor_list(
        quantities='ijS',
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        self_interaction=True,  # we want edges from atom to itself in different periodic images
        use_scaled_positions=False,
    )

    if not self_interaction:
        # Eliminate true self-edges that don't cross periodic boundaries
        bad_edge = first_index == second_index
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge

        if not np.any(keep_edge):
            raise ValueError('After eliminating self-edges, no edges remain in this system')

        first_index = first_index[keep_edge]
        second_index = second_index[keep_edge]
        shifts = shifts[keep_edge]

    # Build output:
    edge_index = np.stack((first_index, second_index))  # [2, n_edges]

    return edge_index, shifts, cell
