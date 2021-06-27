from typing import Tuple, Optional

import ase.neighborlist
import numpy as np


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,
    self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None:
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, bool) for i in pbc)
    assert cell.shape == (3, 3)

    first_index, second_index, unit_shifts = ase.neighborlist.primitive_neighbor_list(
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
        true_self_edge = first_index == second_index
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        first_index = first_index[keep_edge]
        second_index = second_index[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output:
    edge_index = np.stack((first_index, second_index))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts
