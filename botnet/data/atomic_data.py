from typing import Optional

import torch.utils.data
import torch_geometric

from botnet.tools import to_one_hot, AtomicNumberTable, atomic_numbers_to_indices
from .neighborhood import get_neighborhood
from .utils import Configuration


class AtomicData(torch_geometric.data.Data):
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor

    def __init__(
            self,
            edge_index: torch.Tensor,  # [2, n_edges]
            node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
            positions: torch.Tensor,  # [n_nodes, 3]
            shifts: torch.Tensor,  # [n_edges, 3], real shifts in X, Y, Z
            forces: Optional[torch.Tensor],  # [n_nodes, 3]
            energy: Optional[torch.Tensor],  # [, ]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert len(node_attrs.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0

        # Aggregate data
        data = {
            'num_nodes': num_nodes,
            'edge_index': edge_index,
            'node_attrs': node_attrs,
            'positions': positions,
            'shifts': shifts,
            'forces': forces,
            'energy': energy,
        }
        super().__init__(**data)

    @classmethod
    def from_config(cls, config: Configuration, z_table: AtomicNumberTable, cutoff: float) -> 'AtomicData':
        indices = atomic_numbers_to_indices(config.atomic_numbers, z_table=z_table)
        one_hot_attrs = to_one_hot(
            indices=torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )

        edge_index, shifts = get_neighborhood(positions=config.positions, cutoff=cutoff)

        # Energies and forces are optional
        forces = torch.tensor(config.forces, dtype=torch.get_default_dtype()) if config.forces is not None else None
        energy = torch.tensor(config.energy, dtype=torch.get_default_dtype()) if config.energy is not None else None

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_attrs=one_hot_attrs.to(torch.get_default_dtype()),
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            forces=forces,
            energy=energy,
        )
