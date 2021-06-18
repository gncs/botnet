from typing import Union

import numpy as np
import torch
from e3nn import o3
from torch_scatter import scatter

from e3nnff.e3nn_tools import node_edge_combined_irreps


class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        t = torch.tensor(atomic_energies, dtype=torch.get_default_dtype())  # [n_elements, ]
        self.register_buffer('atomic_energies', t)

    def forward(
            self,
            x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps_in: o3.Irreps,
        node_attrs_irreps_in: o3.Irreps,
        edge_feats_irreps_in: o3.Irreps,
        node_feats_irreps_out: o3.Irreps,
    ) -> None:
        super().__init__()

        mid_irreps = node_edge_combined_irreps(node_feats_irreps_in, edge_feats_irreps_in, node_feats_irreps_out)
        self.conv_tp = o3.FullyConnectedTensorProduct(node_feats_irreps_in, edge_feats_irreps_in, mid_irreps)
        self.linear = o3.Linear(mid_irreps, node_feats_irreps_out)
        self.skip_tp = o3.FullyConnectedTensorProduct(node_feats_irreps_out, node_attrs_irreps_in,
                                                      node_feats_irreps_out)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        edge_info = self.conv_tp(node_feats[sender], edge_feats)
        x = scatter(edge_info, index=receiver, dim=0, dim_size=num_nodes, reduce='sum')
        x = self.linear(x)
        x_skip = self.skip_tp(x, node_attrs)

        return x + x_skip
