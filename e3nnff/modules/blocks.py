from typing import Union

import numpy as np
import torch
from e3nn import o3
from torch_scatter import scatter_sum

from .cutoff import PolynomialCutoff
from .radial_basis import BesselBasis


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
            self,
            edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths).unsqueeze(-1)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale = torch.tensor(scale, dtype=torch.get_default_dtype())
        self.shift = torch.tensor(shift, dtype=torch.get_default_dtype())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift


class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=o3.Irreps('0e'))

    def forward(
            self,
            x: torch.Tensor  # [n_nodes, irreps]
    ) -> torch.Tensor:  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


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


class SkipInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,  # [n_edges, sph]
        edge_feats_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
    ) -> None:
        super().__init__()

        self.irreps_out = out_irreps

        self.conv_tp = o3.FullyConnectedTensorProduct(node_feats_irreps,
                                                      edge_attrs_irreps,
                                                      self.irreps_out,
                                                      shared_weights=False,
                                                      internal_weights=False)
        weights_irreps = o3.Irreps(f'{self.conv_tp.weight_numel}x0e')
        self.conv_tp_weights = o3.Linear(edge_feats_irreps, weights_irreps)

        self.linear_1 = o3.Linear(self.irreps_out, self.irreps_out)

        self.skip_tp = o3.FullyConnectedTensorProduct(self.irreps_out, node_attrs_irreps, self.irreps_out)
        self.linear_2 = o3.Linear(self.irreps_out, self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        # Message
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  # [n_edges, irreps]
        mji = self.linear_1(mji)
        m = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)  # [n_nodes, irreps]

        # Update
        x_skip = self.skip_tp(m, node_attrs)  # [n_nodes, irreps]
        x_skip = self.linear_2(x_skip)
        new_node_feats = m + x_skip

        return new_node_feats
