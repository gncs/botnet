from typing import Union

import e3nn.nn
import numpy as np
import torch
import torch.nn.functional
from e3nn import o3, nn
from torch_scatter import scatter_sum

from .cutoff import PolynomialCutoff
from .nonlinearities import ShiftedSoftPlus
from .radial_basis import BesselBasis
from .irreps_tools import tp_out_irreps_with_instructions, linear_out_irreps


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
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


class NonlinearTensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_feats_in: int, num_feats_out: int):
        super().__init__()

        input_dims = num_feats_in + 2 * num_elements
        self.mlp = e3nn.nn.FullyConnectedNet([input_dims, input_dims, num_feats_out], act=torch.nn.functional.relu)

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        sender, receiver = edge_index
        x = torch.cat([edge_feats, node_attrs[sender], node_attrs[receiver]], dim=-1)
        return self.mlp(x)


class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_feats_in: int, num_feats_out: int):
        super().__init__()

        weights1 = torch.empty((num_elements, num_elements, num_feats_in, num_feats_in),
                               dtype=torch.get_default_dtype())
        torch.nn.init.xavier_uniform_(weights1)
        self.weights1 = torch.nn.Parameter(weights1)

        weights2 = torch.empty((num_feats_in, num_feats_out), dtype=torch.get_default_dtype())
        torch.nn.init.xavier_uniform_(weights2)
        self.weights2 = torch.nn.Parameter(weights2)

    def forward(
        self,
        node_attrs,  # assumes that the node attributes are one-hot encoded
        edge_feats,
        edge_index: torch.Tensor,
    ):
        sender, receiver = edge_index
        return torch.einsum('bn, bi, bj, ijnn, nk -> bk', edge_feats, node_attrs[sender], node_attrs[receiver],
                            self.weights1, self.weights2)


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


class SingleInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        num_node_attrs: int,
        num_edge_feats: int,
        node_feats_irreps: o3.Irreps,  # [n_nodes, irreps]
        edge_attrs_irreps: o3.Irreps,  # [n_edges, sph]
        out_irreps: o3.Irreps,
    ) -> None:
        super().__init__()

        self.irreps_out = out_irreps

        self.conv_tp = o3.FullyConnectedTensorProduct(node_feats_irreps,
                                                      edge_attrs_irreps,
                                                      self.irreps_out,
                                                      shared_weights=False,
                                                      internal_weights=False)

        self.tp_weights_fn = NonlinearTensorProductWeightsBlock(num_elements=num_node_attrs,
                                                                num_feats_in=num_edge_feats,
                                                                num_feats_out=self.conv_tp.weight_numel)

        self.linear = o3.Linear(self.irreps_out, self.irreps_out)

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

        tp_weights = self.tp_weights_fn(node_attrs=node_attrs, edge_feats=edge_feats, edge_index=edge_index)
        mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  # [n_edges, irreps]
        mji = self.linear(mji)
        return scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)  # [n_nodes, irreps]


class SkipInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,  # [n_edges, sph]
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
    ) -> None:
        super().__init__()

        # Simple linear
        self.linear_1 = o3.Linear(node_feats_irreps, node_feats_irreps, internal_weights=True, shared_weights=True)

        # Main TensorProduct
        irreps_mid1, instructions = tp_out_irreps_with_instructions(node_feats_irreps, edge_attrs_irreps, target_irreps)
        self.conv_tp = o3.TensorProduct(node_feats_irreps,
                                        edge_attrs_irreps,
                                        irreps_mid1,
                                        instructions=instructions,
                                        shared_weights=False,
                                        internal_weights=False)
        self.conv_tp_weights = nn.FullyConnectedNet(
            [edge_feats_irreps.num_irreps, edge_feats_irreps.num_irreps, self.conv_tp.weight_numel],
            ShiftedSoftPlus(),
        )

        # Main linear
        irreps_mid1 = irreps_mid1.simplify()
        irreps_mid2 = linear_out_irreps(irreps_mid1, target_irreps)
        irreps_mid2 = irreps_mid2.simplify()
        self.linear_2 = o3.Linear(irreps_mid1, irreps_mid2, internal_weights=True, shared_weights=True)

        # Skip-connection TensorProduct
        self.irreps_out, _ = tp_out_irreps_with_instructions(irreps_mid2, node_attrs_irreps, target_irreps)
        self.irreps_out.simplify()
        self.skip_tp = o3.FullyConnectedTensorProduct(irreps_mid2, node_attrs_irreps, self.irreps_out)

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
        node_feats = self.linear_1(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  # [n_edges, irreps]
        m = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)  # [n_nodes, irreps]
        m = self.linear_2(m)

        # Update
        x_skip = self.skip_tp(m, node_attrs)  # [n_nodes, irreps]
        return m + x_skip
