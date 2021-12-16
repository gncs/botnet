from abc import ABC, abstractmethod
from typing import Union, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional
from e3nn import nn, o3
from torch_scatter import scatter_sum

from .irreps_tools import tp_out_irreps_with_instructions, linear_out_irreps
from .radial import BesselBasis, PolynomialCutoff


class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
            self,
            node_attrs: torch.Tensor,  # [n_nodes, irreps]
    ):
        return self.linear(node_attrs)


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


class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=o3.Irreps('0e'))

    def forward(
            self,
            x: torch.Tensor  # [n_nodes, irreps]
    ) -> torch.Tensor:  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, MLP_irreps: o3.Irreps, gate: Callable):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.norm = o3.Norm(irreps_in, squared=True)
        self.linear_1 = o3.Linear(irreps_in=self.norm.irreps_out, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(irreps_in=self.hidden_irreps, irreps_out=o3.Irreps('0e'))

    def forward(
            self,
            x: torch.Tensor  # [n_nodes, irreps]
    ) -> torch.Tensor:  # [..., ]
        x = self.norm(x)
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        self.register_buffer('atomic_energies', torch.tensor(atomic_energies,
                                                             dtype=torch.get_default_dtype()))  # [n_elements, ]

    def forward(
            self,
            x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ', '.join([f'{x:.4f}' for x in self.atomic_energies])
        return f'{self.__class__.__name__}(energies=[{formatted_energies}])'


class InteractionBlock(ABC, torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class SimpleInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(self.node_feats_irreps, self.edge_attrs_irreps,
                                                                   self.target_irreps)
        self.conv_tp = o3.TensorProduct(self.node_feats_irreps,
                                        self.edge_attrs_irreps,
                                        irreps_mid,
                                        instructions=instructions,
                                        shared_weights=False,
                                        internal_weights=False)
        self.conv_tp_weights = o3.Linear(self.edge_feats_irreps, o3.Irreps(f'{self.conv_tp.weight_numel}x0e'))

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True)

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(self.irreps_out, self.node_attrs_irreps, self.irreps_out)

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

        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  # [n_edges, irreps]
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)  # [n_nodes, irreps]
        message = self.linear(message)
        return self.skip_tp(message, node_attrs)  # [n_nodes, irreps]


class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty((num_elements, num_edge_feats, num_feats_out), dtype=torch.get_default_dtype())
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: torch.Tensor,  # assumes that the node attributes are one-hot encoded
        edge_feats: torch.Tensor,
    ):
        return torch.einsum('be, ba, aek -> bk', edge_feats, sender_or_receiver_node_attrs, self.weights)

    def __repr__(self):
        return f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), ' \
               f'weights={np.prod(self.weights.shape)})'


class ElementDependentInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(self.node_feats_irreps, self.edge_attrs_irreps,
                                                                   self.target_irreps)
        self.conv_tp = o3.TensorProduct(self.node_feats_irreps,
                                        self.edge_attrs_irreps,
                                        irreps_mid,
                                        instructions=instructions,
                                        shared_weights=False,
                                        internal_weights=False)
        self.conv_tp_weights = TensorProductWeightsBlock(num_elements=self.node_attrs_irreps.num_irreps,
                                                         num_edge_feats=self.edge_feats_irreps.num_irreps,
                                                         num_feats_out=self.conv_tp.weight_numel)

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True)

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(self.irreps_out, self.node_attrs_irreps, self.irreps_out)

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

        tp_weights = self.conv_tp_weights(node_attrs[sender], edge_feats)
        mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  # [n_edges, irreps]
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)  # [n_nodes, irreps]
        message = self.linear(message)
        return self.skip_tp(message, node_attrs)  # [n_nodes, irreps]


def init_layer(layer: torch.nn.Linear, w_scale=1.0) -> torch.nn.Linear:
    torch.nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)  # type: ignore
    torch.nn.init.constant_(layer.bias.data, 0)
    return layer


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dims: Tuple[int, ...], gate: torch.nn.Module):
        super().__init__()
        self.dims = (input_dim, ) + output_dims
        self.layers = torch.nn.ModuleList(
            [init_layer(torch.nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(self.dims[:-1], self.dims[1:])])
        self.gate = gate
        self.output_dim = self.dims[-1]

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        x = self.layers[-1](x)
        return x

    def __repr__(self):
        layers_str = ", ".join(str(layer) for layer in self.layers)
        return f"{self.__class__.__name__}(layers={{" + layers_str + f"}}, act={self.gate}, " \
               f"weights={sum(layer.weight.numel() + layer.bias.numel() for layer in self.layers)})"


class NonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(self.node_feats_irreps, self.edge_attrs_irreps,
                                                                   self.target_irreps)
        self.conv_tp = o3.TensorProduct(self.node_feats_irreps,
                                        self.edge_attrs_irreps,
                                        irreps_mid,
                                        instructions=instructions,
                                        shared_weights=False,
                                        internal_weights=False)
        input_dim = self.node_attrs_irreps.num_irreps + self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = MLP(input_dim=input_dim,
                                   output_dims=(input_dim, self.conv_tp.weight_numel),
                                   gate=torch.nn.ReLU())

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True)

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(self.irreps_out, self.node_attrs_irreps, self.irreps_out)

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

        tp_weights = self.conv_tp_weights(torch.cat([node_attrs[sender], edge_feats], dim=-1))
        mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  # [n_edges, irreps]
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)  # [n_nodes, irreps]
        message = self.linear(message)
        return self.skip_tp(message, node_attrs)  # [n_nodes, irreps]


class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.get_default_dtype()))
        self.register_buffer('shift', torch.tensor(shift, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift

    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})'
