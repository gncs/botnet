from typing import Dict, Any, Type

import numpy as np
import torch.nn
from e3nn import o3
from torch_scatter import scatter_sum

from e3nnff.data import AtomicData
from .blocks import (AtomicEnergiesBlock, RadialEmbeddingBlock, LinearReadoutBlock, InteractionBlock, ScaleShiftBlock,
                     LinearNodeEmbeddingBlock,NonLinearReadoutBlock)
from .utils import get_edge_vectors_and_lengths, compute_forces


class BodyOrderedModel(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
    ):
        super().__init__()

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=node_attr_irreps, irreps_out=node_feats_irreps)
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f'{self.radial_embedding.out_dim}x0e')

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization='component')

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for _ in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
            )
            self.interactions.append(inter)
            self.readouts.append(LinearReadoutBlock(inter.irreps_out))

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(positions=data.positions,
                                                        edge_index=data.edge_index,
                                                        shifts=data.shifts)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(src=node_energies, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        output = {
            'energy': total_energy,
            'contributions': contributions,
            'forces': compute_forces(energy=total_energy, positions=data.positions, training=training),
        }

        return output


class BodyOrderedModel_NonLinear(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        gate: str,
    ):
        super().__init__()

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=node_attr_irreps, irreps_out=node_feats_irreps)
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f'{self.radial_embedding.out_dim}x0e')

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization='component')

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
            )
            self.interactions.append(inter)
            if i ==  num_interactions - 2 : 
                self.readouts.append(NonLinearReadoutBlock(inter.irreps_out,gate))
            else :
                self.readouts.append(LinearReadoutBlock(inter.irreps_out))

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(positions=data.positions,
                                                        edge_index=data.edge_index,
                                                        shifts=data.shifts)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(src=node_energies, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        output = {
            'energy': total_energy,
            'contributions': contributions,
            'forces': compute_forces(energy=total_energy, positions=data.positions, training=training),
        }

        return output


class SingleReadoutModel(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
    ):
        super().__init__()

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=node_attr_irreps, irreps_out=node_feats_irreps)
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f'{self.radial_embedding.out_dim}x0e')

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization='component')

        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
        )
        self.interactions = torch.nn.ModuleList([inter])

        for _ in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
            )
            self.interactions.append(inter)

        self.readouts = torch.nn.ModuleList([LinearReadoutBlock(self.interactions[-1].irreps_out)])

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(positions=data.positions,
                                                        edge_index=data.edge_index,
                                                        shifts=data.shifts)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        for interaction in self.interactions:
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)

        node_inter_es = self.readouts[0](node_feats).squeeze(-1)  # {[n_nodes, ], }

        # Sum over nodes in graph
        inter_e = scatter_sum(src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        total_e = e0 + inter_e

        return {
            'energy': total_e,
            'forces': compute_forces(energy=total_e, positions=data.positions, training=training),
        }


class ScaleShiftBodyOrderedModel(BodyOrderedModel):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(scale=atomic_inter_scale, shift=atomic_inter_shift)

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(positions=data.positions,
                                                        edge_index=data.edge_index,
                                                        shifts=data.shifts)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)

            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            'energy': total_e,
            'forces': compute_forces(energy=total_e, positions=data.positions, training=training),
        }

        return output


class ScaleShiftBodyOrderedModel_NonLinear(BodyOrderedModel_NonLinear):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(scale=atomic_inter_scale, shift=atomic_inter_shift)

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(positions=data.positions,
                                                        edge_index=data.edge_index,
                                                        shifts=data.shifts)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)

            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            'energy': total_e,
            'forces': compute_forces(energy=total_e, positions=data.positions, training=training),
        }

        return output

