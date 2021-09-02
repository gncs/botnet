from typing import Dict, Any, Type

import numpy as np
import torch.nn
from e3nn import o3
from torch_scatter import scatter_sum

from e3nnff.data import AtomicData
from .blocks import AtomicEnergiesBlock, RadialEmbeddingBlock, LinearReadoutBlock, InteractionBlock
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
        include_forces=True,
    ):
        super().__init__()

        # Embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization='component')

        node_attr_irreps = o3.Irreps(f'{num_elements}x0e')

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_attr_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=o3.Irreps(f'{self.radial_embedding.out_dim}x0e'),
            target_irreps=hidden_irreps,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for _ in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=o3.Irreps(f'{self.radial_embedding.out_dim}x0e'),
                target_irreps=hidden_irreps,
            )
            self.interactions.append(inter)
            self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        self.include_forces = include_forces

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        if self.include_forces:
            data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Embeddings
        vectors, lengths = get_edge_vectors_and_lengths(positions=data.positions,
                                                        edge_index=data.edge_index,
                                                        shifts=data.shifts)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        node_feats = data.node_attrs

        # Interactions
        node_energies = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)
            node_energies.append(readout(node_feats).squeeze(-1))

        # Sum over interactions
        node_inter = torch.sum(torch.stack(node_energies, dim=0), dim=0)  # [n_nodes, ]

        # Sum over nodes
        inter_energy = scatter_sum(src=node_inter, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        total_energy = e0 + inter_energy

        output = {
            'energy': total_energy,
            'interaction_energy': inter_energy,
        }

        if self.include_forces:
            output['forces'] = compute_forces(energy=total_energy, positions=data.positions, training=training)

        return output
