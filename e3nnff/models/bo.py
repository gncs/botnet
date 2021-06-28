from typing import Dict, Any, Tuple

import numpy as np
import torch.nn
from e3nn import o3
from torch_scatter import scatter_sum

from e3nnff import tools
from e3nnff.data import AtomicData
from e3nnff.modules import (AtomicEnergiesBlock, SkipInteractionBlock, RadialEmbeddingBlock, LinearReadoutBlock,
                            ScaleShiftBlock)


def get_edge_vectors_and_lengths(
        positions: torch.Tensor,  # [n_nodes, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
        shifts: torch.Tensor,  # [n_edges, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender, receiver = edge_index
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1)  # [n_edges]
    return vectors, lengths


def compute_forces(energy: torch.Tensor, positions: torch.Tensor, training=True) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=energy,  # [n_graphs, ]
        inputs=positions,  # [n_nodes, 3]
        grad_outputs=torch.ones_like(energy),
        only_inputs=True,  # Diff only w.r.t. inputs
        retain_graph=training,  # Make sure the graph is not destroyed
        allow_unused=False,
    )[0]  # [n_nodes, 3]

    return -1 * gradient


class BodyOrderedModel(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        num_channels_input: int,
        hidden_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
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

        node_attr_irreps = o3.Irreps(f'{num_channels_input}x0e')
        num_e0_channels = tools.get_num_e0_channels(hidden_irreps)
        node_embed_irreps = o3.Irreps(f'{num_e0_channels}x0e')
        self.node_embedding = o3.Linear(node_attr_irreps, node_embed_irreps, internal_weights=True)

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = SkipInteractionBlock(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_embed_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=self.radial_embedding.irreps_out,
            out_irreps=hidden_irreps,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for _ in range(num_interactions - 1):
            inter = SkipInteractionBlock(
                node_feats_irreps=inter.irreps_out,
                node_attrs_irreps=node_attr_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=self.radial_embedding.irreps_out,
                out_irreps=hidden_irreps,
            )
            self.interactions.append(inter)
            self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        self.scale_shift = ScaleShiftBlock(scale=1.0, shift=0.0)

    def forward(self, data: AtomicData, training=False, include_forces=True) -> Dict[str, Any]:
        # Setup
        if include_forces:
            data.positions.requires_grad = True

        # Atomic energies
        node_energies = [self.atomic_energies_fn(data.node_attrs)]

        # Embeddings
        vectors, lengths = get_edge_vectors_and_lengths(positions=data.positions,
                                                        edge_index=data.edge_index,
                                                        shifts=data.shifts)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        node_feats = self.node_embedding(data.node_attrs)

        # Interactions
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)
            node_energy = readout(node_feats)
            node_energies.append(self.scale_shift(node_energy).squeeze(-1))

        # Compute graph energies
        energies = [
            scatter_sum(src=node_energy, index=data.batch, dim=-1, dim_size=data.num_graphs)  # sum over nodes
            for node_energy in node_energies
        ]

        total_energy = torch.sum(torch.stack(energies, dim=0), dim=0)

        output = {
            'energy': total_energy,
            'energies': energies,
        }

        if include_forces:
            output['forces'] = compute_forces(energy=total_energy, positions=data.positions, training=training)

        return output
