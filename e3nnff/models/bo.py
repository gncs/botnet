from typing import Tuple, Dict

import numpy as np
import torch.nn
from e3nn import o3
from torch_scatter import scatter

from e3nnff.atomic_data import AtomicData
from e3nnff.nn.modules import AtomicEnergiesBlock, InteractionBlock, EdgeEmbeddingBlock, ReadoutBlock, ScaleShiftBlock


class BondOrderModel(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        num_channels_input: int,
        num_channels_hidden: int,
        atomic_energies: np.ndarray,
    ):
        super().__init__()

        self.edge_embedding = EdgeEmbeddingBlock(max_ell=max_ell,
                                                 r_max=r_max,
                                                 num_bessel=num_bessel,
                                                 num_polynomial_cutoff=num_polynomial_cutoff)

        node_attr_irreps = o3.Irreps(f'{num_channels_input}x0e')
        node_embed_irreps = o3.Irreps(f'{num_channels_hidden}x0e')
        self.node_embedding = o3.Linear(node_attr_irreps, node_embed_irreps, internal_weights=True)

        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = InteractionBlock(
            max_ell=max_ell,
            num_channels=num_channels_hidden,
            node_feats_irreps=node_embed_irreps,
            node_attrs_irreps=node_attr_irreps,
            edge_feats_irreps=self.edge_embedding.irreps_out,
        )
        self.interactions = [inter]
        self.read_outs = [ReadoutBlock(inter.irreps_out)]

        for _ in range(num_interactions - 1):
            inter = InteractionBlock(max_ell=max_ell,
                                     num_channels=num_channels_hidden,
                                     node_feats_irreps=inter.irreps_out,
                                     node_attrs_irreps=node_attr_irreps,
                                     edge_feats_irreps=self.edge_embedding.irreps_out)
            self.interactions.append(inter)
            self.read_outs.append(ReadoutBlock(inter.irreps_out))

        self.scale_shift = ScaleShiftBlock(scale=1.0, shift=0.0)

    def forward(self, data: AtomicData) -> Tuple[torch.Tensor, Dict]:
        # Atomic energies
        node_energies = [self.atomic_energies_fn(data.node_attrs)]

        # Embeddings
        edge_feats = self.edge_embedding(data.edge_vectors, data.edge_lengths)
        node_feats = self.node_embedding(data.node_attrs)

        # Interactions
        for interaction, read_out in zip(self.interactions, self.read_outs):
            node_feats = interaction(node_feats, data.node_attrs, edge_feats, data.edge_index)
            node_energy = read_out(node_feats)
            node_energies.append(self.scale_shift(node_energy).squeeze(-1))

        # Compute graph energies
        energies = [
            scatter(node_energy, index=data.batch, dim=-1, dim_size=data.num_graphs, reduce='sum')
            for node_energy in node_energies
        ]

        total_energy = torch.sum(torch.stack(energies, dim=0), dim=0)

        return total_energy, {
            'energies': energies,
        }
