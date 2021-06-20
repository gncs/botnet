import numpy as np
import torch.nn
from e3nn import o3
from torch_scatter import scatter

from e3nnff.atomic_data import AtomicData
from e3nnff.nn.modules import AtomicEnergiesBlock, InteractionBlock, EdgeEmbeddingBlock
from e3nnff.utils import TensorDict


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

        self.edge_embedding_block = EdgeEmbeddingBlock(max_ell=max_ell,
                                                       r_max=r_max,
                                                       num_bessel=num_bessel,
                                                       num_polynomial_cutoff=num_polynomial_cutoff)

        node_attr_irreps = o3.Irreps(f'{num_channels_input}x0e')
        node_embed_irreps = o3.Irreps(f'{num_channels_hidden}x0e')
        self.node_embedding = o3.Linear(node_attr_irreps, node_embed_irreps, internal_weights=True)

        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        # interaction_fns = [InteractionBlock(node_embed_irreps, node_attr_irreps, edge_irreps, )]
        # interaction_fns += [] * (num_interactions - 1)

    def forward(self, data: AtomicData) -> TensorDict:
        # Radial component
        edge_embedding = self.edge_embedding_block(data.edge_vectors, data.edge_lengths)

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter(node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs, reduce='sum')

        # Node embedding
        node_embed = self.node_embedding(data.node_attrs)

        return {
            'e0': e0,
        }
