from typing import Tuple

import torch.nn
from torch_geometric.data import Batch

from e3nnff.tools import TensorDict


class EnergyForceLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, force_weight=1.0) -> None:
        super().__init__()
        self.register_buffer('energy_weight', torch.tensor(energy_weight, dtype=torch.get_default_dtype()))
        self.register_buffer('force_weight', torch.tensor(force_weight, dtype=torch.get_default_dtype()))

    def forward(self, batch: Batch, predictions: TensorDict) -> Tuple[torch.Tensor, TensorDict]:
        raise NotImplementedError
