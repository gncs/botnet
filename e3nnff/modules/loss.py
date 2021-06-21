import torch
from torch_geometric.data import Batch

from e3nnff.tools import TensorDict


def mean_squared_error_energy(batch: Batch, predictions: TensorDict) -> torch.Tensor:
    return torch.mean(torch.square(batch['energy'] - predictions['energy']))


class EnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer('energy_weight', torch.tensor(energy_weight, dtype=torch.get_default_dtype()))
        self.register_buffer('forces_weight', torch.tensor(forces_weight, dtype=torch.get_default_dtype()))

    def forward(self, batch: Batch, predictions: TensorDict) -> torch.Tensor:
        return mean_squared_error_energy(batch, predictions)
