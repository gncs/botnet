import torch
from torch_geometric.data import Batch

from e3nnff.tools import TensorDict


def mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    return torch.mean(torch.square(ref['energy'] - pred['energy']))  # []


def mean_squared_error_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_atoms, 3]
    return torch.mean(torch.square(ref['forces'] - pred['forces']))  # []


class EnergyLoss(torch.nn.Module):
    @staticmethod
    def forward(ref: Batch, pred: TensorDict) -> torch.Tensor:
        return mean_squared_error_energy(ref, pred)


class EnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer('energy_weight', torch.tensor(energy_weight, dtype=torch.get_default_dtype()))
        self.register_buffer('forces_weight', torch.tensor(forces_weight, dtype=torch.get_default_dtype()))

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (self.energy_weight * mean_squared_error_energy(ref, pred) +
                self.forces_weight * mean_squared_error_forces(ref, pred))
