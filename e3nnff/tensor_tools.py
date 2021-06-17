import torch

from e3nnff.data import Configuration
from e3nnff.keys import POSITIONS_KEY, FORCES_KEY, ENERGY_KEY, ATOMIC_NUMBERS_KEY
from e3nnff.utils import TensorDict, AtomicNumberTable


def to_one_hot(indices: torch.Tensor, num_classes: int, device=None) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>

    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def config_to_tensor_dict(config: Configuration) -> TensorDict:
    d = {
        ATOMIC_NUMBERS_KEY: torch.tensor(config.atomic_numbers, dtype=torch.long),
        POSITIONS_KEY: torch.tensor(config.positions, dtype=torch.get_default_dtype()),
    }

    if config.forces is not None:
        d[FORCES_KEY] = torch.tensor(config.forces, dtype=torch.get_default_dtype())

    if config.energy is not None:
        d[ENERGY_KEY] = torch.tensor(config.energy, dtype=torch.get_default_dtype())

    return d


def atomic_numbers_to_one_hot(atomic_numbers: torch.Tensor, table: AtomicNumberTable) -> torch.Tensor:
    tmp = torch.clone(atomic_numbers)
    tmp.apply_(lambda z: table.z_to_index(z))
    return to_one_hot(tmp.unsqueeze(-1), num_classes=len(table))
