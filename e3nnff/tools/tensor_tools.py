import numpy as np
import torch


def to_one_hot(indices: torch.Tensor, num_classes: int, device=None) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>

    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes, )
    oh = torch.zeros(shape, device=device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()
