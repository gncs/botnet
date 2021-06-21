import logging
from typing import Dict

import numpy as np
import torch
from e3nn import o3

TensorDict = Dict[str, torch.Tensor]


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


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def get_num_e0_channels(irreps: o3.Irreps) -> int:
    for channels, (ell, p) in irreps:
        if ell == 0 and p == 1:
            return channels

    raise RuntimeError(f'Could not find e0 irrep in {irreps}')


def init_device(device_str: str) -> torch.device:
    if device_str == 'cuda':
        assert (torch.cuda.is_available()), 'No CUDA device available!'
        logging.info('CUDA Device: {}'.format(torch.cuda.current_device()))
        torch.cuda.init()
        return torch.device('cuda')
    else:
        logging.info('Using CPU')
        return torch.device('cpu')
