from typing import Union

import numpy as np
import torch


class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        t = torch.tensor(atomic_energies, dtype=torch.get_default_dtype())  # [n_elements, ]
        self.register_buffer('atomic_energies', t)

    def forward(
            self,
            x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)
