import numpy as np
import torch


class ShiftedSoftPlus(torch.nn.Module):
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self.log2 = np.log(2.0)

    def forward(self, x):
        return self.softplus(x) - self.log2
