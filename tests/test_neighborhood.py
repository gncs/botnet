import numpy as np

from e3nnff.data import get_neighborhood


class TestNeighborhood:
    def test_basic(self):
        positions = np.array([
            [-1.0, 0.0, 0.0],
            [+0.0, 0.0, 0.0],
            [+1.0, 0.0, 0.0],
        ])

        indices, shifts, cell = get_neighborhood(positions, cutoff=1.5)
        assert indices.shape == (2, 4)
        assert shifts.shape == (4, 3)
        assert cell.shape == (3, 3)
