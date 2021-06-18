import torch

from e3nnff.nn.cutoff import PolynomialCutoff
from e3nnff.nn.radial_basis import BesselBasis


class TestBasis:
    def test_bessel_basis(self):
        d = torch.linspace(start=0.5, end=5.5, steps=10)
        bessel_basis = BesselBasis(r_max=6.0, num_basis=5)
        output = bessel_basis(d)
        assert output.shape == (10, 5)


class TestCutoff:
    def test_polynomial_cutoff(self):
        d = torch.linspace(start=0.5, end=5.5, steps=10)
        cutoff_fn = PolynomialCutoff(r_max=5.0)
        output = cutoff_fn(d)
        assert output.shape == (10, )
