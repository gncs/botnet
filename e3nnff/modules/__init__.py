from typing import Callable, Dict, Type

from .blocks import (AtomicEnergiesBlock, RadialEmbeddingBlock, LinearReadoutBlock, SimpleInteractionBlock,
                     ElementDependentInteractionBlock, InteractionBlock, NonlinearInteractionBlock,
                     NonLinearReadoutBlock)
from .loss import EnergyForcesLoss, ACELoss, WeightedEnergyForcesLoss
from .models import (BodyOrderedModel, ScaleShiftBodyOrderedModel, SingleReadoutModel,
                     ScaleShiftNonLinearBodyOrderedModel)
from .radial import BesselBasis, PolynomialCutoff
from .utils import compute_mean_std_atomic_inter_energy, compute_mean_rms_energy_forces

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    'SimpleInteractionBlock': SimpleInteractionBlock,
    'ElementDependentInteractionBlock': ElementDependentInteractionBlock,
    'NonlinearInteractionBlock': NonlinearInteractionBlock,
}

scaling_classes: Dict[str, Type[Callable]]  = {
    'std_scaling': compute_mean_std_atomic_inter_energy,
    'rms_forces_scaling': compute_mean_rms_energy_forces,
}

__all__ = [
    'AtomicEnergiesBlock', 'RadialEmbeddingBlock', 'LinearReadoutBlock', 'SimpleInteractionBlock', 'PolynomialCutoff',
    'BesselBasis', 'EnergyForcesLoss', 'ACELoss', 'WeightedEnergyForcesLoss', 'interaction_classes', 'InteractionBlock',
    'BodyOrderedModel', 'ScaleShiftBodyOrderedModel', 'SingleReadoutModel', 'compute_mean_std_atomic_inter_energy'
]
