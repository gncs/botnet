from typing import Dict, Type

from .blocks import (AtomicEnergiesBlock, RadialEmbeddingBlock, LinearReadoutBlock, SimpleInteractionBlock,
                     ElementDependentInteractionBlock, InteractionBlock, NonlinearInteractionBlock)
from .loss import EnergyForcesLoss, ACELoss, WeightedEnergyForcesLoss
from .models import BodyOrderedModel, ScaleShiftBodyOrderedModel, SingleReadoutModel
from .radial import BesselBasis, PolynomialCutoff
from .utils import compute_mean_std_atomic_inter_energy

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    'SimpleInteractionBlock': SimpleInteractionBlock,
    'ElementDependentInteractionBlock': ElementDependentInteractionBlock,
    'NonlinearInteractionBlock': NonlinearInteractionBlock,
}

__all__ = [
    'AtomicEnergiesBlock', 'RadialEmbeddingBlock', 'LinearReadoutBlock', 'SimpleInteractionBlock', 'PolynomialCutoff',
    'BesselBasis', 'EnergyForcesLoss', 'ACELoss', 'WeightedEnergyForcesLoss', 'interaction_classes', 'InteractionBlock',
    'BodyOrderedModel', 'ScaleShiftBodyOrderedModel', 'SingleReadoutModel', 'compute_mean_std_atomic_inter_energy'
]
