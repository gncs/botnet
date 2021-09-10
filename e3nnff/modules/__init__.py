from typing import Dict, Type

from .blocks import (AtomicEnergiesBlock, RadialEmbeddingBlock, LinearReadoutBlock, SimpleInteractionBlock,
                     ElementDependentInteractionBlock, InteractionBlock)
from .loss import EnergyForcesLoss, ACELoss
from .models import BodyOrderedModel, ScaleShiftBodyOrderedModel
from .radial import BesselBasis, PolynomialCutoff
from .utils import compute_mean_std_atomic_inter_energy

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    'SimpleInteractionBlock': SimpleInteractionBlock,
    'ElementDependentInteractionBlock': ElementDependentInteractionBlock,
}

__all__ = [
    'AtomicEnergiesBlock', 'RadialEmbeddingBlock', 'LinearReadoutBlock', 'SimpleInteractionBlock', 'PolynomialCutoff',
    'BesselBasis', 'EnergyForcesLoss', 'ACELoss', 'interaction_classes', 'InteractionBlock', 'BodyOrderedModel',
    'ScaleShiftBodyOrderedModel', 'compute_mean_std_atomic_inter_energy'
]
