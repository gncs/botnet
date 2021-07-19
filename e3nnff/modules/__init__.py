from typing import Dict, Type

from .blocks import (AtomicEnergiesBlock, SkipInteractionBlock, RadialEmbeddingBlock, LinearReadoutBlock,
                     ScaleShiftBlock, SimpleInteractionBlock, ElementDependentInteractionBlock, InteractionBlock)
from .loss import EnergyForcesLoss, EnergyLoss
from .radial import BesselBasis, PolynomialCutoff
from .utils import compute_mean_std_atomic_inter_energy

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    'SkipInteractionBlock': SkipInteractionBlock,
    'SimpleInteractionBlock': SimpleInteractionBlock,
    'ElementDependentInteractionBlock': ElementDependentInteractionBlock,
}

__all__ = [
    'AtomicEnergiesBlock', 'SkipInteractionBlock', 'RadialEmbeddingBlock', 'LinearReadoutBlock', 'ScaleShiftBlock',
    'SimpleInteractionBlock', 'PolynomialCutoff', 'BesselBasis', 'EnergyForcesLoss', 'EnergyLoss',
    'compute_mean_std_atomic_inter_energy', 'interaction_classes', 'InteractionBlock'
]
