from typing import Dict, Type

from .blocks import (AtomicEnergiesBlock, RadialEmbeddingBlock, LinearReadoutBlock, SimpleInteractionBlock,
                     ElementDependentInteractionBlock, InteractionBlock)
from .loss import EnergyForcesLoss, EnergyLoss
from .radial import BesselBasis, PolynomialCutoff

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    'SimpleInteractionBlock': SimpleInteractionBlock,
    'ElementDependentInteractionBlock': ElementDependentInteractionBlock,
}

__all__ = [
    'AtomicEnergiesBlock', 'RadialEmbeddingBlock', 'LinearReadoutBlock', 'SimpleInteractionBlock', 'PolynomialCutoff',
    'BesselBasis', 'EnergyForcesLoss', 'EnergyLoss', 'interaction_classes', 'InteractionBlock'
]
