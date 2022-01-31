from typing import Callable, Dict, Type

from .blocks import (AtomicEnergiesBlock, RadialEmbeddingBlock, LinearReadoutBlock, SimpleInteractionBlock,
                     ElementDependentInteractionBlock, InteractionBlock, NonlinearInteractionBlock,
                     NonLinearReadoutBlock, AgnosticNonlinearInteractionBlock, AgnosticResidualNonlinearInteractionBlock, NequIPInteractionBlock)
from .loss import EnergyForcesLoss, ACELoss, WeightedEnergyForcesLoss
from .models import (BodyOrderedModel, ScaleShiftBodyOrderedModel, SingleReadoutModel,
                     ScaleShiftNonLinearBodyOrderedModel, ScaleShiftSingleReadoutModel, ScaleShiftNonLinearSingleReadoutModel)
from .radial import BesselBasis, PolynomialCutoff
from .utils import compute_mean_std_atomic_inter_energy, compute_mean_rms_energy_forces, compute_average_neigbhors

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    'SimpleInteractionBlock': SimpleInteractionBlock,
    'ElementDependentInteractionBlock': ElementDependentInteractionBlock,
    'NonlinearInteractionBlock': NonlinearInteractionBlock,
    'AgnosticNonlinearInteractionBlock': AgnosticNonlinearInteractionBlock,
    'AgnosticResidualNonlinearInteractionBlock': AgnosticResidualNonlinearInteractionBlock,
    'NequIPInteractionBlock': NequIPInteractionBlock,
}

scaling_classes: Dict[str, Type[Callable]]  = {
    'std_scaling': compute_mean_std_atomic_inter_energy,
    'rms_forces_scaling': compute_mean_rms_energy_forces,
}

__all__ = [
    'AtomicEnergiesBlock', 'RadialEmbeddingBlock', 'LinearReadoutBlock', 'SimpleInteractionBlock', 'PolynomialCutoff',
    'BesselBasis', 'EnergyForcesLoss', 'ACELoss', 'WeightedEnergyForcesLoss', 'interaction_classes', 'InteractionBlock',
    'BodyOrderedModel', 'ScaleShiftBodyOrderedModel', 'SingleReadoutModel', 'ScaleShiftSingleReadoutModel', 'ScaleShiftNonLinearSingleReadoutModel',
    'compute_mean_std_atomic_inter_energy',
]