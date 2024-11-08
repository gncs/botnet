from typing import Callable, Dict, Type

from .blocks import (AtomicEnergiesBlock, RadialEmbeddingBlock, LinearReadoutBlock, SimpleInteractionBlock,
                     ElementDependentInteractionBlock, InteractionBlock, NonlinearInteractionBlock,
                     NonLinearReadoutBlock, AgnosticNonlinearInteractionBlock, ResidualElementDependentInteractionBlock,
                     AgnosticResidualNonlinearInteractionBlock, NequIPInteractionBlock,
                     AgnosticNoScNonlinearInteractionBlock)
from .loss import EnergyForcesLoss, ACELoss, WeightedEnergyForcesLoss
from .models import (BodyOrderedModel, ScaleShiftBodyOrderedModel, SingleReadoutModel,
                     ScaleShiftNonLinearBodyOrderedModel, ScaleShiftSingleReadoutModel,
                     ScaleShiftNonLinearSingleReadoutModel, NonLinearBodyOrderedModel)
from .radial import BesselBasis, PolynomialCutoff
from .utils import compute_mean_std_atomic_inter_energy, compute_mean_rms_energy_forces, compute_avg_num_neighbors

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    'SimpleInteractionBlock': SimpleInteractionBlock,
    'ElementDependentInteractionBlock': ElementDependentInteractionBlock,
    'ResidualElementDependentInteractionBlock': ResidualElementDependentInteractionBlock,
    'NonlinearInteractionBlock': NonlinearInteractionBlock,
    'AgnosticNonlinearInteractionBlock': AgnosticNonlinearInteractionBlock,
    'AgnosticNoScNonlinearInteractionBlock': AgnosticNoScNonlinearInteractionBlock,
    'AgnosticResidualNonlinearInteractionBlock': AgnosticResidualNonlinearInteractionBlock,
    'NequIPInteractionBlock': NequIPInteractionBlock,
}

scaling_classes: Dict[str, Callable] = {
    'std_scaling': compute_mean_std_atomic_inter_energy,
    'rms_forces_scaling': compute_mean_rms_energy_forces,
}

__all__ = [
    'AtomicEnergiesBlock', 'RadialEmbeddingBlock', 'LinearReadoutBlock', 'SimpleInteractionBlock', 'PolynomialCutoff',
    'AgnosticNoScNonlinearInteractionBlock', 'BesselBasis', 'EnergyForcesLoss', 'ACELoss', 'WeightedEnergyForcesLoss',
    'interaction_classes', 'InteractionBlock', 'BodyOrderedModel', 'ScaleShiftBodyOrderedModel', 'SingleReadoutModel',
    'ScaleShiftSingleReadoutModel', 'ScaleShiftNonLinearSingleReadoutModel', 'NonLinearBodyOrderedModel',
    'ScaleShiftNonLinearBodyOrderedModel', 'compute_mean_std_atomic_inter_energy', 'compute_avg_num_neighbors'
]
