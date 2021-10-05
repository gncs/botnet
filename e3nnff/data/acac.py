import logging
import os
from typing import Dict

from .utils import Configurations, unpack_configs_from_archive

# Atomic energies (in eV)
atomic_energies = {
    1: -13.568422178253735,
    6: -1026.8538996116154,
    8: -2037.796869412825,
}


def load(directory: str) -> Dict[str, Configurations]:
    logging.info('Loading acetylacetone (acac) dataset')
    path = os.path.join(directory, 'dataset_acac.tar.gz')
    return unpack_configs_from_archive(path=path)
