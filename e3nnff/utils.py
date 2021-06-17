import logging
import os
import sys
from typing import Tuple, Union, Optional

import scipy.constants


# noinspection PyPep8Naming
def kcal_to_kJ(x):
    return x * scipy.constants.calorie


def get_split_sizes(size: int, first_fraction: float) -> Tuple[int, int]:
    assert 0.0 < first_fraction < 1.0
    first_size = int(first_fraction * size)
    return first_size, size - first_size


def setup_logger(level: Union[int, str] = logging.INFO, tag: Optional[str] = None, directory: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + '.log')
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)
