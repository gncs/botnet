import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

# The positions of the atoms in the system
POSITIONS_KEY: Final[str] = 'pos'

# The [2, n_edge] index tensor giving center -> neighbor relations
EDGE_INDEX_KEY: Final[str] = 'edge_index'
# A [n_edge, 3] tensor of how many periodic cells each edge crosses in each cell vector
EDGE_CELL_SHIFT_KEY: Final[str] = 'edge_cell_shift'

# A [n_edge, 3] tensor of displacement vectors associated to edges
EDGE_VECTORS_KEY: Final[str] = 'edge_vectors'
# A [n_edge] tensor of the lengths of EDGE_VECTORS
EDGE_LENGTH_KEY: Final[str] = 'edge_lengths'
# [n_edge, dim] (possibly equivariant) attributes of each edge
EDGE_ATTRS_KEY: Final[str] = 'edge_attrs'

CELL_KEY: Final[str] = 'cell'
PBC_KEY: Final[str] = 'pbc'

ATOMIC_NUMBERS_KEY: Final[str] = 'atomic_numbers'
NODE_ATTRS_KEY: Final[str] = 'node_attrs'
NODE_FEATURES_KEY: Final[str] = 'node_features'

FORCES_KEY: Final[str] = 'forces'
ENERGY_KEY: Final[str] = 'total_energy'
