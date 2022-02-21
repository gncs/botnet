from typing import Dict, Any

colors = [
    '#1f77b4',  # muted blue
    '#d62728',  # brick red
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
]

dashed = (0, (5, 1))
dotted = (0, (1, 1))

style_dict: Dict[str, Dict[str, Any]] = {
    'dft': {
        'color': 'black',
        'label': 'DFT',
        'linestyle': dotted,
    },
    'nequip': {
        'color': colors[1],
        'label': 'NequIP',
    },
    'nequip-linear': {
        'color': colors[3],
        'label': 'NequIP Linear',
        'linestyle': dashed,
    },
    'nequip-shifted': {
        'color': colors[4],
        'label': 'NequIP Shift',
    },
    'nequip-bo': {
        'color': colors[5],
        'label': 'NequIP BO',
    },
    'nequip-e0': {
        'color': colors[3],
        'label': 'NequIP E0',
    },
    'nequip-ssh': {
        'color': colors[4],
        'label': 'NequIP SSH',
        'linestyle': dashed,
    },
    'nequip-e0-ssh': {
        'color': colors[1],
        'label': 'NequIP E0 SSH',
        'linestyle': dashed,
    },
    'nequip-nosc-e0': {
        'color': colors[5],
        'label': 'NequIP NoSC E0',
    },
    'nequip-nosc-e0-ssh': {
        'color': colors[5],
        'label': 'NequIP NoSC E0 SSH',
        'linestyle': dashed,
    },
    'botnet': {
        'color': colors[0],
        'label': 'BOTNet',
    },
    'botnet-e0': {
        'color': colors[0],
        'label': 'BOTNet E0',
    },
    'botnet-ssh': {
        'color': colors[8],
        'label': 'BOTNet-SSH',
        'linestyle': dashed,
    },
    'ace': {
        'color': colors[2],
        'label': 'linACE',
    }
}
