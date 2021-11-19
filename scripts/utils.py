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

style_dict = {
    'dft': {
        'color': 'black',
        'label': 'DFT',
    },
    'nequip': {
        'color': colors[1],
        'label': 'NequIP',
        'linestyle': dashed,
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
    'botnet': {
        'color': colors[0],
        'label': 'BOTNet',
    },
    'botnet-ssh': {
        'color': colors[0],
        'label': 'BOTNet-SSH',
        'linestyle': dashed,
    },
    'ace': {
        'color': colors[2],
        'label': 'linACE',
    }
}
