import argparse
import json
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

fig_width = 2.5
fig_height = 2.1

plt.rcParams.update({'font.size': 6})

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


def parse_json_lines_file(path: str) -> List[dict]:
    dicts = []
    with open(path, mode='r') as f:
        for line in f:
            dicts.append(json.loads(line))
    return dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot e3nn-ff training statistics')
    parser.add_argument('--path', help='path to results file', required=True)
    parser.add_argument('--min_epoch', help='minimum epoch', default=50, required=False)
    return parser.parse_args()


def plot(data: pd.DataFrame, min_epoch: int) -> None:
    fig, axes = plt.subplots(nrows=2,
                             ncols=1,
                             figsize=(fig_width, 2 * fig_height),
                             constrained_layout=True,
                             sharex='col')
    valid_data = data[data['mode'] == 'eval']
    valid_data = valid_data[valid_data['epoch'] > min_epoch]

    ax = axes[0]
    ax.plot(valid_data['epoch'], valid_data['loss'], color=colors[0], label='Loss')

    ax.legend()

    ax = axes[1]
    ax.plot(valid_data['epoch'], valid_data['mae_e'], color=colors[1], label='MAE Energy [eV]')
    ax.plot(valid_data['epoch'], valid_data['mae_f'], color=colors[2], label='MAE Forces [eV/Å]')

    ax.legend()
    ax.set_xlabel('Epoch')

    fig.savefig('training.pdf')


def main():
    args = parse_args()
    data = pd.DataFrame(parse_json_lines_file(args.path))
    plot(data, min_epoch=args.min_epoch)


if __name__ == '__main__':
    main()
