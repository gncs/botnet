import argparse

import ase.io
import numpy as np
from matplotlib import cm, colors
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 6})


def get_cm(color: str):
    cmap = cm.get_cmap(color, 512)
    new_colors = cmap(np.linspace(0, 1, 512))
    new_colors[0, -1] = 0.0
    return colors.ListedColormap(new_colors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='path to training configurations', required=True)
    parser.add_argument('--test', help='path to test configurations', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    training_atoms = ase.io.read(filename=args.train, format='extxyz', index=':')
    test_atoms = ase.io.read(filename=args.test, format='extxyz', index=':')

    step_size = 0.05
    bins = np.arange(0.80, 2.25 + step_size, step=step_size)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.0 + 1.0, 2.0), constrained_layout=True)

    d1s_train = np.array([atoms.get_distance(3, 11) for atoms in training_atoms])
    d2s_train = np.array([atoms.get_distance(5, 11) for atoms in training_atoms])
    train_hist = ax.hist2d(d1s_train, d2s_train, bins=bins, cmap=get_cm('Blues'))

    d1s_test = np.array([atoms.get_distance(3, 11) for atoms in test_atoms])
    d2s_test = np.array([atoms.get_distance(5, 11) for atoms in test_atoms])
    test_hist = ax.hist2d(d1s_test, d2s_test, bins=bins, cmap=get_cm('Reds'))

    major_ticks = np.arange(start=0.8, stop=2.4, step=0.2)
    ax.set_xticks(major_ticks)
    ax.set_xlabel(r'$d_1$ [Å]')
    ax.set_yticks(major_ticks)
    ax.set_ylabel(r'$d_2$ [Å]')

    test_cb = fig.colorbar(test_hist[3], ax=ax)
    test_cb.set_label('Test')
    test_cb.set_ticks(range(int(test_cb.vmin), int(test_cb.vmax) + 1))

    train_cb = fig.colorbar(train_hist[3], ax=ax)
    train_cb.set_label('Train')

    ax.set_aspect('equal', adjustable='box')

    fig.savefig('acac_hist.pdf')


if __name__ == '__main__':
    main()
