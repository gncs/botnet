import argparse
from typing import Tuple

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from utils import style_dict

plt.rcParams.update({'font.size': 6})

opt_pos = np.array([[ 0.00832539, -0.0818815,  -0.48916197],
 [ 0.50161103,  1.19423113,  0.18777597],
 [-0.83937646, -0.86183849,  0.33615192],
 [-0.59614022, 0.1767398 , -1.38445053],
 [ 0.88248886,-0.67200338, -0.86504484],
 [ 1.14055339,  0.96085354,  1.06685176],
 [ 1.1082602 ,  1.81342654, -0.50510823],
 [-0.35458929,  1.802107 ,   0.54342772],
 [-0.31917678 ,-1.115707  ,  1.12065934]]).flatten()
 
nm874 = np.array([[ 0.0090944, -0.0347656, -0.0836763],
                [ 0.0849409,  0.1252834, -0.0138433],
                [-0.0609697, -0.0697392,  0.0465933],
                [-0.0550655, -0.1223949, -0.0622117],
                [-0.0649261, -0.1489499, -0.0499826],
                [-0.1089109, -0.0452769,  0.0848378],
                [ 0.1408352,  0.4660962,  0.339914 ],
                [-0.0544728, -0.1141104,  0.0723458],
                [-0.0101275, -0.0071352,  0.0377034]]).flatten()
norm874 = np.linalg.norm(nm874)


nm3005 = np.array([[-5.179180e-02,  2.430730e-02, -5.683990e-02],
       [ 9.310500e-03, -1.498160e-02,  1.326170e-02],
       [ 1.589600e-03,  1.274000e-04,  7.249000e-04],
       [ 5.015432e-01, -2.133678e-01,  7.274081e-01],
       [ 1.066781e-01, -8.137340e-02, -5.556380e-02],
       [-8.356730e-02,  3.236070e-02, -1.138591e-01],
       [ 8.048020e-02,  7.334830e-02, -8.891410e-02],
       [-1.111322e-01,  7.494960e-02,  4.920480e-02],
       [-1.309240e-02,  9.436000e-04, -1.050630e-02]]).flatten()
norm3005 = np.linalg.norm(nm3005)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_configs', required=True)
    parser.add_argument('--removal_configs', action='append', required=True)
    parser.add_argument('--soft_configs', action='append', required=True)
    parser.add_argument('--hard_configs', action='append', required=True)
    return parser.parse_args()


def parse_train_configs(path: str) -> pd.DataFrame:
    atoms_list = ase.io.read(path, format='extxyz', index=':')
    return pd.DataFrame({
        'distance': [atoms.info['OH_dist'] for atoms in atoms_list],
        'positions': [atoms.get_positions().flatten() for atoms in atoms_list],
    })


def parse_removal_configs(parse_removal_path: str) -> Tuple[str, pd.DataFrame]:
    name, *paths = parse_removal_path.split(',')
    frames = pd.DataFrame([{
        'distance': atoms.info['OH_dist'],
        'energy': atoms.info['energy'],
    } for path in paths for atoms in ase.io.read(path, format='extxyz', index=':')])

    df = pd.DataFrame(frames).groupby(['distance']).aggregate(
        mean_energy=pd.NamedAgg(column='energy', aggfunc='mean'),
        std_energy=pd.NamedAgg(column='energy', aggfunc='std'),
    ).reset_index()

    return name, df

def parse_displacement_path(name_path_tuple: str) -> Tuple[str, pd.DataFrame]:
    name, *paths = name_path_tuple.split(',')
    frames = pd.DataFrame([{
        'displacement': atoms.info['displacement'],
        'energy': atoms.info['energy'],
    } for path in paths for atoms in ase.io.read(path, format='extxyz', index=':')])

    df = pd.DataFrame(frames).groupby(['displacement']).aggregate(
        mean_energy=pd.NamedAgg(column='energy', aggfunc='mean'),
        std_energy=pd.NamedAgg(column='energy', aggfunc='std'),
    ).reset_index()

    return name, df


def main():
    args = parse_args()
    training = parse_train_configs(args.train_configs)
    removal_predictions = [parse_removal_configs(path) for path in args.removal_configs]
    soft_predictions = [parse_displacement_path(path) for path in args.soft_configs]
    hard_predictions = [parse_displacement_path(path) for path in args.hard_configs]

    fig, axes = plt.subplots(nrows=2,
                             ncols=3,
                             figsize=(6.0, 2.6),
                             constrained_layout=True,
                             sharey='row',
                             sharex='col',
                             gridspec_kw={'height_ratios': (5, 1)})

    # Removal energies
    ax = axes[0, 0]
    for index, (name, df) in enumerate(removal_predictions):
        selection = df[df['distance'] <= 5.0]
        ax.plot(selection['distance'], selection['mean_energy'], **style_dict[name])
        ax.fill_between(
            x=selection['distance'],
            y1=selection['mean_energy'] - selection['std_energy'],
            y2=selection['mean_energy'] + selection['std_energy'],
            alpha=0.3,
            zorder=2 * index,
            **style_dict[name],
        )

    ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.set_title('Hydrogen Abstraction')
    ax.set_ylabel(r'$E$ [eV]')
    

    # Removal histogram
    ax = axes[1, 0]
    d_range = np.arange(0.95, 5.0, step=0.025)
    ax.hist(training['distance'], bins=d_range, color='black', label='Training data')
    ax.set_ylabel('Count')
    ax.set_xlabel('Distance')


    # Soft
    ax = axes[0, 1]
    for index, (name, df) in enumerate(soft_predictions):
        ax.plot(df['displacement'], df['mean_energy'], **style_dict[name])
        ax.fill_between(
            x=df['displacement'],
            y1=df['mean_energy'] - df['std_energy'],
            y2=df['mean_energy'] + df['std_energy'],
            alpha=0.3,
            zorder=2 * index,
            **style_dict[name],
        )


    ax.set_title(r'$\tilde{\nu} = 874 \, \mathrm{cm}^{-1}$')
    ax.set_xticks([-1.0, 0.0, 1.0, 2.0])

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    
    #Soft histogram
    ax = axes[1, 1]
    projections = []
    for position in training['positions'] :
        disp = position - opt_pos
        proj = np.dot(disp, nm874) / norm874**2
        projections.append(proj / (1 + np.linalg.norm(disp - proj * nm3005)))

    ax.hist(projections, color='black', label='Training data')
    ax.set_xlabel("Displacement")
    ax.set_ylabel("Count")

    # Hard
    ax = axes[0, 2]
    for index, (name, df) in enumerate(hard_predictions):
        ax.plot(-df['displacement'], df['mean_energy'], **style_dict[name])
        ax.fill_between(
            x=-df['displacement'],
            y1=df['mean_energy'] - df['std_energy'],
            y2=df['mean_energy'] + df['std_energy'],
            alpha=0.3,
            zorder=2 * index,
            **style_dict[name],
        )

    ax.set_title(r'$\tilde{\nu} = 3005 \, \mathrm{cm}^{-1}$')
    ax.set_xticks([1.0, 0.5, 0.0, -0.5, -1.0])

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    #Hard histogram
    ax = axes[1, 2]
    projections = []
    for position in training['positions'] :
        disp = opt_pos - position
        proj = np.dot(disp, nm3005) / norm3005**2
        projections.append(proj / (1 + np.linalg.norm(disp - proj * nm3005)))

    ax.hist(projections, color='black', label='Training data')
    ax.set_xlabel("Displacement")
    ax.set_ylabel("Count")

    fig.savefig('ethanol_proj.pdf',bbox_inches='tight')


if __name__ == '__main__':
    main()
