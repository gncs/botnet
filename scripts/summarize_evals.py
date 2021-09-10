import argparse
import glob
import json
import os
from typing import List, Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory pointing to evaluation files', required=True)
    return parser.parse_args()


def get_paths(directory: dir) -> List[str]:
    return glob.glob(os.path.join(directory, '*_eval.txt'))


def read_results(path: str) -> List[Dict]:
    with open(path, mode='r') as f:
        return [json.loads(line) for line in f]


def main():
    args = parse_args()
    results_list = [result for path in get_paths(directory=args.dir) for result in read_results(path)]
    df = pd.DataFrame(results_list)

    converted = ['mae_e', 'mae_f', 'rmse_e', 'rmse_f']
    df = df.apply(lambda column: column * 1000 if column.name in converted else column)
    print(f'Columns {", ".join(converted)} are in meV or meV/Ang')

    df = df.groupby(['name', 'subset']).agg([np.mean, np.std])
    df = df.drop(columns=['seed'])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False,
                           'display.float_format', '{:.3f}'.format):
        print(df)


if __name__ == '__main__':
    main()
