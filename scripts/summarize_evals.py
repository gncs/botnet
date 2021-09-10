import argparse
import dataclasses
import glob
import json
import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory pointing to evaluation files', required=True)
    return parser.parse_args()


def get_paths(directory: dir) -> List[str]:
    return glob.glob(os.path.join(directory, '*_eval.txt'))


@dataclasses.dataclass
class ExperimentInfo:
    name: str
    seed: int


name_re = re.compile(r'(?P<name>.+)_run-(?P<seed>\d+)_eval.txt')


def parse_path(path: str) -> ExperimentInfo:
    match = name_re.match(os.path.basename(path))
    if not match:
        raise RuntimeError(f'Cannot parse {path}')

    return ExperimentInfo(name=match.group('name'), seed=int(match.group('seed')))


def read_results(path: str) -> List[Dict]:
    exp_info = parse_path(path)

    results = []
    with open(path, mode='r') as f:
        for line in f.readlines():
            result = json.loads(line)
            result['exp'] = exp_info.name
            result['seed'] = exp_info.seed
            results.append(result)

    return results


def main():
    args = parse_args()
    results_list = [result for path in get_paths(directory=args.dir) for result in read_results(path)]
    df = pd.DataFrame(results_list)

    df = df.drop(columns=['seed'])
    df = df.groupby(['exp', 'name']).agg([np.mean, np.std])

    converted = ['mae_e', 'mae_f', 'rmse_e', 'rmse_f']
    df = df.apply(lambda column: column * 1000 if column.name in converted else column)
    print(f'Columns {", ".join(converted)} are in meV or meV/Ang')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False,
                           'display.float_format', '{:.3f}'.format):
        print(df)


if __name__ == '__main__':
    main()
