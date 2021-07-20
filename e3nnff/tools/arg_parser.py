import argparse
from typing import Optional


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Command line tool of MolGen3D')

    # Name and seed
    parser.add_argument('--name', help='experiment name', required=True)
    parser.add_argument('--seed', help='run ID', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--checkpoints_dir', help='directory for checkpoint files', type=str, default='checkpoints')
    parser.add_argument('--results_dir', help='directory for results', type=str, default='results')

    # Device and logging
    parser.add_argument('--device', help='select device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--log_level', help='log level', type=str, default='INFO')

    # Model
    parser.add_argument('--r_max', help='distance cutoff (in Ang)', type=float, default=4.0)
    parser.add_argument('--num_radial_basis', help='number of radial basis functions', type=int, default=10)
    parser.add_argument('--num_cutoff_basis', help='number of basis functions for smooth cutoff', type=int, default=8)
    parser.add_argument('--max_ell', help=r'maximum \ell in spherical harmonics series expansion', type=int, default=2)
    parser.add_argument('--interaction', help='name of interaction block', type=str, default='SkipInteractionBlock')
    parser.add_argument('--num_interactions', help='number of interactions', type=int, default=4)
    parser.add_argument('--hidden_irreps',
                        help='irreps for hidden node states',
                        type=str,
                        default='10x0e + 10x0o + 10x1e + 10x1o + 10x2e + 10x2o')
    parser.add_argument('--no_forces', help='no forces in loss function', action='store_true', default=False)
    parser.add_argument('--scale_shift', help='scale and shift interaction energy', action='store_true', default=False)

    # Optimizer
    parser.add_argument('--optimizer',
                        help='Optimizer for parameter optimization',
                        type=str,
                        default='amsgrad',
                        choices=['adam', 'amsgrad'])
    parser.add_argument('--batch_size', help='batch size', type=int, default=10)
    parser.add_argument('--lr', help='Learning rate of optimizer', type=float, default=0.01)
    parser.add_argument('--lr_scheduler_gamma', help='Gamma of learning rate scheduler', type=float, default=0.9995)
    parser.add_argument('--max_num_epochs', help='Maximum number of epochs', type=int, default=2048)
    parser.add_argument('--patience',
                        help='Maximum number of consecutive epochs of increasing loss',
                        type=int,
                        default=64)
    parser.add_argument('--eval_interval', help='evaluate model every <n> epochs', type=int, default=1)
    parser.add_argument('--keep_checkpoints', help='keep all checkpoints', action='store_true', default=False)
    parser.add_argument('--restart_latest',
                        help='restart optimizer from latest checkpoint',
                        action='store_true',
                        default=False)
    return parser


def add_rmd17_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--downloads_dir', help='directory for downloads', type=str, default='downloads')
    parser.add_argument('--subset', help='subset name', default='uracil')
    parser.add_argument('--split', help='train test split', type=int, default=1)
    parser.add_argument('--valid_fraction',
                        help='fraction of the training set used for validation',
                        type=float,
                        default=0.1)
    parser.add_argument('--max_size_train',
                        help='maximum number of items in training set (int or None)',
                        type=check_int_or_none,
                        default=None)
    parser.add_argument('--max_size_test',
                        help='maximum number of items in test set (int or None)',
                        type=check_int_or_none,
                        default=None)

    return parser


def check_int_or_none(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        if value != 'None':
            raise argparse.ArgumentTypeError(f'{value} is an invalid value (int or None)') from None
        return None
