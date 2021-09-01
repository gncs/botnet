import argparse
from typing import Optional


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Name and seed
    parser.add_argument('--name', help='experiment name', required=True)
    parser.add_argument('--seed', help='run ID', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--checkpoints_dir', help='directory for checkpoint files', type=str, default='checkpoints')
    parser.add_argument('--results_dir', help='directory for results', type=str, default='results')

    # Device and logging
    parser.add_argument('--device', help='select device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--default_dtype',
                        help='set default dtype',
                        type=str,
                        choices=['float32', 'float64'],
                        default='float64')
    parser.add_argument('--log_level', help='log level', type=str, default='INFO')

    # Model
    parser.add_argument('--r_max', help='distance cutoff (in Ang)', type=float, default=4.0)
    parser.add_argument('--num_radial_basis', help='number of radial basis functions', type=int, default=10)
    parser.add_argument('--num_cutoff_basis', help='number of basis functions for smooth cutoff', type=int, default=8)
    parser.add_argument('--max_ell', help=r'maximum \ell in spherical harmonics series expansion', type=int, default=3)
    parser.add_argument('--interaction',
                        help='name of interaction block',
                        type=str,
                        default='ElementDependentInteractionBlock')
    parser.add_argument('--num_interactions', help='number of interactions', type=int, default=6)
    parser.add_argument('--hidden_irreps',
                        help='irreps for hidden node states',
                        type=str,
                        default='32x0e + 32x1o + 32x2e + 32x3o')

    # Optimizer
    parser.add_argument('--optimizer',
                        help='Optimizer for parameter optimization',
                        type=str,
                        default='amsgrad',
                        choices=['adam', 'amsgrad'])
    parser.add_argument('--batch_size', help='batch size', type=int, default=10)
    parser.add_argument('--lr', help='Learning rate of optimizer', type=float, default=0.01)
    parser.add_argument('--lr_scheduler_gamma', help='Gamma of learning rate scheduler', type=float, default=1.0)
    parser.add_argument('--max_num_epochs', help='Maximum number of epochs', type=int, default=2048)
    parser.add_argument('--patience',
                        help='Maximum number of consecutive epochs of increasing loss',
                        type=int,
                        default=2048)
    parser.add_argument('--eval_interval', help='evaluate model every <n> epochs', type=int, default=1)
    parser.add_argument('--keep_checkpoints', help='keep all checkpoints', action='store_true', default=False)
    parser.add_argument('--restart_latest',
                        help='restart optimizer from latest checkpoint',
                        action='store_true',
                        default=False)
    return parser


def check_int_or_none(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        if value != 'None':
            raise argparse.ArgumentTypeError(f'{value} is an invalid value (int or None)') from None
        return None
