import torch
from targets import IsingMNIST


target_choices = {'mnist0'}


def add_target_args(parser):
    parser.add_argument('--target', type=str, default='mnist0', choices=target_choices)
    parser.add_argument('--downsample', type=int, default=0)
    parser.add_argument('--corruption_prob', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=2.1)


def get_target_id(args):
    size = 28 // (2**args.downsample)
    size_str = str(size) + 'x' + str(size)
    return '{}_{}_p{}_b{}_m{}'.format(args.target, size_str, args.corruption_prob, args.beta, args.mu)


def get_target(args):
    assert args.target in target_choices

    # Target
    if args.target == 'mnist0':
        target_dist = IsingMNIST(idx=0, downsample=args.downsample, corruption_prob=args.corruption_prob, beta=args.beta, mu=args.mu)

    return target_dist
