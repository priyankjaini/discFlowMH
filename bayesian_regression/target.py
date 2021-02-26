import torch
from targets import BayesianRegression
from sklearn.datasets import make_regression


target_choices = {'toy100', 'toy200', 'toy400'}


def add_target_args(parser):
    parser.add_argument('--target', type=str, default='toy100', choices=target_choices)
    parser.add_argument('--w', type=float, default=4.0)


def get_target_id(args):
    return '{}'.format(args.target)


def get_target(args):
    assert args.target in target_choices

    # Target
    if args.target == 'toy100':
        data = make_regression(n_samples=200, n_features=100, n_informative=10)
        x, y = torch.from_numpy(data[0]).float(), torch.from_numpy(data[1]).float()
    elif args.target == 'toy200':
        data = make_regression(n_samples=400, n_features=200, n_informative=20)
        x, y = torch.from_numpy(data[0]).float(), torch.from_numpy(data[1]).float()
    elif args.target == 'toy400':
        data = make_regression(n_samples=800, n_features=400, n_informative=40)
        x, y = torch.from_numpy(data[0]).float(), torch.from_numpy(data[1]).float()

    target_dist = BayesianRegression(data=x, y=y, w=args.w)

    return target_dist
