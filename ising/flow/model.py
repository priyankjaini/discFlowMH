# Path
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
basedir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(basedir)
from flow import CouplingFlow


def add_model_args(parser):
    parser.add_argument('--num_flows', type=int, default=4)
    parser.add_argument('--actnorm', type=eval, default=False)
    parser.add_argument('--affine', type=eval, default=True)
    parser.add_argument('--scale_fn', type=str, default='tanh_exp', choices={'exp', 'softplus', 'sigmoid', 'tanh_exp'})
    parser.add_argument('--hidden_units', type=eval, default=[512])
    parser.add_argument('--activation', type=str, default='relu', choices={'relu', 'elu', 'gelu'})
    parser.add_argument('--permutation', type=str, default='shuffle', choices={'learned', 'shuffle', 'reverse'})
    parser.add_argument('--context_size', type=int, default=64)


def get_model_id(args):
    return 'flow'


def get_model(args, target):

    return CouplingFlow(target=target,
                        num_bits=1,
                        num_flows=args.num_flows,
                        actnorm=args.actnorm,
                        affine=args.affine,
                        scale_fn=args.scale_fn,
                        hidden_units=args.hidden_units,
                        activation=args.activation,
                        permutation=args.permutation,
                        context_size=args.context_size)
