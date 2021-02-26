import torch
from survae.utils import sum_except_batch
from targets import LogisticRegressionIris, LogisticRegressionWine, LogisticRegressionBreastCancer


target_choices = {'iris_logreg', 'wine_logreg', 'bcancer_logreg'}


def add_target_args(parser):
    parser.add_argument('--target', type=str, default='iris_logreg', choices=target_choices)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--num_bits', type=int, default=4, choices={1,2,3,4})
    parser.add_argument('--prior', type=str, default='uniform', choices={'gaussian','uniform'})


def get_target_id(args):
    return '{}{}_{}_{}bit'.format(args.target, args.split is not None if args.split else '', args.prior, args.num_bits)


def get_target(args):
    assert args.target in target_choices

    # Target
    if args.target == 'iris_logreg':
        target_dist = LogisticRegressionIris(num_bits=args.num_bits, prior=args.prior, cv_folds=args.cv_folds)
    if args.target == 'wine_logreg':
        target_dist = LogisticRegressionWine(num_bits=args.num_bits, prior=args.prior, cv_folds=args.cv_folds)
    if args.target in 'bcancer_logreg':
        target_dist = LogisticRegressionBreastCancer(num_bits=args.num_bits, prior=args.prior, cv_folds=args.cv_folds)

    return target_dist
