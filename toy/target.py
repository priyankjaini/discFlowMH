import torch
from targets import Cat, DGMM, GMM

target_choices = {'cat','gmm5',*['dgmm5_s{}'.format(s) for s in range(10)]}


def get_target(args):
    assert args.target in target_choices

    # Target
    if args.target == 'cat':
        assert args.num_bits is not None
        target_dist = Cat(num_dims=args.num_dims, num_bits=args.num_bits)
        target_shape = (args.num_dims,)
    if args.target in ['dgmm5_s{}'.format(s) for s in range(19)]:
        seed = int(args.target[-1])
        torch.manual_seed(seed)
        print('Set seed to {}'.format(seed))
        assert args.num_bits is not None
        target_dist = DGMM(num_dims=args.num_dims, num_bits=args.num_bits, num_mix=5)
        target_shape = (args.num_dims,)
    if args.target == 'gmm5':
        assert args.num_bits is None
        target_dist = GMM(num_dims=args.num_dims, num_mix=5)
        target_shape = (args.num_dims,)

    return target_dist, target_shape
