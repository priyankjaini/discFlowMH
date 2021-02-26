import torch
import argparse

# Plot
import matplotlib.pyplot as plt

# Target
from target import get_target, target_choices

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

# Target params
parser.add_argument('--target', type=str, default='cat', choices=target_choices)
parser.add_argument('--num_dims', type=int, default=2)
parser.add_argument('--num_bits', type=int, default=None)
parser.add_argument('--device', type=str, default='cpu')

# Plotting params
parser.add_argument('--num_samples', type=int, default=128*1000)
parser.add_argument('--pixels', type=int, default=1000)
parser.add_argument('--dpi', type=int, default=96)
parser.add_argument('--minimal', type=eval, default=True)
parser.add_argument('--density', type=eval, default=False)
# Check the DPI of your monitor at: https://www.infobyip.com/detectmonitordpi.php

args = parser.parse_args()
assert args.num_dims == 2, 'Plotting only supported for num_dims=2.'

torch.manual_seed(0)

####################
## Specify target ##
####################

dist, shape = get_target(args)

##############
## Sampling ##
##############

theta = dist.sample(num_samples=args.num_samples).detach().numpy()

plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
if args.num_bits is not None:
    plt.hist2d(theta[:,0], theta[:,1], bins=list(range(2**args.num_bits+1)), density=True)
else:
    plt.hist2d(theta[:,0], theta[:,1], bins=100, density=True)
if args.minimal:
    plt.axis('off')
else:
    plt.title('Target Distribution')
    plt.colorbar()
    if args.num_bits is not None:
        plt.xticks(list(range(2**args.num_bits)))
        plt.yticks(list(range(2**args.num_bits)))
# plt.savefig('figures/{}.png'.format(args.target), bbox_inches = 'tight', pad_inches = 0)



# Plot density
if args.density:
    xv, yv = torch.meshgrid([torch.arange(2**args.num_bits).float(), torch.arange(2**args.num_bits).float()])
    x = torch.cat([xv.reshape(-1,1), yv.reshape(-1,1)], dim=-1)
    with torch.no_grad():
        logprobs = dist.log_prob(x)
    plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
    plt.pcolormesh(xv, yv, logprobs.exp().reshape(xv.shape))
    if args.minimal:
        plt.axis('off')
    else:
        plt.title('Target Distribution Density')
        plt.colorbar()
        if args.num_bits is not None:
            plt.xticks(list(range(2**args.num_bits)))
            plt.yticks(list(range(2**args.num_bits)))
    plt.show()
