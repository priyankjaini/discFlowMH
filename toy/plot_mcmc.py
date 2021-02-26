import os
import torch
import pickle
import argparse

# Target
from target import get_target, target_choices

# Eval
import matplotlib.pyplot as plt

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='cat', choices=target_choices)

# Plot params
parser.add_argument('--pixels', type=int, default=1000)
parser.add_argument('--dpi', type=int, default=96)
parser.add_argument('--minimal', type=eval, default=True)
# Check the DPI of your monitor at: https://www.infobyip.com/detectmonitordpi.php

eval_args = parser.parse_args()

path_args = 'log/{}_args.pkl'.format(eval_args.target)
path_chain = 'log/{}_mcmc_chain.pt'.format(eval_args.target)

torch.manual_seed(0)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

################
## Load chain ##
################

mcmc_chain = torch.load(path_chain)

print(mcmc_chain.shape, mcmc_chain.min(), mcmc_chain.max())
num_chains, num_samples, num_dims = mcmc_chain.shape
mcmc_chain = mcmc_chain.reshape(num_chains*num_samples, num_dims)

##############
## Sampling ##
##############

if args.num_dims == 2:

    # Make dir
    if not os.path.exists('figures'):
        os.mkdir('figures')

    theta = mcmc_chain.numpy()

    plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
    if args.num_bits is not None: plt.hist2d(theta[:,0], theta[:,1], bins=list(range(2**args.num_bits+1)), density=True)
    else:                         plt.hist2d(theta[:,0], theta[:,1], bins=100, density=True)
    if args.minimal:
        plt.axis('off')
    else:
        plt.title('MCMC Samples')
        plt.colorbar()
        if args.num_bits is not None:
            plt.xticks(list(range(2**args.num_bits)))
            plt.yticks(list(range(2**args.num_bits)))
    plt.savefig('figures/{}_mcmc.png'.format(args.target), bbox_inches = 'tight', pad_inches = 0)

    # # Display plots
    # plt.show()
