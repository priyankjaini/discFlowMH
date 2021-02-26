import os
import math
import time
import torch
import pickle
import argparse

# Path
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
basedir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(basedir)

# MCMC
from ess import ESS

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--mcmc', type=str, required=True)
parser.add_argument('--lag', type=int, default=None)
parser.add_argument('--chains', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--verbose', type=eval, default=False)
eval_args = parser.parse_args()

torch.manual_seed(0)
exp_path = os.path.join('log', eval_args.exp)
path_args = os.path.join(exp_path, 'args.pkl')
mcmc_path = os.path.join(exp_path, eval_args.mcmc)
path_chain = os.path.join(mcmc_path, 'chain.pt')

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

#################
## Compute ESS ##
#################

theta = torch.load(path_chain).to(eval_args.device)
B, L, D = theta.shape
assert B % eval_args.chains == 0
num = B // eval_args.chains
means = torch.zeros(num)
for i in range(num):
    l, u = eval_args.chains*i, eval_args.chains*(i+1)
    _, _, _, mean, _ = ESS(theta[l:u], lag=eval_args.lag, verbose=eval_args.verbose)
    means[i] = mean

print(means)
mean = torch.mean(means).item()
std = torch.std(means).item() / math.sqrt(num)

##################
## Save samples ##
##################

print('Saving...')

# Save ESS
with open(os.path.join(mcmc_path, 'ess_std{}.txt'.format(eval_args.lag if eval_args.lag else '')), 'w') as f:
    f.write('mean: {}\n'.format(mean))
    f.write('std: {}\n'.format(std))
