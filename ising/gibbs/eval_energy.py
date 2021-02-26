import os
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

# Target
from target import get_target, get_target_id, add_target_args

# MCMC
from ess import ESS

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
eval_args = parser.parse_args()

torch.manual_seed(0)
exp_path = os.path.join('log', eval_args.exp)
path_args = os.path.join(exp_path, 'args.pkl')
path_chain = os.path.join(exp_path, 'chain.pt')

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

####################
## Specify target ##
####################

target = get_target(args).to(eval_args.device)
target_id = get_target_id(args)

####################
## Compute energy ##
####################

theta = torch.load(path_chain).to(eval_args.device)
num_chains, num_samples, num_dims = theta.shape
energy = torch.zeros(num_chains, num_samples).to(eval_args.device)
for i in range(num_samples):
    energy[:,i] = - target.log_prob(theta[:,i])
    print('{}/{}'.format(i+1, num_samples), end='\r')
min, med, max = torch.min(energy).item(), torch.median(energy).item(), torch.max(energy).item()
mean, std = torch.mean(energy).item(), torch.std(energy).item()

##################
## Save samples ##
##################

print('Saving...')

# Save ESS
with open(os.path.join(exp_path, 'energy.txt'), 'w') as f:
    f.write('min: {}\n'.format(min))
    f.write('med: {}\n'.format(med))
    f.write('max: {}\n'.format(max))
    f.write('mean: {}\n'.format(mean))
    f.write('std: {}\n'.format(std))
