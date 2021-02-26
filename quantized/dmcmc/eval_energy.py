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

###################
## Define splits ##
###################

if args.split is None:
    splits = list(range(args.cv_folds))
    mins, maxs, means = [], [], []
else:
    splits = [args.split]

for split in splits:

    print('Split {}:'.format(split))
    target.set_split(split=split)
    path_chain = os.path.join(exp_path, 'chain{}.pt'.format(split))

    #################
    ## Compute ESS ##
    #################

    theta = torch.load(path_chain).to(eval_args.device)
    num_chains, num_samples, num_dims = theta.shape
    energy = torch.zeros(num_chains, num_samples).to(eval_args.device)
    for i in range(num_samples):
        energy[:,i] = - target.log_prob(theta[:,i])
        print('{}/{}'.format(i+1, num_samples), end='\r')
    energy_min, energy_med, energy_max = torch.min(energy).item(), torch.median(energy).item(), torch.max(energy).item()
    energy_mean, energy_std = torch.mean(energy).item(), torch.std(energy).item()

    ##################
    ## Save samples ##
    ##################

    print('Saving...')

    # Save ESS
    with open(os.path.join(exp_path, 'energy{}.txt'.format(split)), 'w') as f:
        f.write('min: {}\n'.format(energy_min))
        f.write('med: {}\n'.format(energy_med))
        f.write('max: {}\n'.format(energy_max))
        f.write('mean: {}\n'.format(energy_mean))
        f.write('std: {}\n'.format(energy_std))
    mins.append(energy_min)
    maxs.append(energy_max)
    means.append(energy_mean)

if args.split is None:
    # Save rate
    tot_min = min(mins)
    tot_max = max(maxs)
    tot_mean = sum(means) / len(means)
    with open(os.path.join(exp_path, 'energy_tot.txt'), 'w') as f:
        f.write('min: {}\n'.format(tot_min))
        f.write('max: {}\n'.format(tot_max))
        f.write('mean: {}\n'.format(tot_mean))
