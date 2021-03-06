import os
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

# Target
from target import get_target, get_target_id, add_target_args

# Model
from model import get_model, get_model_id, add_model_args
from survae.distributions import StandardNormal

# MCMC
from metropolis_hastings import metropolis_hastings

# Eval
from prettytable import PrettyTable

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--mcmc', type=str, default=time.strftime("%Y-%m-%d_%H-%M-%S"))
parser.add_argument('--num_chains', type=int, default=128)
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--steps_per_sample', type=int, default=1)
parser.add_argument('--burnin_steps', type=int, default=0)
parser.add_argument('--proposal_scale', type=float, default=0.1)
eval_args = parser.parse_args()

torch.manual_seed(0)
exp_path = os.path.join('log', eval_args.exp)
path_args = os.path.join(exp_path, 'args.pkl')
mcmc_path = os.path.join(exp_path, eval_args.mcmc)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

####################
## Specify target ##
####################

target = get_target(args)
target_id = get_target_id(args)

###################
## Specify model ##
###################

pi = get_model(args, target=target, num_bits=args.num_bits).to(args.device)
p = StandardNormal((target.size,)).to(args.device)
model_id = get_model_id(args)

###################
## Define splits ##
###################

if args.split is None:
    splits = list(range(args.cv_folds))
    accept_rates = []
    runtimes = []
else:
    splits = [args.split]

for split in splits:

    print('Split {}:'.format(split))
    target.set_split(split=split)

    path_check = os.path.join(exp_path, 'model{}.pt'.format(split))
    state_dict = torch.load(path_check)
    pi.load_state_dict(state_dict)

    ##############
    ## Training ##
    ##############

    print('Running MCMC...')
    time_before = time.time()
    samples, rate = metropolis_hastings(pi=pi,
                                        num_dims=target.size,
                                        num_chains=eval_args.num_chains,
                                        num_samples=eval_args.num_samples,
                                        steps_per_sample=eval_args.steps_per_sample,
                                        burnin_steps=eval_args.burnin_steps,
                                        proposal_scale=eval_args.proposal_scale)
    runtime = time.time() - time_before

    print('')
    print('Projecting...')
    num_chains, num_samples, dim = samples.shape
    theta = torch.zeros(num_chains, num_samples, dim)
    with torch.no_grad():
        for i in range(num_samples):
            print('{}/{}'.format(i+1, num_samples), end='\r')
            for t in pi.transforms:
                samples[:,i], _ = t(samples[:,i])
                theta[:,i] = samples[:,i].detach().cpu()

    ##################
    ## Save samples ##
    ##################

    print('Saving...')

    # Make dir
    if not os.path.exists(mcmc_path): os.mkdir(mcmc_path)

    # Save args
    if not os.path.exists(os.path.join(mcmc_path, 'args.pkl')):
        with open(os.path.join(mcmc_path, 'args.pkl'), "wb") as f:
            pickle.dump(eval_args, f)
        table = PrettyTable(['Arg', 'Value'])
        for arg, val in vars(eval_args).items():
            table.add_row([arg, val])
        with open(os.path.join(mcmc_path, 'args.txt'), 'w') as f:
            f.write(str(table))

    # Save model
    torch.save(samples, os.path.join(mcmc_path, 'chain{}.pt'.format(split)))

    # Save rate
    with open(os.path.join(mcmc_path, 'accept_rate{}.txt'.format(split)), 'w') as f:
        f.write(str(rate))
    if args.split is None:
        accept_rates.append(rate)

    # Save time
    with open(os.path.join(mcmc_path, 'runtime{}.txt'.format(split)), 'w') as f:
        f.write(str(runtime))
    if args.split is None:
        runtimes.append(runtime)

if args.split is None:
    # Save rate
    rate_avg = sum(accept_rates) / len(accept_rates)
    with open(os.path.join(mcmc_path, 'accept_rate_avg.txt'), 'w') as f:
        f.write(str(rate_avg))

    # Save time
    runtime_avg = sum(runtimes) / len(runtimes)
    with open(os.path.join(mcmc_path, 'runtime_avg.txt'), 'w') as f:
        f.write(str(runtime))
