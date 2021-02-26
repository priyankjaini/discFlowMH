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

# MCMC
from metropolis_hastings_discrete import metropolis_hastings_discrete

# Eval
from prettytable import PrettyTable

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

# Log params
parser.add_argument('--name', type=str, default=time.strftime("%Y-%m-%d_%H-%M-%S"))

# Target params
add_target_args(parser)

# Train params
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

# MCMC params
parser.add_argument('--num_chains', type=int, default=128)
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--steps_per_sample', type=int, default=1)
parser.add_argument('--burnin_steps', type=int, default=0)
parser.add_argument('--proposal_prob', type=float, default=0.5)
parser.add_argument('--reflective', type=eval, default=True)

args = parser.parse_args()

torch.manual_seed(0)
if not os.path.exists('log'): os.mkdir('log')
exp_path = os.path.join('log', args.name)

####################
## Specify target ##
####################

target = get_target(args)
target_id = get_target_id(args)

##############
## Training ##
##############

print('Running MCMC...')
time_before = time.time()
samples, rate = metropolis_hastings_discrete(pi=target.to(args.device),
                                             num_dims=target.size,
                                             num_chains=args.num_chains,
                                             num_samples=args.num_samples,
                                             num_bits=1,
                                             steps_per_sample=args.steps_per_sample,
                                             burnin_steps=args.burnin_steps,
                                             proposal_prob=args.proposal_prob,
                                             reflective=args.reflective,
                                             device=args.device)
runtime = time.time() - time_before

##################
## Save samples ##
##################

print('Saving...')

# Make dir
os.mkdir(exp_path)

# Save args
with open(os.path.join(exp_path, 'args.pkl'), "wb") as f:
    pickle.dump(args, f)
table = PrettyTable(['Arg', 'Value'])
for arg, val in vars(args).items():
    table.add_row([arg, val])
with open(os.path.join(exp_path, 'args.txt'), 'w') as f:
    f.write(str(table))

# Save model
torch.save(samples, os.path.join(exp_path, 'chain.pt'))

# Save rate
with open(os.path.join(exp_path, 'accept_rate.txt'), 'w') as f:
    f.write(str(rate))

# Save time
with open(os.path.join(exp_path, 'runtime.txt'), 'w') as f:
    f.write(str(runtime))
