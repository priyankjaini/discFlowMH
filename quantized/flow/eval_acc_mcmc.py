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

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--mcmc', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
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

target = get_target(args).to(eval_args.device)
target_id = get_target_id(args)

###################
## Define splits ##
###################

if args.split is None:
    splits = list(range(args.cv_folds))
    num_correct = 0
    num_total = 0
else:
    splits = [args.split]

for split in splits:

    print('Split {}:'.format(split))
    target.set_split(split=split)
    path_chain = os.path.join(mcmc_path, 'chain{}.pt'.format(split))
    theta = torch.load(path_chain).to(eval_args.device)

    ##############
    ## Training ##
    ##############

    print('Evaluating...')
    with torch.no_grad():
        acc = 0.0
        time_steps = theta.shape[1]
        for t in range(time_steps):
            acc += target.accuracy(theta[:,t], test=True)
    acc = acc.mean().cpu().item() / time_steps
    print('Avg. Test Acc.: {:.3f}'.format(acc))

    ################
    ## Save model ##
    ################

    print('Saving...')

    # Save result
    N_test = target.X_test.shape[0]
    num_total += N_test
    num_correct += N_test * acc
    with open(os.path.join(mcmc_path, 'acc{}.txt'.format(split)), 'w') as f:
        f.write(str(acc))

if args.split is None:
    # Save accuracy
    acc_avg = num_correct / num_total
    with open(os.path.join(mcmc_path, 'acc_avg.txt'), 'w') as f:
        f.write(str(acc_avg))
