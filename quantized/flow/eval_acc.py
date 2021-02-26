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

# Model
from model import get_model, get_model_id, add_model_args
from survae.distributions import StandardNormal

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--num_samples', type=int, default=100)
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
## Specify model ##
###################

pi = get_model(args, target=target, num_bits=args.num_bits).to(eval_args.device)
p = StandardNormal((target.size,)).to(eval_args.device)
model_id = get_model_id(args)

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

    path_check = os.path.join(exp_path, 'model{}.pt'.format(split))
    state_dict = torch.load(path_check)
    pi.load_state_dict(state_dict)

    ##############
    ## Training ##
    ##############

    print('Evaluating...')
    with torch.no_grad():
        z = p.sample(eval_args.num_samples)
        for t in pi.transforms:
            z, _ = t(z)
        theta = z
        print('theta min/max:', theta.min(), theta.max())
        acc = target.accuracy(theta, test=True)
    acc = acc.mean().cpu().item()
    print('Test Acc: {:.3f}'.format(acc))

    ################
    ## Save model ##
    ################

    print('Saving...')

    # Save accuracy
    N_test = target.X_test.shape[0]
    num_total += N_test
    num_correct += N_test * acc
    with open(os.path.join(exp_path, 'acc{}.txt'.format(split)), 'w') as f:
        f.write(str(acc))

if args.split is None:
    # Save accuracy
    acc_avg = num_correct / num_total
    with open(os.path.join(exp_path, 'acc_avg.txt'), 'w') as f:
        f.write(str(acc_avg))
