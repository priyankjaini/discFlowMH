import os
import torch
import pickle
import argparse
import torchvision.utils as vutils

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
parser.add_argument('--num_samples', type=int, default=64)
parser.add_argument('--nrow', type=int, default=8)
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

target = get_target(args)
target_id = get_target_id(args)

##############
## Sampling ##
##############

print('Sampling...')
with torch.no_grad():
    theta = torch.load(path_chain) # (C,T,D)
    theta = theta[:,-1] # (C,D)
    perm = torch.randperm(theta.shape[0])
    idx = perm[:eval_args.num_samples]
    theta = theta[idx]
    imgs = target.vec2img(theta).cpu().float().unsqueeze(1)

############
## Sample ##
############

path_samples = os.path.join(exp_path, 'samples.png')
vutils.save_image(imgs, fp=path_samples, nrow=eval_args.nrow)

data_true = (target.img.unsqueeze(0).unsqueeze(0)+1)/2
data_corr = (target.img_corrupted.unsqueeze(0).unsqueeze(0)+1)/2
vutils.save_image(data_true, fp=os.path.join(exp_path, 'data_true.png'), nrow=1)
vutils.save_image(data_corr, fp=os.path.join(exp_path, 'data_corr.png'), nrow=1)
