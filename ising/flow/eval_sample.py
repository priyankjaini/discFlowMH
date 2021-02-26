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

# Model
from model import get_model, get_model_id, add_model_args
from survae.distributions import StandardNormal

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
path_check = os.path.join(exp_path, 'model.pt')

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

pi = get_model(args, target=target).to(args.device)
p = StandardNormal((target.size,)).to(args.device)
model_id = get_model_id(args)

state_dict = torch.load(path_check)
pi.load_state_dict(state_dict)

##############
## Sampling ##
##############

print('Sampling...')
pi = pi.eval()
with torch.no_grad():
    z = p.sample(eval_args.num_samples)
    for t in pi.transforms:
        z, _ = t(z)
    theta = z
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
