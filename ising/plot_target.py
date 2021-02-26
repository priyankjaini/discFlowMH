import torch
import argparse

# Plot
import matplotlib.pyplot as plt

# Target
from target import get_target, get_target_id, add_target_args

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

# Target params
add_target_args(parser)

# Printing params
parser.add_argument('--samples', type=int, default=16)
parser.add_argument('--plot', type=eval, default=False)

args = parser.parse_args()

torch.manual_seed(0)

####################
## Specify target ##
####################

dist = get_target(args)

##############
## Printing ##
##############

if args.plot:
    plt.figure()
    plt.imshow(dist.img)
    plt.figure()
    plt.imshow(dist.img_corrupted)
    plt.show()

print('Config size: {}'.format(dist.size))
