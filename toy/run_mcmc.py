import os
import torch
import pickle
import argparse

# Path
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# Target
from target import get_target, target_choices

# Model
import torch.nn as nn
from survae.flows import Flow, ConditionalFlow
from survae.distributions import StandardNormal
from survae.transforms import ConditionalAdditiveCouplingBijection, ConditionalAffineCouplingBijection
from survae.transforms import AdditiveCouplingBijection, AffineCouplingBijection
from survae.transforms import ActNormBijection, Reverse, Shuffle, Sigmoid, Logit
from survae.nn.layers import LambdaLayer, ElementwiseParams, scale_fn
from survae.nn.nets import MLP
from quantization_variational import VariationalQuantization

# MCMC
from metropolis_hastings import metropolis_hastings

# Eval
from prettytable import PrettyTable

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='cat', choices=target_choices)
parser.add_argument('--num_chains', type=int, default=128)
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--steps_per_sample', type=int, default=1)
parser.add_argument('--burnin_steps', type=int, default=0)
parser.add_argument('--proposal_scale', type=float, default=1.0)
eval_args = parser.parse_args()

path_args = 'log/{}_args.pkl'.format(eval_args.target)
path_check = 'log/{}.pt'.format(eval_args.target)

torch.manual_seed(0)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

####################
## Specify target ##
####################

target, shape = get_target(args)

###################
## Specify model ##
###################

D = args.num_dims # Number of data dimensions
P = 2 if args.affine else 1 # Number of elementwise parameters
C = args.context_size # Size of context

assert D % 2 == 0, 'Only even dimension supported currently.'
assert C % D == 0, 'context_size needs to be multiple of num_dims.'

# Decoder
if args.num_bits is not None:
    transforms = [Logit()]
    for _ in range(args.num_flows):
        net = nn.Sequential(MLP(C+D//2, P*D//2,
                                hidden_units=args.hidden_units,
                                activation=args.activation),
                            ElementwiseParams(P))
        context_net = nn.Sequential(LambdaLayer(lambda x: 2*x.float()/(2**args.num_bits-1) - 1),
                                    MLP(D, C,
                                        hidden_units=args.hidden_units,
                                        activation=args.activation))
        if args.affine: transforms.append(ConditionalAffineCouplingBijection(coupling_net=net, context_net=context_net, scale_fn=scale_fn(args.scale_fn)))
        else:           transforms.append(ConditionalAdditiveCouplingBijection(coupling_net=net, context_net=context_net))
        if args.actnorm: transforms.append(ActNormBijection(D))
        if args.permutation == 'reverse':   transforms.append(Reverse(D))
        elif args.permutation == 'shuffle': transforms.append(Shuffle(D))
    transforms.pop()
    decoder = ConditionalFlow(base_dist=StandardNormal((D,)), transforms=transforms).to(args.device)

# Flow
transforms = []
for _ in range(args.num_flows):
    net = nn.Sequential(MLP(D//2, P*D//2,
                            hidden_units=args.hidden_units,
                            activation=args.activation),
                        ElementwiseParams(P))
    if args.affine: transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(args.scale_fn)))
    else:           transforms.append(AdditiveCouplingBijection(net))
    if args.actnorm: transforms.append(ActNormBijection(D))
    if args.permutation == 'reverse':   transforms.append(Reverse(D))
    elif args.permutation == 'shuffle': transforms.append(Shuffle(D))
transforms.pop()
if args.num_bits is not None:
    transforms.append(Sigmoid())
    transforms.append(VariationalQuantization(decoder, num_bits=args.num_bits))


pi = Flow(base_dist=target,
          transforms=transforms).to(args.device)

p = StandardNormal(shape).to(args.device)

##############
## Training ##
##############

state_dict = torch.load(path_check)
pi.load_state_dict(state_dict)

print('Running MCMC...')
samples, rate = metropolis_hastings(pi=pi,
                                    num_dims=args.num_dims,
                                    num_chains=eval_args.num_chains,
                                    num_samples=eval_args.num_samples,
                                    steps_per_sample=eval_args.steps_per_sample,
                                    burnin_steps=eval_args.burnin_steps,
                                    proposal_scale=eval_args.proposal_scale)

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

# Save args
with open('log/{}_mcmc_args.pkl'.format(args.target), "wb") as f:
    pickle.dump(eval_args, f)
table = PrettyTable(['Arg', 'Value'])
for arg, val in vars(eval_args).items():
    table.add_row([arg, val])
with open('log/{}_mcmc_args.txt'.format(args.target), 'w') as f:
    f.write(str(table))

# Save model
torch.save(samples, 'log/{}_mcmc_chain.pt'.format(args.target))

# Save rate
with open('log/{}_mcmc_accept_rate.txt'.format(args.target), 'w') as f:
    f.write(str(rate))
