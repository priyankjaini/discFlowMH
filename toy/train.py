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

# Optim
from torch.optim import Adam, Adamax

# Eval
import matplotlib.pyplot as plt
from prettytable import PrettyTable

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

# Target params
parser.add_argument('--target', type=str, default='gmm5', choices=target_choices)
parser.add_argument('--num_dims', type=int, default=2)
parser.add_argument('--num_bits', type=int, default=None)

# Model params
parser.add_argument('--num_flows', type=int, default=4)
parser.add_argument('--actnorm', type=eval, default=False)
parser.add_argument('--affine', type=eval, default=True)
parser.add_argument('--scale_fn', type=str, default='tanh_exp', choices={'exp', 'softplus', 'sigmoid', 'tanh_exp'})
parser.add_argument('--hidden_units', type=eval, default=[50])
parser.add_argument('--activation', type=str, default='relu', choices={'relu', 'elu', 'gelu'})
parser.add_argument('--permutation', type=str, default='reverse', choices={'reverse', 'shuffle'})
parser.add_argument('--context_size', type=int, default=64)

# Train params
parser.add_argument('--iter', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='adam', choices={'adam', 'adamax'})
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--print_every', type=int, default=100)

# Plot params
parser.add_argument('--num_samples', type=int, default=128*1000)
parser.add_argument('--pixels', type=int, default=1000)
parser.add_argument('--dpi', type=int, default=96)
parser.add_argument('--minimal', type=eval, default=True)
# Check the DPI of your monitor at: https://www.infobyip.com/detectmonitordpi.php

args = parser.parse_args()
assert args.iter % args.print_every == 0

torch.manual_seed(0)

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

I = D // 2
O = D // 2 + D % 2

# Decoder
if args.num_bits is not None:
    transforms = [Logit()]
    for _ in range(args.num_flows):
        net = nn.Sequential(MLP(C+I, P*O,
                                hidden_units=args.hidden_units,
                                activation=args.activation),
                            ElementwiseParams(P))
        context_net = nn.Sequential(LambdaLayer(lambda x: 2*x.float()/(2**args.num_bits-1) - 1),
                                    MLP(D, C,
                                        hidden_units=args.hidden_units,
                                        activation=args.activation))
        if args.affine: transforms.append(ConditionalAffineCouplingBijection(coupling_net=net, context_net=context_net, scale_fn=scale_fn(args.scale_fn), num_condition=I))
        else:           transforms.append(ConditionalAdditiveCouplingBijection(coupling_net=net, context_net=context_net, num_condition=I))
        if args.actnorm: transforms.append(ActNormBijection(D))
        if args.permutation == 'reverse':   transforms.append(Reverse(D))
        elif args.permutation == 'shuffle': transforms.append(Shuffle(D))
    transforms.pop()
    decoder = ConditionalFlow(base_dist=StandardNormal((D,)), transforms=transforms).to(args.device)

# Flow
transforms = []
for _ in range(args.num_flows):
    net = nn.Sequential(MLP(I, P*O,
                            hidden_units=args.hidden_units,
                            activation=args.activation),
                        ElementwiseParams(P))
    if args.affine: transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(args.scale_fn), num_condition=I))
    else:           transforms.append(AdditiveCouplingBijection(net, num_condition=I))
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

#######################
## Specify optimizer ##
#######################

if args.optimizer == 'adam':
    optimizer = Adam(pi.parameters(), lr=args.lr)
elif args.optimizer == 'adamax':
    optimizer = Adamax(pi.parameters(), lr=args.lr)

##############
## Training ##
##############

print('Training...')
loss_sum = 0.0
for i in range(args.iter):
    z, log_p_z = p.sample_with_log_prob(args.batch_size)
    log_pi_z = pi.log_prob(z)
    KL = (log_p_z - log_pi_z).mean()
    optimizer.zero_grad()
    loss = KL
    loss.backward()
    optimizer.step()
    loss_sum += loss.detach().cpu().item()
    d = 1+(i % args.print_every)
    print('Iter: {}/{}, KL: {:.3f}'.format(i+1, args.iter, loss_sum/d), end='\r')
    if (i+1) % args.print_every == 0:
        final_loss = loss_sum / args.print_every
        loss_sum = 0.0
        print('')

################
## Save model ##
################

print('Saving...')

# Make dir
if not os.path.exists('log'):
    os.mkdir('log')

# Save args
with open('log/{}_args.pkl'.format(args.target), "wb") as f:
    pickle.dump(args, f)
table = PrettyTable(['Arg', 'Value'])
for arg, val in vars(args).items():
    table.add_row([arg, val])
with open('log/{}_args.txt'.format(args.target), 'w') as f:
    f.write(str(table))

# Save result
with open('log/{}_loss.txt'.format(args.target), 'w') as f:
    f.write(str(final_loss))

# Save model
torch.save(pi.state_dict(), 'log/{}.pt'.format(args.target))


##############
## Sampling ##
##############

if args.num_dims == 2:

    print('Sampling...')

    # Make dir
    if not os.path.exists('figures'):
        os.mkdir('figures')

    # Learned distribution
    z = p.sample(num_samples=args.num_samples)
    for t in pi.transforms:
        z, _ = t(z)
    theta = z.detach().numpy()

    plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
    if args.num_bits is not None: plt.hist2d(theta[:,0], theta[:,1], bins=list(range(2**args.num_bits+1)), density=True)
    else:                         plt.hist2d(theta[:,0], theta[:,1], bins=100, density=True)
    if args.minimal:
        plt.axis('off')
    else:
        plt.title('Learned Distribution')
        plt.colorbar()
        if args.num_bits is not None:
            plt.xticks(list(range(2**args.num_bits)))
            plt.yticks(list(range(2**args.num_bits)))
    plt.savefig('figures/{}_flow.png'.format(args.target), bbox_inches = 'tight', pad_inches = 0)

    # Target Distribution
    theta = target.sample(num_samples=args.num_samples).detach().numpy()

    plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
    if args.num_bits is not None: plt.hist2d(theta[:,0], theta[:,1], bins=list(range(2**args.num_bits+1)), density=True)
    else:                         plt.hist2d(theta[:,0], theta[:,1], bins=100, density=True)
    if args.minimal:
        plt.axis('off')
    else:
        plt.title('Target Distribution')
        plt.colorbar()
        if args.num_bits is not None:
            plt.xticks(list(range(2**args.num_bits)))
            plt.yticks(list(range(2**args.num_bits)))
    plt.savefig('figures/{}.png'.format(args.target), bbox_inches = 'tight', pad_inches = 0)

    # # Display plots
    # plt.show()
