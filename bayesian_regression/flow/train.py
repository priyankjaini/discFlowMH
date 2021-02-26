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

# Optim
from torch.optim import Adam, Adamax

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

# Model params
add_model_args(parser)

# Train params
parser.add_argument('--iter', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='adam', choices={'adam', 'adamax'})
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--print_every', type=int, default=100)

args = parser.parse_args()
assert args.iter % args.print_every == 0

torch.manual_seed(0)
if not os.path.exists('log'): os.mkdir('log')
exp_path = os.path.join('log', args.name)

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
time_before = time.time()
loss_sum = 0.0
for i in range(args.iter):
    z, log_p_z = p.sample_with_log_prob(args.batch_size)
    log_pi_z_tilde = pi.log_prob(z)
    KL_tilde = (log_p_z - log_pi_z_tilde).mean()
    optimizer.zero_grad()
    loss = KL_tilde
    loss.backward()
    optimizer.step()
    loss_sum += loss.detach().cpu().item()
    d = 1+(i % args.print_every)
    print('Iter: {}/{}, Unnormalized KL: {:.3f}'.format(i+1, args.iter, loss_sum/d), end='\r')
    if (i+1) % args.print_every == 0:
        final_loss = loss_sum / args.print_every
        loss_sum = 0.0
        print('')
runtime = time.time() - time_before

################
## Save model ##
################

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

# Save result
with open(os.path.join(exp_path, 'loss.txt'), 'w') as f:
    f.write(str(final_loss))

# Save model
torch.save(pi.state_dict(), os.path.join(exp_path, 'model.pt'))

# Save time
with open(os.path.join(exp_path, 'runtime.txt'), 'w') as f:
    f.write(str(runtime))
