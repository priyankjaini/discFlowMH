import copy
import torch
import torch.nn.functional as F


def gibbs_core(z_old, target):
    num_chains, num_dims = z_old.shape
    device = z_old.device
    i = torch.randint(num_dims, (num_chains,))
    io = F.one_hot(i, num_classes=num_dims).bool()

    z_new = z_old.clone()

    # Case 1
    z_new[io] = 1.0
    log_prob_plus = target.log_prob(z_new)

    # Case 0
    z_new[io] = 0.0
    log_prob_minus = target.log_prob(z_new)

    p = torch.sigmoid(log_prob_plus-log_prob_minus)
    z_new[io] = torch.bernoulli(p)

    return z_new


def gibbs_binary(target, num_dims, num_chains, num_samples, steps_per_sample=1, burnin_steps=0, z_init=None, device=None):
    # target := posterior density
    # num_dims := number of dimensions
    # num_samples := number of samples to be generated
    # burnin_steps := burn-in step for MCMC chain
    # steps_per_sample := thinning by storing only every kth sample
    # z_init := initial z value

    if not device: device = next(iter(target.parameters())).device

    samples = torch.zeros(num_chains, num_samples, num_dims).to(device)
    if z_init is None: z_init = torch.randint(2, size=(num_chains, num_dims)).float().to(device)

    with torch.no_grad():

        z = z_init

        # Burnin
        for i in range(burnin_steps):
            z = gibbs_core(z_old=z, target=target)
            print('Burnin {}/{}'.format(i+1, burnin_steps), end='\r')

        # Sampling
        num_accept = 0.0
        for i in range(num_samples):
            for k in range(steps_per_sample):
                z = gibbs_core(z_old=z, target=target)

            samples[:,i] = z
            print('{}/{}'.format(i+1, num_samples), end='\r')

    return samples
