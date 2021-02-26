import copy
import torch


def mcmc_core(z_old, log_prob_old, pi, proposal_scale):
    num_chains, num_dims = z_old.shape

    z_new = z_old + torch.randn((num_chains, num_dims)).to(z_old.device) * proposal_scale
    log_prob_new = pi.log_prob(z_new)

    A = torch.exp(log_prob_new - log_prob_old)
    accepted = torch.rand((num_chains,)).to(z_old.device) < A

    return z_new, log_prob_new, accepted


def metropolis_hastings(pi, num_dims, num_chains, num_samples, steps_per_sample=1, burnin_steps=0, proposal_scale=1.0, z_init=None, device=None):
    # pi := posterior density
    # num_dims := number of dimensions
    # num_samples := number of samples to be generated
    # burnin_steps := burn-in step for MCMC chain
    # steps_per_sample := thinning by storing only every kth sample
    # proposal_scale := scale for the proposal distribution
    # z_init := initial z value

    if not device: device = next(iter(pi.parameters())).device

    samples = torch.zeros(num_chains, num_samples, num_dims).to(device)
    if z_init is None: z_init = torch.randn(num_chains, num_dims).to(device)

    with torch.no_grad():

        z = z_init
        log_prob = pi.log_prob(z)

        # Burnin
        for i in range(burnin_steps):

            z_new, log_prob_new, accepted = mcmc_core(z_old=z,
                                                      log_prob_old=log_prob,
                                                      pi=pi,
                                                      proposal_scale=proposal_scale)

            z[accepted] = z_new[accepted]
            log_prob[accepted] = log_prob_new[accepted]
            print('Burnin {}/{}'.format(i+1, burnin_steps), end='\r')

        # Sampling
        num_accept = 0.0
        for i in range(num_samples):
            for k in range(steps_per_sample):

                z_new, log_prob_new, accepted = mcmc_core(z_old=z,
                                                          log_prob_old=log_prob,
                                                          pi=pi,
                                                          proposal_scale=proposal_scale)

                z[accepted] = z_new[accepted]
                log_prob[accepted] = log_prob_new[accepted]

                num_accept += accepted.float().sum().cpu().item()

                accept_rate = num_accept / (num_chains*(steps_per_sample*i+k+1))

            samples[:,i] = z
            print('{}/{}, Accept rate: {:.3f}'.format(i+1, num_samples, accept_rate), end='\r')

    return samples, accept_rate
