import copy
import torch


def transition_probs(z, proposal_prob, num_bits, reflective):
    num_chains, num_dims = z.shape
    probs = torch.ones((num_chains, num_dims, 3)) * torch.tensor([proposal_prob/2, 1-proposal_prob, proposal_prob/2])
    if not reflective:
        probs[z==0.0] = torch.tensor([0.0, 1-proposal_prob, proposal_prob])
        probs[z==2**num_bits-1] = torch.tensor([proposal_prob, 1-proposal_prob, 0.0])
    return probs


def mcmc_core(z_old, log_prob_old, pi, proposal_prob, num_bits, reflective):
    num_chains, num_dims = z_old.shape
    device = z_old.device

    probs = transition_probs(z_old, proposal_prob, num_bits, reflective).to(device)

    sample = torch.multinomial(probs.reshape(-1,3), num_samples=1, replacement=True).reshape(num_chains, num_dims)
    epsilon = sample.float()-1

    z_new = z_old + epsilon
    if reflective: z_new = z_new % (2**num_bits)
    log_prob_new = pi.log_prob(z_new)

    if reflective:
        A = torch.exp(log_prob_new - log_prob_old)
    else:
        raise NotImplementedError()
        # probs_rev = transition_probs(z_new, proposal_prob, num_bits)
        # log_prob_new_to_old = torch.log(probs[sample])
        # log_prob_old_to_new = torch.log(probs[sample])
        # A = torch.exp(log_prob_new + log_prob_new_to_old - log_prob_old - log_prob_old_to_new)

    accepted = torch.rand((num_chains,)).to(device)<A

    return z_new, log_prob_new, accepted


def metropolis_hastings_discrete(pi, num_dims, num_chains, num_samples, num_bits, steps_per_sample=1, burnin_steps=0, proposal_prob=0.5, reflective=True, z_init=None, device=None):
    # pi := posterior density
    # num_dims := number of dimensions
    # num_samples := number of samples to be generated
    # burnin_steps := burn-in step for MCMC chain
    # steps_per_sample := thinning by storing only every kth sample
    # proposal_prob := probability of proposing change for each coordinate
    # z_init := initial z value
    assert reflective

    if not device: device = next(iter(pi.parameters())).device

    samples = torch.zeros(num_chains, num_samples, num_dims).to(device)
    if z_init is None: z_init = torch.randint(2**num_bits, size=(num_chains, num_dims)).float().to(device)

    with torch.no_grad():

        z = z_init
        log_prob = pi.log_prob(z)

        # Burnin
        for i in range(burnin_steps):

            z_new, log_prob_new, accepted = mcmc_core(z_old=z,
                                                      log_prob_old=log_prob,
                                                      pi=pi,
                                                      proposal_prob=proposal_prob,
                                                      num_bits=num_bits,
                                                      reflective=reflective)

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
                                                          proposal_prob=proposal_prob,
                                                          num_bits=num_bits,
                                                          reflective=reflective)

                z[accepted] = z_new[accepted]
                log_prob[accepted] = log_prob_new[accepted]

                num_accept += accepted.float().sum().cpu().item()

                accept_rate = num_accept / (num_chains*(steps_per_sample*i+k+1))

            samples[:,i] = z
            print('{}/{}, Accept rate: {:.3f}'.format(i+1, num_samples, accept_rate), end='\r')

    return samples, accept_rate
