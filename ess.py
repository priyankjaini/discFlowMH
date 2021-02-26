import torch


def W(samples):
    '''
    Compute within-chain variance.

    Args:
        samples (Tensor): MCMC samples of shape (num_chains, num_samples, num_dims).
    '''
    return samples.var(1).mean(0) # (num_dims,)


def B(samples):
    '''
    Compute between-chain variance.

    Args:
        samples (Tensor): MCMC samples of shape (num_chains, num_samples, num_dims).
    '''
    return samples.shape[1] * samples.mean(1).var(0) # (num_dims,)


def var_plus(samples):
    '''
    Compute Eq. (11.3) in Bayesian Data Analysis, 3. Ed.

    Args:
        samples (Tensor): MCMC samples of shape (num_chains, num_samples, num_dims).
    '''
    num_chains, num_samples, num_dims = samples.shape
    return (B(samples) + (num_samples-1) * W(samples)) / num_samples


def variogram(samples, lags=None, verbose=False, eps=1e-12):
    '''
    Compute variogram.

    Args:
        samples (Tensor): Tensor of shape (num_chains, num_samples, num_dims).
        lags (int): Number of lags for which to compute correlation.
        eps (float): Small constant added to avoid dividing by zero variance.
    '''
    num_chains, num_samples, num_dims = samples.shape

    if lags == None or lags == 0:
        lags = num_samples-1
    assert lags < num_samples

    V = torch.ones(num_chains, lags, num_dims) # (C,L,D)
    for lag in range(1,lags+1):
        V[:,lag-1] = torch.mean((samples[:,lag:,] - samples[:,:-lag,])**2, dim=1).cpu() + eps # (C,D)
        if verbose: print('Computing variogram: Lag {}/{}'.format(lag, lags), end='\r')
    if verbose: print('')
    return V.to(samples.device) # (C,L,D)


def _get_rho_sum(rho):
    '''
    Computes sum of rho up to the point where two consecutive rhos are negative.
    This is described in Bayesian Data Analysis, 3. Ed. Sec. 11.5, and also in
    https://mc-stan.org/docs/2_24/reference-manual/effective-sample-size-section.html

    Args:
        rho (Tensor): Autocorrelation of shape (num_samples, num_dims).
    '''
    num_lags, num_dims = rho.shape

    rhoT = rho[1:1+2*((rho.shape[0]-1)//2)]
    rhoT = rhoT.reshape(rhoT.shape[0]//2, 2, -1).sum(1)

    mask = torch.ones(num_lags, num_dims).to(rho.device)
    for d in range(num_dims):
        negative_pairs = 2 * torch.where(rhoT[:,d]<=0)[0]
        if len(negative_pairs) > 0:
            T = negative_pairs[0]
            mask[T+1:,d] = 0.0

    return 1 + 2 * torch.sum(rho * mask, dim=0)


def ESS(samples, lag, verbose=False):
    if verbose: print('Computing ESS...')
    with torch.no_grad():
        num_chains, num_samples, num_dims = samples.shape # (B,L,D)
        V = variogram(samples, lag, verbose=verbose).mean(0) # (L,D)
        vp = var_plus(samples)+1e-12 # (D,)
        rho = 1 - V / (2*vp) # (L,D) [Eq. 11.7 in BDL 3. Ed]
        rho_sum = _get_rho_sum(rho) # (D,)
        ess_per_dim = num_samples / rho_sum # (D,)
        min, med, max = torch.min(ess_per_dim).item(), torch.median(ess_per_dim).item(), torch.max(ess_per_dim).item()
        mean, std = torch.mean(ess_per_dim).item(), torch.std(ess_per_dim).item()
    return min, med, max, mean, std
