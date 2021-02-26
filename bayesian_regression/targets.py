import torch
from survae.distributions import Distribution
import numpy as np


class BayesianRegression(Distribution):
    def __init__(self, data, y, w):
        super(BayesianRegression, self).__init__()
        self.register_buffer('data', data) # (N,D)
        self.register_buffer('y', y) # (N,)
        self.register_buffer('y_inner', torch.sum(y**2))
        self.m = data.shape[0] # Number of data points
        self.n = data.shape[1] # Number of features
        self.w = w # default = 4.0
        self.lambd = np.linalg.lstsq(data.numpy(), y.numpy(), rcond=None)[1].item()/(data.shape[0] - data.shape[1])
        self.nu_sq = 10./self.lambd
        self.register_buffer('Sigma', self.data.t() @ self.data + torch.eye(data.shape[1]) * self.nu_sq **-1) # (D,D)

    @property
    def size(self):
        return self.n

    def log_prob_legacy(self, theta):
        '''PyTorch adaptation of https://github.com/jvorstrupgoldman/tabu_dc_dzz.'''
        batch_size, num_dims = theta.shape
        energy = theta.new_zeros(batch_size)
        for i in range(batch_size):
            indices = torch.nonzero(theta[i], as_tuple=True)[0]
            active_parameters = int(theta[i].sum().item())
            data_subset = self.data[:, indices] # (m, act_params)
            C = torch.cholesky(data_subset.t() @ data_subset + torch.eye(active_parameters).to(data_subset.device)* self.nu_sq **-1) # (act_params, act_params)
            C_inv = torch.inverse(C) # (act_params, act_params)
            b = data_subset.t() @ self.y # (act_params, )
            C_inv_b = C_inv @ b # (act_params, )
            sigma_estimate = 1/self.m * (self.y_inner - C_inv_b.t() @ C_inv_b)
            energy[i] = -(- torch.diag(C).log().sum() - active_parameters * np.log(np.sqrt(self.nu_sq)) - (self.w+self.m)/2 * torch.log(self.w*self.lambd/self.m + sigma_estimate))
        return -energy

    def log_prob_iterative(self, theta):
        '''Differentiable wrt. theta, but still inefficient.'''
        batch_size, num_dims = theta.shape
        energy = theta.new_zeros(batch_size)
        for i in range(batch_size):
            Sigma = self.Sigma * theta[i] * theta[i].unsqueeze(-1) # (D,D)
            d = torch.diag(1-theta[i]) # (D,D)
            # U, P, V = torch.svd(Sigma+d)
            # Sigma_inv = U @ torch.diag(1/P) @ V.t()# - d
            # log_det = P.log().sum()
            Sigma_inv = torch.inverse(Sigma + d) # (D,D)
            log_det = torch.logdet(Sigma + d)
            b = (self.data * theta[i]).t() @ self.y # (D,)
            sigma_estimate = 1/self.m * (self.y_inner - b.t() @ Sigma_inv @ b)
            num_params = theta[i].sum()
            energy[i] = -(- 0.5 * log_det - num_params * np.log(np.sqrt(self.nu_sq)) - (self.w+self.m)/2 * torch.log(self.w*self.lambd/self.m + sigma_estimate))
        return -energy

    def log_prob(self, theta):
        '''Differentiable wrt. theta and parallelized.'''
        batch_size, num_dims = theta.shape
        Sigma = torch.einsum('de,bd,be->bde', self.Sigma, theta, theta) # (B,D,D)
        d = theta.new_zeros(batch_size, num_dims, num_dims) # (B,D,D)
        for i in range(batch_size): d[i] = torch.diag(1-theta[i])
        Sigma_inv = torch.inverse(Sigma + d) # (B,D,D)
        log_det = torch.logdet(Sigma + d) # (B,)
        b = torch.einsum('nd,bd,n->bd', self.data, theta, self.y) # (B,D)
        sigma_estimate = 1/self.m * (self.y_inner - torch.einsum('bd,bde,be->b', b, Sigma_inv, b)).clamp(min=1e-12) # (B,)
        num_params = theta.sum(1) # (B,)
        term1 = - 0.5 * log_det
        term2 = - num_params * np.log(np.sqrt(self.nu_sq))
        term3 = - (self.w+self.m)/2 * torch.log((self.w*self.lambd/self.m + sigma_estimate))
        energy = -(term1 + term2 + term3)
        return -energy
