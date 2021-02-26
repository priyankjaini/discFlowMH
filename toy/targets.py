import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily
from survae.distributions import Distribution
from survae.utils import sum_except_batch


class Cat(Distribution):
    def __init__(self, num_dims, num_bits):
        super(Cat, self).__init__()
        self.num_dims = num_dims
        self.num_bits = num_bits
        self.register_buffer('logits', torch.rand(self.num_dims, 2**self.num_bits))
    @property
    def dist(self):
        return Categorical(logits=self.logits)
    def log_prob(self, x):
        return sum_except_batch(self.dist.log_prob(x))
    def sample(self, num_samples):
        return self.dist.sample((num_samples,))


class GMM(Distribution):

    def __init__(self, num_dims, num_mix):
        super(GMM, self).__init__()
        self.num_dims = num_dims
        self.num_mix = num_mix
        self.register_buffer('logit_pi', torch.rand(self.num_mix))
        self.register_buffer('loc', 0.75*(2*torch.rand(self.num_mix, self.num_dims)-1))
        self.register_buffer('log_scale', -2.5+1.0*torch.rand(self.num_mix, self.num_dims))

    @property
    def dist(self):
        mix = Categorical(logits=self.logit_pi)
        comp = Independent(Normal(self.loc, self.log_scale.exp()), 1)
        return MixtureSameFamily(mix, comp)

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))


class DGMM(GMM):

    def __init__(self, num_dims, num_mix, num_bits):
        super(DGMM, self).__init__(num_dims, num_mix)
        self.num_bits = num_bits
        self.eps = torch.finfo(self.loc.dtype).eps

    def log_prob(self, x):
        log_prob = torch.zeros(x.shape[0], self.num_mix)
        for d in range(self.num_dims):
            xd_low = 2 * (x[:,d] / 2**self.num_bits) - 1
            xd_high = 2 * ((x[:,d]+1.0) / 2**self.num_bits) - 1
            xd_low[x[:,d] == 0]                     = -1e16
            xd_high[x[:,d] == 2**self.num_bits-1]   = +1e16
            for m in range(self.num_mix):
                dm = Normal(self.loc[m,d], self.log_scale[m,d].exp())
                prob_dm = dm.cdf(xd_high) - dm.cdf(xd_low)
                log_prob[:,m] += torch.log(prob_dm+self.eps)
        return torch.logsumexp(log_prob + torch.log_softmax(self.logit_pi, dim=-1), dim=-1)

    def sample(self, num_samples):
        samples = self.dist.sample((num_samples,))
        s = (2**self.num_bits) * (samples+1.0)/2
        return s.round().clamp(min=0.0, max=2**self.num_bits-1)
