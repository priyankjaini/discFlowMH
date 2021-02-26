import torch
from survae.distributions import ConditionalDistribution
from survae.transforms.surjections import Surjection


class FloorSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class VariationalQuantization(Surjection):
    '''
    A variational quantization layer.
    This is useful for converting continuous variables to discrete.
    Forward:
        `z = floor(x)`
        where `x` is continuous, `x \in [0, 1]^D`
    Inverse:
        `x = z + u, where u ~ decoder(context=z)`
    Args:
        decoder: ConditionalDistribution, a conditional distribution/flow which
            outputs samples in `[0,1]^D` conditioned on `z`.
        num_bits: int, number of bits in quantization,
            i.e. 8 for `x \in {0,1,2,...,255}^D`
            or 5 for `x \in {0,1,2,...,31}^D`.
    References:
        [1] RNADE: The real-valued neural autoregressive density-estimator,
            Uria et al., 2013, https://arxiv.org/abs/1306.0186
        [2] Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design,
            Ho et al., 2019, https://arxiv.org/abs/1902.00275
    '''

    stochastic_forward = False

    def __init__(self, decoder, num_bits=8, ste=True, long=False):
        super(VariationalQuantization, self).__init__()
        assert isinstance(decoder, ConditionalDistribution)
        assert not (ste and long)
        self.num_bits = num_bits
        self.quantization_bins = 2**num_bits
        self.register_buffer('ldj_per_dim', torch.log(torch.tensor(self.quantization_bins, dtype=torch.float)))
        self.decoder = decoder
        if ste: self.floor = FloorSTE().apply
        else:   self.floor = lambda x: torch.floor(x).long() if long else torch.floor(x)

    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = shape[1:].numel()
        ldj = self.ldj_per_dim * num_dims
        return ldj.repeat(batch_size)

    def forward(self, x):
        assert x.min() >= 0 and x.max() <= 1
        x = self.quantization_bins * x
        z = self.floor(x)
        u = x-z
        pu = self.decoder(u, context=z, mode='log_prob')
        ldj = self._ldj(z.shape) + pu
        return z, ldj

    def inverse(self, z):
        u = self.decoder(context=z, mode='sample')
        x = (z.type(u.dtype) + u) / self.quantization_bins
        return x
