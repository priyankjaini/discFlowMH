import torch.nn as nn
from survae.flows import Flow, ConditionalFlow
from survae.distributions import StandardNormal
from survae.transforms import AdditiveCouplingBijection, AffineCouplingBijection
from survae.transforms import ConditionalAdditiveCouplingBijection, ConditionalAffineCouplingBijection
from survae.transforms import ActNormBijection, Reverse, Shuffle, Linear, Sigmoid, Logit
from survae.nn.layers import LambdaLayer, ElementwiseParams, scale_fn as sfn
from survae.nn.nets import MLP
from quantization_variational import VariationalQuantization


class CouplingFlow(Flow):

    def __init__(self, target, num_bits, num_flows, actnorm, affine, scale_fn,
                 hidden_units, activation, permutation, context_size):

        D = target.size # Number of target dimensions
        P = 2 if affine else 1 # Number of elementwise parameters

        I = D // 2
        O = D // 2 + D % 2

        # Flow
        transforms = []
        for _ in range(num_flows):
            net = nn.Sequential(MLP(I, P*O,
                                    hidden_units=hidden_units,
                                    activation=activation),
                                ElementwiseParams(P))
            if affine: transforms.append(AffineCouplingBijection(net, scale_fn=sfn(scale_fn), num_condition=I))
            else:      transforms.append(AdditiveCouplingBijection(net, num_condition=I))
            if actnorm: transforms.append(ActNormBijection(D))
            if permutation == 'reverse':   transforms.append(Reverse(D))
            elif permutation == 'shuffle': transforms.append(Shuffle(D))
            elif permutation == 'learned': transforms.append(Linear(D))
        transforms.pop()

        transforms.append(Sigmoid())
        decoder = Decoder(target_size=D,
                          num_flows=num_flows,
                          actnorm=actnorm,
                          affine=affine,
                          scale_fn=scale_fn,
                          hidden_units=hidden_units,
                          activation=activation,
                          permutation=permutation,
                          context_size=context_size)
        transforms.append(VariationalQuantization(decoder, num_bits=num_bits))

        super(CouplingFlow, self).__init__(base_dist=target,
                                           transforms=transforms)


class Decoder(ConditionalFlow):

    def __init__(self, target_size, num_flows, actnorm, affine, scale_fn,
                 hidden_units, activation, permutation, context_size):

        D = target_size # Number of data dimensions
        P = 2 if affine else 1 # Number of elementwise parameters
        C = context_size # Size of context

        I = D // 2
        O = D // 2 + D % 2

        # Decoder
        transforms = [Logit()]
        for _ in range(num_flows):
            net = nn.Sequential(MLP(C+I, P*O,
                                    hidden_units=hidden_units,
                                    activation=activation),
                                ElementwiseParams(P))
            context_net = nn.Sequential(LambdaLayer(lambda x: 2*x.float() - 1),
                                        MLP(D, C,
                                            hidden_units=hidden_units,
                                            activation=activation))
            if affine: transforms.append(ConditionalAffineCouplingBijection(coupling_net=net, context_net=context_net, scale_fn=sfn(scale_fn), num_condition=I))
            else:           transforms.append(ConditionalAdditiveCouplingBijection(coupling_net=net, context_net=context_net, num_condition=I))
            if actnorm: transforms.append(ActNormBijection(D))
            if permutation == 'reverse':   transforms.append(Reverse(D))
            elif permutation == 'shuffle': transforms.append(Shuffle(D))
            elif permutation == 'learned': transforms.append(Linear(D))
        transforms.pop()
        super(Decoder, self).__init__(base_dist=StandardNormal((D,)), transforms=transforms)
