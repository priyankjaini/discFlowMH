import torch
import torch.nn.functional as F
from survae.distributions import Distribution
from survae.utils import sum_except_batch
from torch.distributions import Normal
from survae.data import DATA_PATH

# Datasets
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer


class LogisticRegressionTarget(Distribution):
    def __init__(self, X, y, num_bits, prior, cv_folds, normalize=True):
        super(LogisticRegressionTarget, self).__init__()
        if normalize: X = self._normalize_features(X)
        self.register_buffer('X', X.float()) # (N, F)
        self.register_buffer('y', y.long()) # (N,)
        self.num_bits = num_bits
        self.prior = prior
        self.cv_folds = cv_folds
        self.set_split(None)
        if prior=='gaussian':
            self.register_buffer('prior_loc', torch.zeros((self.size,)))
            self.register_buffer('prior_scale', torch.ones((self.size,)))

    def set_split(self, split):
        self.split = split
        if split is None:
            self.X_train = self.X.float() # (N, F)
            self.y_train = self.y.long() # (N,)
            self.X_test = None
            self.y_test = None
        else:
            X_train, y_train, X_test, y_test = self.split_data(self.X, self.y, split=split, folds=self.cv_folds)
            self.X_train = X_train.float() # (N_train, F)
            self.y_train = y_train.long() # (N_train,)
            self.X_test = X_test.float() # (N_test, F)
            self.y_test = y_test.long() # (N_test,)

    @staticmethod
    def _normalize_features(X):
        std = X.std(dim=0, keepdim=True)
        std[std==0.0] = 1.0
        return (X - X.mean(dim=0, keepdim=True)) / std

    @staticmethod
    def split_data(X, y, split, folds):
        N = X.shape[0]
        idx_test0 = split * (N//folds) + min(split, N%folds)
        idx_test1 = (split+1) * (N//folds) + min(split+1, N%folds)

        torch.manual_seed(0)
        perm = torch.randperm(N)
        idx_test = perm[idx_test0:idx_test1]
        idx_train = torch.cat([perm[:idx_test0], perm[idx_test1:]], dim=0)
        return X[idx_train], y[idx_train], X[idx_test], y[idx_test]

    def scale_theta(self, theta):
        assert theta.dim() == 2, 'theta should have shape (samples, {})'.format(self.size)
        assert theta.shape[1] == self.size, 'theta should have shape (samples, {})'.format(self.size)
        if self.num_bits is None:
            pass
        elif self.num_bits == 1:
            theta = 2*theta-1 # {0,1} -> {-1,1}
        elif self.num_bits >= 2:
            theta = theta-2**(self.num_bits-1) # e.g. {0,1,...,255} -> {-128,...,127}
        return theta

    def logits(self, theta, test=False):
        s = theta.shape[0]
        theta = theta.reshape(s, self.num_features+1, self.num_classes)
        w = theta[:,:self.num_features] # (samples, F, C)
        b = theta[:,self.num_features:] # (samples, 1, C)
        # (S, F, C), (N, F) -> (N, C, S); (S, 1, C) -> (1, C, S)
        X = self.X_test if test else self.X_train
        logits = torch.einsum('sfc,nf->ncs', w, X) + torch.einsum('soc->ocs', b)
        return logits

    def accuracy(self, theta, test=False):
        theta = self.scale_theta(theta)
        logits = self.logits(theta, test=test) # (N, C, S)
        probs = torch.softmax(logits, dim=1).mean(dim=2) # (N, C,)
        pred = probs.argmax(1) # (N,)
        y = self.y_test if test else self.y_train
        acc = (pred==y).float().mean()
        return acc

    def log_prior(self, theta):
        theta = self.scale_theta(theta)
        if self.prior == 'uniform':
            return 0
        else:
            prior = Normal(loc=self.prior_loc, scale=self.prior_scale)
            if self.num_bits is None:
                return sum_except_batch(prior.log_prob(theta))
            else:
                return sum_except_batch(torch.log(prior.cdf(theta+1)-prior.cdf(theta)+1e-12))


    def log_likelihood(self, theta, test=False):
        theta = self.scale_theta(theta)
        s = theta.shape[0]
        logits = self.logits(theta, test=test)
        y = self.y_test if test else self.y_train
        loglik = -F.cross_entropy(logits, y.unsqueeze(-1).repeat(1,s), reduction='none') # (N, C, S), (N, S) -> (N, S)
        return loglik.sum(dim=0) # (N, S) -> (S,)

    def log_prob(self, theta, test=False):
        return self.log_likelihood(theta, test=test) + self.log_prior(theta)

    @property
    def size(self):
        return (self.num_features + 1) * self.num_classes

    @property
    def num_features(self):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()


class LogisticRegressionBreastCancer(LogisticRegressionTarget):
    def __init__(self, num_bits, prior, cv_folds, normalize=True):
        X, y = load_breast_cancer(return_X_y=True)
        super(LogisticRegressionBreastCancer, self).__init__(X=torch.from_numpy(X),
                                                             y=torch.from_numpy(y),
                                                             num_bits=num_bits,
                                                             prior=prior,
                                                             cv_folds=cv_folds,
                                                             normalize=normalize)

    @property
    def num_features(self):
        return 30

    @property
    def num_classes(self):
        return 2

class LogisticRegressionIris(LogisticRegressionTarget):
    def __init__(self, num_bits, prior, cv_folds, normalize=True):
        X, y = load_iris(return_X_y=True)
        super(LogisticRegressionIris, self).__init__(X=torch.from_numpy(X),
                                                     y=torch.from_numpy(y),
                                                     num_bits=num_bits,
                                                     prior=prior,
                                                     cv_folds=cv_folds,
                                                     normalize=normalize)

    @property
    def num_features(self):
        return 4

    @property
    def num_classes(self):
        return 3


class LogisticRegressionWine(LogisticRegressionTarget):
    def __init__(self, num_bits, prior, cv_folds, normalize=True):
        X, y = load_wine(return_X_y=True)
        super(LogisticRegressionWine, self).__init__(X=torch.from_numpy(X),
                                                     y=torch.from_numpy(y),
                                                     num_bits=num_bits,
                                                     prior=prior,
                                                     cv_folds=cv_folds,
                                                     normalize=normalize)

    @property
    def num_features(self):
        return 13

    @property
    def num_classes(self):
        return 3
