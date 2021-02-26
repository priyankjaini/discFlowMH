import torch
from survae.data.datasets.image import UnsupervisedMNIST
from survae.distributions import Distribution
from survae.data.transforms import StaticBinarize
from torchvision.transforms import Compose, Resize, ToTensor


class IsingMNIST(Distribution):
    def __init__(self, idx, downsample, corruption_prob, beta, mu):
        super(IsingMNIST, self).__init__()
        self.beta = beta
        # Load img
        assert downsample in {0,1,2}
        self.N = 28 // (2**downsample)
        dataset = UnsupervisedMNIST(transform=Compose([Resize(self.N), ToTensor(), StaticBinarize()]))
        img = dataset[idx].squeeze(0).float()
        # Create corrupted img
        torch.manual_seed(0)
        corruption_matrix = torch.bernoulli(torch.ones(img.shape)*corruption_prob)
        img_corrupted = (img + corruption_matrix) % 2
        # Convert from {0,1} to {1-,1}
        self.img = 2 * img - 1
        self.img_corrupted = 2 * img_corrupted - 1
        self.mu = mu

    @property
    def size(self):
        return self.N**2

    def img2vec(self, img):
        return img.view(img.shape[0], self.N**2)

    def vec2img(self, vec):
        return vec.view(vec.shape[0], self.N, self.N)

    def log_prob(self, theta):
        img = 2*self.vec2img(theta)-1
        energy = 0.0
        for i in range(self.N):
            for j in range(self.N):
                cur = img[:,i, j]
                neighbours = img[:,(i+1)%self.N, j] + img[:,i,(j+1)%self.N] + img[:,(i-1)%self.N, j] + img[:,i,(j-1)%self.N]
                energy += - cur * neighbours - self.mu * cur * self.img_corrupted[i,j]
        return - self.beta * energy
