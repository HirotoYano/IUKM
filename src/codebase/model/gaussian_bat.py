import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints


def model(N, obs, L, K):
    with pyro.plate("cluster_param_plate", N):
        theta = pyro.sample("theta", dist.Dirichlet(torch.ones(N, L)))

    return theta


def guide(N, obs, L, K):
    alpha = pyro.param("alpha", init_tensor=torch.ones(N, L), constraint=constraints.positive)
    with pyro.plate("cluster_pram_plate", N):
        pyro.sample("theta", dist.Dirichlet(alpha))


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super(Encoder, self).__init__()
        self.s: int = hidden_dim
        self.k: int = latent_dim
        self.input_to_hidden: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_latent: nn.Linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x) -> torch.Tensor:
        bn: nn.BatchNorm1d = nn.BatchNorm1d(self.s)
        h: torch.Tensor = bn(self.input_to_hidden(x))
        leak: nn.LeakyReLU = nn.LeakyReLU(0.1)
        o: torch.Tensor = torch.max(h, leak(h))
        softmax: nn.Softmax = nn.Softmax(dim=1)

        return softmax(self.hidden_to_latent(o))


class Generator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Generator, self).__init__()
        self.k: int = input_size
        self.s: int = hidden_size
        self.v: int = output_size
        self.input_to_hidden: nn.Linear = nn.Linear(input_size, hidden_size)
        self.hidden_to_output: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, theta) -> torch.Tensor:
        bn: nn.BatchNorm1d = nn.BatchNorm1d(self.s)
        h: torch.Tensor = bn(self.input_to_hidden(theta))
        leak: nn.LeakyReLU = nn.LeakyReLU(0.1)
        o: torch.Tensor = torch.max(h, leak(h))
        softmax: nn.Softmax = nn.Softmax(dim=1)

        return softmax(self.hidden_to_output(o))


class Discriminator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Discriminator, self).__init__()
        self.v_k: int = input_size
        self.s: int = hidden_size
        self.out: int = output_size
        self.input_to_hidden: nn.Linear = nn.Linear(input_size, hidden_size)
        self.hidden_to_output: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, p) -> torch.Tensor:
        return self.hidden_to_output(self.input_to_hidden(p))
