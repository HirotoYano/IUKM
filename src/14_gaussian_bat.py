import os

import numpy as np
import polars as pl
import pyro
import pyro.distributions as dist
import torch
from dotenv import load_dotenv
from pyro.infer import SVI, TraceEnum_ELBO
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.model.gaussian_bat import Discriminator, Encoder, Generator, guide, model

load_dotenv("/workspace/src/.env")
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    doc_representation_path: str = (
        f"{OUTPUT_PATH}/pseudo_document_representation/2023-7-20/4-30-2/documents_representation.csv"
    )

    df_doc_representation: pl.DataFrame = data_loder(doc_representation_path, has_header=True)
    d_real: torch.Tensor = torch.from_numpy(df_doc_representation.to_numpy().astype(np.float32)).clone()
    encoder_input_dim: int = d_real.shape[1]
    encoder_hidden_dim: int = 10
    encoder_latent_dim: int = d_real.shape[0]

    encoder: Encoder = Encoder(encoder_input_dim, encoder_hidden_dim, encoder_latent_dim)
    theta_real: torch.Tensor = encoder(d_real)
    p_real: torch.Tensor = torch.cat([theta_real, d_real], dim=1)

    pyro.clear_param_store()

    adam_params = {"lr": 0.001, "betas": (0.9, 0.9)}
    optimizer = pyro.optim.Adam(adam_params)
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    pyro.set_rng_seed(123)
    svi = SVI(model, guide, optimizer, loss=elbo)

    for _ in tqdm(range(10000)):
        svi.step(d_real.shape[0], d_real, d_real.shape[0], 49)

    theta_fake: torch.Tensor = pyro.sample("theta_fake", dist.Dirichlet(pyro.param("alpha")))

    generator_input_dim: int = d_real.shape[0]
    generator_hidden_dim: int = 10
    generator_latent_dim: int = d_real.shape[1]

    generator: Generator = Generator(generator_input_dim, generator_hidden_dim, generator_latent_dim)
    d_fake: torch.Tensor = generator(theta_fake)
    p_fake: torch.Tensor = torch.cat([theta_fake, d_fake], dim=1)

    discriminator_input_dim: int = p_real.shape[1]
    discriminator_hidden_dim: int = 10
    discriminator_output_dim: int = 1

    discriminator: Discriminator = Discriminator(
        discriminator_input_dim, discriminator_hidden_dim, discriminator_output_dim
    )
    print(discriminator(p_fake))


if __name__ == "__main__":
    main()
