import torch
import os
from torch.utils.data import DataLoader
from dataclasses import dataclass

from src.preprocessor import Preprocessor
from src.dataset import PBMC3kDataset
from src.models.wgan import Generator, Critic
from src.utils import calculate_grad_penalty


@dataclass
class Config:
    data_dir = "data/3k_pbmc/filtered_gene_bc_matrices/hg19"
    preprocessed_out_filename = "preprocessed_3k_pbmc.h5ad"

    min_genes = 200
    min_cells = 3
    num_hvgs = 2000

    epochs = 3
    batch_size = 32

    latent_dim = 128
    n_blocks = 2
    base_features = 256

    lr_gen = 1e-4
    lr_critic = 1e-4
    critic_iter = 5
    lambda_gp = 10


config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

pp = Preprocessor(
    config.data_dir,
    config.preprocessed_out_filename,
    config.min_genes,
    config.min_cells,
    config.num_hvgs,
)

pp.process()

dataset = PBMC3kDataset(os.path.join(config.data_dir, config.preprocessed_out_filename))
data_loader = DataLoader(
    dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
)

gen = Generator(
    config.latent_dim, config.num_hvgs, config.n_blocks, config.base_features
).to(device)
critic = Critic(config.num_hvgs, config.n_blocks, config.base_features).to(device)

optim_gen = torch.optim.Adam(gen.parameters(), lr=config.lr_gen)
optim_critic = torch.optim.Adam(critic.parameters(), lr=config.lr_critic)

for epoch in range(config.epochs):
    for batch_idx, real in enumerate(data_loader):
        real = real.to(device)
        critic_batch_loss = 0.0

        for _ in range(config.critic_iter):
            noise = torch.randn(real.size(0), config.latent_dim).to(device)
            fake = gen(noise)

            C_real = critic(real)
            C_fake = critic(fake.detach())

            gp = calculate_grad_penalty(critic, real, fake, device)

            C_loss = torch.mean(C_fake) - torch.mean(C_real) + config.lambda_gp * gp
            critic_batch_loss += C_loss.item()

            optim_critic.zero_grad()
            C_loss.backward()
            optim_critic.step()

        noise = torch.randn(real.size(0), config.latent_dim).to(device)
        fake = gen(noise)

        G_loss = -torch.mean(critic(fake))

        optim_gen.zero_grad()
        G_loss.backward()
        optim_gen.step()

        print(
            f"G_loss: {G_loss.item():.4f} and C_loss: {(critic_batch_loss / config.critic_iter):.4f}"
        )
