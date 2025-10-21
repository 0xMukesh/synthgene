import torch
import os
from torch.utils.data import DataLoader
from dataclasses import dataclass

from src.preprocessor import Preprocessor
from src.datasets.pbmc3k import PBMC3kDataset
from src.models.wgan import Generator, Critic
from src.utils import calculate_grad_penalty, run_eval


@dataclass
class Config:
    data_dir = "/kaggle/input/pbmc-3k-dataset"
    checkpoints_dir = "checkpoints"
    preprocessed_out_file_path = "/kaggle/working/preprocessed_3k_pbmc.h5ad"

    epochs = 6
    batch_size = 32
    lr_gen = 1e-4
    lr_critic = 1e-4
    critic_iter = 5
    lambda_gp = 10

    min_cells = 3
    min_genes = 10

    latent_dim = 128
    n_blocks = 2
    base_features = 256
    library_size = 20_000

    print_after_every = 20
    save_after_every = 5
    run_eval_after_every = 2


config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

pp = Preprocessor(
    config.data_dir,
    config.preprocessed_out_file_path,
    config.min_genes,
    config.min_cells,
)

adata = pp.process()
os.makedirs(config.checkpoints_dir, exist_ok=True)

dataset = PBMC3kDataset(config.preprocessed_out_file_path)
data_loader = DataLoader(
    dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
)

gen = Generator(
    config.latent_dim, adata.n_vars, config.n_blocks, config.base_features
).to(device)
critic = Critic(adata.n_vars, config.n_blocks, config.base_features).to(device)

optim_gen = torch.optim.Adam(gen.parameters(), lr=config.lr_gen, betas=(0.0, 0.9))
optim_critic = torch.optim.Adam(
    critic.parameters(), lr=config.lr_critic, betas=(0.0, 0.9)
)

for epoch in range(config.epochs):
    gen.train()
    critic.train()

    epoch_loss_gen = 0.0
    epoch_loss_critic = 0.0

    for batch_idx, real in enumerate(data_loader):
        real = real.to(device)
        critic_batch_loss = 0.0

        # critic: max(E[critic(real)] - E[critic(gen(noise))])
        for _ in range(config.critic_iter):
            noise = torch.randn(real.shape[0], config.latent_dim).to(device)
            fake = gen(noise)

            C_real = critic(real)
            C_fake = critic(fake.detach())

            gp = calculate_grad_penalty(critic, real, fake, device)

            C_loss = torch.mean(C_fake) - torch.mean(C_real) + config.lambda_gp * gp
            critic_batch_loss += C_loss.item()

            optim_critic.zero_grad()
            C_loss.backward()
            optim_critic.step()

        # generator: min(E[critic(gen(noise))])
        noise = torch.randn(real.size(0), config.latent_dim).to(device)
        fake = gen(noise)

        G_loss = -torch.mean(critic(fake))

        optim_gen.zero_grad()
        G_loss.backward()
        optim_gen.step()

        avg_batch_critic_loss = critic_batch_loss / config.critic_iter

        epoch_loss_gen += G_loss.item()
        epoch_loss_critic += avg_batch_critic_loss

        if batch_idx % config.print_after_every == 0:
            print(
                f"[epoch {epoch + 1}, batch {batch_idx + 1}/{len(data_loader)}] gen loss: {G_loss.item():.4f}, critic loss: {avg_batch_critic_loss:.4f}"
            )

    if (epoch + 1) % config.run_eval_after_every == 0:
        run_eval(adata, gen, critic, 2000, config.latent_dim, device)

    avg_epoch_loss_gen = epoch_loss_gen / len(data_loader)
    avg_epoch_loss_critic = epoch_loss_critic / len(data_loader)

    print(f"\n{'='*60}")
    print(f"epoch {epoch+1} completed")
    print(f"avg gen loss: {avg_epoch_loss_gen:.4f}")
    print(f"avg critic loss: {avg_epoch_loss_critic:.4f}")
    print(f"{'='*60}\n")

    if (epoch + 1) % config.save_after_every == 0:
        torch.save(
            {
                "epoch": epoch,
                "gen_state_dict": gen.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "gen_optim_state_dict": optim_gen.state_dict(),
                "critic_optim_state_dict": optim_critic.state_dict(),
                "gen_loss": avg_epoch_loss_gen,
                "critic_loss": avg_epoch_loss_critic,
            },
            os.path.join(config.checkpoints_dir, f"epoch_{epoch+1}.pth"),
        )
