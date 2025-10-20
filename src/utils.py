import torch
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from torch import nn
from typing import Literal, TypeAlias


Device: TypeAlias = Literal["cuda", "cpu"]


def calculate_grad_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: Device,
) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1).to(device)

    mixed = alpha * real + (1 - alpha) * fake
    mixed.requires_grad_(True)

    critic_mixed = critic(mixed)

    gradient = torch.autograd.grad(
        inputs=mixed,
        outputs=critic_mixed,
        create_graph=True,
        grad_outputs=torch.ones_like(critic_mixed),
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.size(0), -1)

    gradient_norm = torch.norm(gradient, p=2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty


def compute_umap(
    adata: ad.AnnData, n_neighbors: int = 10, n_pcs: int = 30, random_state: int = 0
):
    sc.pp.neighbors(adata, n_neighbors, n_pcs)
    sc.tl.leiden(
        adata,
        0.7,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=random_state,
    )
    sc.tl.umap(adata, random_state=random_state)


def run_eval(
    real_adata: ad.AnnData,
    gen: nn.Module,
    critic: nn.Module,
    latent_dim: int,
    device: Device,
):
    gen.eval()
    critic.eval()

    with torch.no_grad():
        noise = torch.randn(real_adata.n_obs, latent_dim).to(device)
        synthetic_data = gen(noise)

    synthetic_data = gen(noise).detach().cpu().numpy()
    synthetic_adata = ad.AnnData(X=synthetic_data)
    synthetic_adata.var_names = real_adata.var_names.to_list()

    compute_umap(synthetic_adata)
    compute_umap(real_adata)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sc.pl.umap(
        real_adata,
        color="leiden",
        ax=ax1,
        show=False,
        title="umap of real data",
    )

    sc.pl.umap(
        synthetic_adata,
        color="leiden",
        ax=ax2,
        show=False,
        title="umap of synthetic data",
    )

    plt.tight_layout()
    plt.show()
