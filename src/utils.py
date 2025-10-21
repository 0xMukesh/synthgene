import torch
import scanpy as sc
import scipy.sparse as sp
import numpy as np
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


def run_gen_eval(
    adata: ad.AnnData,
    gen: nn.Module,
    critic: nn.Module,
    n_cells: int,
    latent_dim: int,
    device: Device,
):
    if adata.X is None:
        raise ValueError("adata.X is None")

    gen.eval()
    critic.eval()

    with torch.no_grad():
        noise = torch.randn(n_cells, latent_dim).to(device)
        synthetic_data = gen(noise)

    synthetic_data = synthetic_data.detach().cpu().numpy()
    real_data = adata.X[:n_cells, :]

    if sp.issparse(real_data):
        real_data = real_data.toarray()  # type: ignore
    else:
        real_data = np.asarray(real_data)

    combined_data = np.concatenate([real_data, synthetic_data], axis=0)
    combined_adata = ad.AnnData(X=combined_data)
    combined_adata.var_names = adata.var_names.to_list()

    cell_types = ["real"] * n_cells + ["synthetic"] * n_cells
    combined_adata.obs["cell_type"] = cell_types

    sc.pp.pca(combined_adata, n_comps=min(30, combined_adata.n_vars - 1))
    sc.pp.neighbors(combined_adata, n_pcs=30)
    sc.tl.tsne(combined_adata)

    _, ax = plt.subplots(figsize=(12, 8))

    sc.pl.tsne(
        combined_adata,
        color="cell_type",
        palette={"real": "blue", "synthetic": "red"},
        size=50,
        ax=ax,
        show=False,
        title=f"t-SNE visualization",
    )

    plt.tight_layout()
    plt.show()
