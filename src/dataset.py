import torch
from torch.utils.data import Dataset
import scanpy as sc
import numpy as np


class PBMC3kDataset(Dataset):
    def __init__(
        self,
        h5ad_file_path: str,
        train: bool,
        train_test_split_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.h5ad_file_path = h5ad_file_path
        adata_full = sc.read_h5ad(self.h5ad_file_path)

        n_cells = adata_full.n_obs
        indices = np.arange(n_cells)

        np.random.seed(seed)
        np.random.shuffle(indices)

        split_point = int(n_cells * train_test_split_ratio)

        if train:
            subset_indices = adata_full[:split_point]
        else:
            subset_indices = adata_full[split_point:]

        self.adata = adata_full[subset_indices, :]

    def __len__(self) -> int:
        return self.adata.n_obs

    def __getitem__(self, idx) -> torch.Tensor:
        if self.adata.X is None:
            raise ValueError("gene expression data is none")

        gene_expr = self.adata.X[idx]
        tensor = torch.tensor(gene_expr, dtype=torch.float32)

        return tensor
