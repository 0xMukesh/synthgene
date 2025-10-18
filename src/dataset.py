import torch
import scanpy as sc
from torch.utils.data import Dataset


class PBMC3kDataset(Dataset):
    def __init__(self, h5ad_file_path: str) -> None:
        super().__init__()

        self.h5ad_file_path = h5ad_file_path
        self.adata = sc.read_h5ad(self.h5ad_file_path)

    def __len__(self) -> int:
        return self.adata.n_obs

    def __getitem__(self, idx) -> torch.Tensor:
        if self.adata.X is None:
            raise ValueError("gene expression data is none")

        gene_expr = self.adata.X[idx]
        tensor = torch.tensor(gene_expr, dtype=torch.float32)

        return tensor
