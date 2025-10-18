import scanpy as sc
import numpy as np
import pandas as pd


class Decoder:
    def __init__(self, adata: sc.AnnData):
        self.adata = adata

        self.gene_means = self.adata.var["mean"].to_numpy()
        self.gene_stds = self.adata.var["std"].to_numpy()
        self.mean_total_counts = self.adata.obs["total_counts"].mean()
        self.hvg_names = self.adata.var_names

    def decode(self, output: np.ndarray):
        # inverse scaling
        unscaled_data = (output * self.gene_stds) + self.gene_means

        # inverse log-transform
        normalized_counts = np.expm1(unscaled_data)
        normalized_counts[normalized_counts < 0] = 0

        # inverse library size normalization
        approx_counts = normalized_counts * self.mean_total_counts / 1e4
        approx_counts = np.round(approx_counts).astype(int)

        if len(approx_counts.shape) == 1:
            approx_counts = approx_counts.reshape(1, -1)

        df = pd.DataFrame(approx_counts, columns=self.hvg_names)

        return df
