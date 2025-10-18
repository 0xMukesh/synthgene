import scanpy as sc
import numpy as np
import os


class Preprocessor:
    def __init__(
        self,
        data_dir: str,
        out_filename: str,
        min_genes: int,
        min_cells: int,
        num_hvgs: int,
    ) -> None:
        self.data_dir = data_dir
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.num_hvgs = num_hvgs
        self.out_filename = out_filename

    def process(self):
        self._read_10x_mtx_file()
        self._filter_low_count_cells_and_genes()
        self._threshold_filtering()
        self._library_size_normalization()
        self._extract_hvgs()
        self._regress_out_values_and_scale()

        self.adata.write(os.path.join(self.data_dir, self.out_filename))

    def _read_10x_mtx_file(self):
        self.adata = sc.read_10x_mtx(
            path=self.data_dir, var_names="gene_symbols", cache=True
        )
        self.adata.var_names_make_unique()

    # filter out the cells which contain very small amount of gene content
    # filter out genes which are present in very less number of cells
    def _filter_low_count_cells_and_genes(self):
        sc.pp.filter_cells(self.adata, min_genes=self.min_genes)
        sc.pp.filter_genes(self.adata, min_cells=self.min_cells)

    # filter out data by thresholding the number of genes present in a cell and % of mitochrondrial genes (MT)
    # too high `n_genes_by_count` possibly means that it is a double droplet
    # too low `n_genes_by_count` possibly means that the cell is damaged/dying
    # too high `pct_counts_mt` means that the cell membrane is broken and cytoplasmic mRNA is leaking out
    def _threshold_filtering(self):
        self.adata.var["mt"] = self.adata.var_names.str.startswith("MT-")

        sc.pp.calculate_qc_metrics(
            self.adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )

        upper_lim = np.quantile(self.adata.obs["n_genes_by_counts"].to_numpy(), 0.97)
        lower_lim = np.quantile(self.adata.obs["n_genes_by_counts"].to_numpy(), 0.03)

        self.adata = self.adata[
            (self.adata.obs["n_genes_by_counts"] < upper_lim)
            & (self.adata.obs["n_genes_by_counts"] > lower_lim)
            & (self.adata.obs["pct_counts_mt"] < 5),
            :,
        ].copy()

    def _library_size_normalization(self):
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)

    def _extract_hvgs(self):
        sc.pp.highly_variable_genes(
            self.adata,
            n_top_genes=self.num_hvgs,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            flavor="seurat_v3",
        )

    def _regress_out_values_and_scale(self):
        sc.pp.regress_out(self.adata, ["total_counts", "pct_counts_mt"])
        sc.pp.scale(self.adata, max_value=10)

        self.adata.raw = self.adata.copy()
        self.adata = self.adata[:, self.adata.var["highly_variable"]]
