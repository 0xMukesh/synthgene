import scanpy as sc


class Preprocessor:
    def __init__(
        self,
        data_dir: str,
        out_file_path: str,
        min_genes: int,
        min_cells: int,
        num_jobs: int = 2,
    ) -> None:
        self.data_dir = data_dir
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.out_file_path = out_file_path
        self.num_jobs = num_jobs

    def process(self):
        sc.settings.n_jobs = self.num_jobs

        self._read_10x_mtx_file()
        self._filter_low_count_cells_and_genes()
        self._library_size_normalization()

        self.adata.write(self.out_file_path)

        return self.adata

    def _read_10x_mtx_file(self):
        self.adata = sc.read_10x_mtx(
            path=self.data_dir, var_names="gene_symbols", cache=True
        )
        self.adata.var_names_make_unique()

    def _filter_low_count_cells_and_genes(self):
        sc.pp.filter_cells(self.adata, min_genes=self.min_genes)
        sc.pp.filter_genes(self.adata, min_cells=self.min_cells)

    def _library_size_normalization(self):
        sc.pp.normalize_total(self.adata, target_sum=2e4)
