import warnings
from src.preprocessor import Preprocessor
from src.dataset import PBMC3kDataset

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

pp = Preprocessor(
    data_dir="./data/3k_pbmc/filtered_gene_bc_matrices/hg19/",
    out_filename="preprocessed_3k_pbmc.h5ad",
    min_genes=100,
    min_cells=3,
    num_hvgs=2000,
)

dataset = PBMC3kDataset(
    "./data/3k_pbmc/filtered_gene_bc_matrices/hg19/preprocessed_3k_pbmc.h5ad"
)
