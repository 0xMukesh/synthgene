from src.preprocessor import Preprocessor
from src.dataset import PBMC3kDataset
from src.decoder import Decoder

pp = Preprocessor(
    data_dir="./data/3k_pbmc/filtered_gene_bc_matrices/hg19/",
    out_filename="preprocessed_3k_pbmc.h5ad",
    min_genes=100,
    min_cells=3,
    num_hvgs=2000,
)

adata = pp.process()

dataset = PBMC3kDataset(
    "./data/3k_pbmc/filtered_gene_bc_matrices/hg19/preprocessed_3k_pbmc.h5ad"
)

decoder = Decoder(adata)
