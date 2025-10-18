from src.preprocessor import Preprocessor

pp = Preprocessor(
    data_dir="./data/3k_pcmb/filtered_gene_bc_matrices/hg19/",
    out_filename="preprocessed_3k_pcmb.h5ad",
    min_genes=100,
    min_cells=3,
    num_hvgs=2000,
)

pp.process()
