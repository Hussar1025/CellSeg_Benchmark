import pandas as pd

coord = pd.read_csv(
    "/home/data/vip232152/Xenium_1/cellist_result/seg_output_d28_swapped/"
    "alpha_0.8_sigma_1.0_beta_10_gene_Frequent_dist_15_twostep_False_cyto_False_noise_0.25_neigh_2.5/"
    "MyPancreas_focus_d28_swapped_Cellist_segmentation_cell_coord.txt",
    sep="\t"
)

print(coord.shape)

print("cells =", coord["Cellist"].nunique())

print("median spots =", coord["nSpot"].median())

print("mean spots =", coord["nSpot"].mean())

print("max spots =", coord["nSpot"].max())
