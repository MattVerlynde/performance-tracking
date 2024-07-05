"""
====================================================================
Clustering on SAR images with Riemannian geometry
====================================================================

This example compares clustering pipelines based on covariance matrices for
synthetic-aperture radar (SAR) image clustering [1]_ [2]_.
"""
# Author: Ammar Mian

import yaml
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from pyriemann.estimation import Covariances
from pyriemann.clustering import Kmeans
from helpers.processing_helpers import SlidingWindowVectorize

from utils_clustering import load_data

import argparse


###############################################################################
# Plot data and results
# ---------

def plot_data(data_visualization, X_image, Y_image, storage_path):

    print("Plotting")
    plot_value = 20*np.log10(np.sum(np.abs(data_visualization)**2, axis=2))
    figure, ax = plt.subplots(figsize=(5, 5))
    plt.pcolormesh(X_image, Y_image, plot_value, cmap="gray")
    plt.colorbar()
    ax.invert_yaxis()
    plt.xlabel("Range (m)")
    plt.ylabel("Azimuth (m)")
    plt.title(r"SAR data: $20\log_{10}(x_{HH}^2 + x_{HV}^2 + x_{VV}^2$)")
    plt.tight_layout()

    os.makedirs(os.path.join(storage_path, "plots"), exist_ok=True)
    figure.savefig(os.path.join(storage_path, "plots", "clustering_data.png"))

def plot_clustering(X_image, Y_image, X_ref, Y_ref, result, pipeline_name, number_run, storage_path):

    figure, ax = plt.subplots(figsize=(5, 5))
    plt.pcolormesh(X_res, Y_res, result, cmap="tab20b")
    plt.xlim(X_image.min(), X_image.max())
    plt.ylim(Y_image.min(), Y_image.max())
    plt.title(f"Clustering with {pipeline_name}")
    plt.colorbar()
    ax.invert_yaxis()
    plt.xlabel("Range (m)")
    plt.ylabel("Azimuth (m)")
    plt.tight_layout()
    
    os.makedirs(os.path.join(storage_path, "plots"), exist_ok=True)
    figure.savefig(os.path.join(storage_path, "plots", f"{pipeline_name}_clustering_results_{number_run}.png"))
    plt.show()

###############################################################################
# References
# ----------
# .. [1] `Statistical classification for heterogeneous polarimetric SAR images
#    <https://hal.science/hal-00638829/>`_
#    Formont, P., Pascal, F., Vasile, G., Ovarlez, J. P., & Ferro-Famil, L.
#    IEEE Journal of selected topics in Signal Processing, 5(3), 567-576. 2010.
#
# .. [2] `On the use of matrix information geometry for polarimetric SAR image
#    classification
#    <https://hal.science/hal-02494996v1>`_
#    Formont, P., Ovarlez, J. P., & Pascal, F.
#    In Matrix Information Geometry (pp. 257-276). 2012.

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.storage_path, "group_info.yaml"), 'r') as f:
        paramYaml = yaml.load(f, Loader=yaml.FullLoader)

    data_path = paramYaml['parameters']['--image'] + ".npy"
    window = int(paramYaml['parameters']['--window'])
    small_dataset = bool(paramYaml['parameters']['--small_dataset'])
    n_clusters = int(paramYaml['parameters']['--n_clusters'])

    data, data_visualization, X_image, Y_image, X_res, Y_res = load_data(data_path, n_clusters, window, small_dataset)
    plot_data(data_visualization, X_image, Y_image, args.storage_path)

    for file in os.listdir(os.path.join(args.storage_path, "output")):
        number_run = file.split("_")[-1].split(".")[0]
        name = file.split("_")[0].replace(" ", "_")
        result = np.load(os.path.join(args.storage_path, "output", file))

        plot_clustering(X_image, Y_image, X_res, Y_res, result, name, number_run, args.storage_path)
    
    print("Done")
    print(f"Plots saved in {args.storage_path}/plots")
    print("End of the script")