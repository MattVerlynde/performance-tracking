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
from sklearn import metrics

from conso_clustering.utils_clustering import load_data

import plotly.express as px

import argparse


###############################################################################
# Plot data and results
# ---------

def get_perf_clustering(storage_path):

    with open(os.path.join(storage_path, "group_info.yaml"), 'r') as f:
        paramYaml = yaml.load(f, Loader=yaml.FullLoader)

    data_path = paramYaml['parameters']['--image']
    n_clusters = int(paramYaml['parameters']['--n_clusters'])
    window_size = int(paramYaml['parameters']['--window'])

    data, data_visualization, X_image, Y_image, X_res, Y_res = load_data(data_path, n_clusters, window_size, small_dataset = True)
    height, width, p = data.shape
    w = window_size//2
    data = np.real(data[w:height-w, w:width-w, :].reshape((-1, p)))

    list_cluster = sorted(os.listdir(os.path.join(storage_path, "output")))
    for file_cluster in list_cluster:
        result = np.load(os.path.join(storage_path, "output", file_cluster))
    perf = {}
    perf["Silhouette"] = metrics.silhouette_score(data, result.flatten())
    perf["Calinski-Harabasz"] = metrics.calinski_harabasz_score(data, result.flatten())
    perf["Davies-Bouldin"] = metrics.davies_bouldin_score(data, result.flatten())

    return perf

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
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()

    perf = get_perf_clustering(args.storage_path)
        
    print("Done")
    print(f"Score is {perf}")
    print("End of the script")

