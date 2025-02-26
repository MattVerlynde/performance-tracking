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

from sklearn.datasets import make_blobs

import plotly.express as px

import argparse


###############################################################################
# Plot data and results
# ---------

def get_perf_clustering_blob(storage_path):

    with open(os.path.join(storage_path, "group_info.yaml"), 'r') as f:
        paramYaml = yaml.load(f, Loader=yaml.FullLoader)

    data_seed = int(paramYaml['parameters']['--data_seed'])
    n_clusters = int(paramYaml['parameters']['--n_clusters'])
    n_samples = 1000
    n_features = 5
    cluster_std = 5.0

    data = make_blobs(n_samples=n_samples, 
                      centers=n_clusters, 
                      n_features=n_features, 
                      cluster_std=cluster_std, 
                      random_state=data_seed)[0]
    
    list_cluster = sorted(os.listdir(os.path.join(storage_path, "output")))
    
    for i in range(int(len(list_cluster)/2)):
        result = np.load(os.path.join(storage_path, "output", f"{i+1}.npy"))
        truth = np.load(os.path.join(storage_path, "output", f"{i+1}_truth.npy"))


    perf = {}
    perf["Silhouette"] = metrics.silhouette_score(data, result.flatten()) if np.unique(result).shape[0] > 1 else None
    perf["Calinski-Harabasz"] = metrics.calinski_harabasz_score(data, result.flatten()) if np.unique(result).shape[0] > 1 else None
    perf["Davies-Bouldin"] = metrics.davies_bouldin_score(data, result.flatten()) if np.unique(result).shape[0] > 1 else None
    perf["Adjusted Rand Index"] = metrics.adjusted_rand_score(truth.flatten(), result.flatten())


    return perf

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()

    perf = get_perf_clustering(args.storage_path)
        
    print("Done")
    print(f"Score is {perf}")
    print("End of the script")

