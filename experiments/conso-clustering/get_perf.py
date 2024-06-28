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

from utils_clustering import load_data

import argparse


###############################################################################
# Plot data and results
# ---------

def get_scores(data_visualization, result):

    return metrics.silhouette_score(data_visualization, result, metric='euclidean')

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
    for file in os.listdir(os.path.join(args.storage_path, "output")):
        result = np.load(os.path.join(args.storage_path, "output", file))

        score = get_scores(data_visualization, result)
    
    print("Done")
    print(f"Score is {score}")
    print("End of the script")