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
from pyriemann.utils.mean import mean_riemann

import argparse


###############################################################################
# Plot data and results
# ---------

def get_scores(storage_path):

    with open(os.path.join(storage_path, "group_info.yaml"), 'r') as f:
        paramYaml = yaml.load(f, Loader=yaml.FullLoader)

    window = int(paramYaml['parameters']['--window'])

    list_covar = sorted(os.listdir(os.path.join(storage_path, "covar")))
    list_cluster = sorted(os.listdir(os.path.join(storage_path, "output")))
    for file_covar, file_cluster in zip(list_covar, list_cluster):
        covmat = np.load(os.path.join(storage_path, "covar", file_covar))
        result = np.load(os.path.join(storage_path, "output", file_cluster))

        covmat = (covmat[(window-1)//2:-(window-1)//2, (window-1)//2:-(window-1)//2, :])
    
    print(result.shape)
    print(covmat.shape)
    scores = {}
    scores["silhouette"] = metrics.silhouette_score(covmat, result)

    return scores

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

    score = get_scores(args.storage_path)
    
    print("Done")
    print(f"Score is {score}")
    print("End of the script")