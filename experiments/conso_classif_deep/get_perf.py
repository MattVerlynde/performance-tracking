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
import pandas as pd
from sklearn import metrics

import plotly.express as px

import argparse


###############################################################################
# Plot data and results
# ---------

def get_perf_classif_deep(storage_path):

    with open(os.path.join(storage_path, "group_info.yaml"), 'r') as f:
        paramYaml = yaml.load(f, Loader=yaml.FullLoader)

    batch_size = int(paramYaml['parameters']['--batch'])
    n_epochs = int(paramYaml['parameters']['--epochs'])
    lr = float(paramYaml['parameters']['--lr'])
    model = paramYaml['parameters']['--model']
    rgb = int(paramYaml['parameters']['--rgb'])
    fine_tune = int(paramYaml['parameters']['--finetune'])
<<<<<<< HEAD
=======

    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Learning rate: {lr}")
    print(f"RGB: {rgb}")
    print(f"Fine-tune: {fine_tune}")

>>>>>>> a2c0c27f3526725ce1370733e5be04c945819ebc

    if model == "ShortCNN":
        if rgb:
            model = 'S-CNN-RGB'
        else:
            model = 'S-CNN-All'
    elif model == "InceptionV3":
        if fine_tune == 2:
            model = 'InceptionV3-FromScratch'
        elif fine_tune == 1:
            model = 'InceptionV3-FineTune'
        else:
            model = 'InceptionV3-Transfer'
    else:
        model = 'Unknown'

    result_classif = pd.read_csv(os.path.join(storage_path, "output", "scores.csv"))

    perf = result_classif
    
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
    args = parser.parse_args()

    perf = get_perf_classif_deep(args.storage_path)
        
    print("Done")
    print(f"Score is {perf}")
    print("End of the script")

