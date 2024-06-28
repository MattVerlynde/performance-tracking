"""
====================================================================
Clustering on SAR images with Riemannian geometry
====================================================================

This example compares clustering pipelines based on covariance matrices for
synthetic-aperture radar (SAR) image clustering [1]_ [2]_.
"""
# Author: Ammar Mian

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from pyriemann.estimation import Covariances
from pyriemann.clustering import Kmeans
from helpers.processing_helpers import SlidingWindowVectorize

import argparse

###############################################################################
# Load data
# ---------
def load_data(data_path, n_clusters, window_size, small_dataset):

    print(f"Loading data :")
    data = np.load(data_path)
    data_visualization = data.copy()  # To avoid aliasing when showing data
    n_components = data.shape[2]
    resolution_x = 1.6  # m, obtained from UAVSAR documentation
    resolution_y = 0.6  # m, obtained from UAVSAR documentation

    # For visualization of image
    x_values = np.arange(0, data.shape[1]) * resolution_x
    y_values = np.arange(0, data.shape[0]) * resolution_y
    X_image, Y_image = np.meshgrid(x_values, y_values)

    if small_dataset:
        reduce_factor_y = 13
        reduce_factor_x = 7
        data = data[::reduce_factor_y, ::reduce_factor_x]
        max_iter = 10
        resolution_x = reduce_factor_x*resolution_x
        resolution_y = reduce_factor_y*resolution_y
    height, width, n_features = data.shape

    # For visualization of results
    x_values = np.arange(window_size//2, width-window_size//2) * resolution_x
    y_values = np.arange(window_size//2, height-window_size//2) * resolution_y
    X_res, Y_res = np.meshgrid(x_values, y_values)

    print("Reading done.")
    return data, data_visualization, X_image, Y_image, X_res, Y_res

###############################################################################
# Pipelines definition
# --------------------

def make_pipelines(window_size, estimator, n_clusters, n_jobs,  max_iter):

    # Logdet pipeline from [1]
    pipeline_logdet = Pipeline([
        ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
        ("covariances", Covariances(estimator=estimator)),
        ("kmeans", Kmeans(
            n_clusters=n_clusters,
            n_jobs=n_jobs,
            max_iter=max_iter,
            metric="logdet",
        ))
    ], verbose=True)

    # Riemannian pipeline from [2]
    pipeline_riemann = Pipeline([
        ("sliding_window", SlidingWindowVectorize(window_size=window_size)),
        ("covariances", Covariances(estimator=estimator)),
        ("kmeans", Kmeans(
            n_clusters=n_clusters,
            n_jobs=n_jobs,
            max_iter=max_iter,
            metric="riemann",
        ))
    ], verbose=True)

    pipelines = [pipeline_logdet, pipeline_riemann]
    pipelines_names = [f"{estimator} and logdet", f"{estimator} and Riemann"]

    return pipelines, pipelines_names

###############################################################################
# Perform clustering
# ------------------

def compute_clustering(pipelines_names, pipelines, data):

    print(f"\nStarting clustering with pipelines: {pipelines_names}")
    results = {}
    for pipeline_name, pipeline in zip(pipelines_names, pipelines):
        print("-"*60)
        print(f"Pipeline: {pipeline_name}")
        pipeline.fit(data)
        preds = pipeline.named_steps["kmeans"].labels_
        covmats = pipeline.named_steps["covariances"].transform(data)
        results[pipeline_name] = \
            pipeline.named_steps["sliding_window"].inverse_predict(preds)
        print("-"*60)
    print("Done")
    return results

def clustering(data_path, n_clusters, window, small_dataset, estimator, n_jobs, max_iter, storage_path, number_run):

    data, data_visualization, X_image, Y_image, X_res, Y_res = load_data(data_path, n_clusters, window, small_dataset)

    pipelines, pipelines_names = make_pipelines(window, estimator, n_clusters, n_jobs,  max_iter)

    results = compute_clustering(pipelines_names, pipelines, data)

    for pipeline_name, labels_pred in results.items():
        np.save(os.path.join(storage_path, "output", f"{pipeline_name}_clustering_results_{number_run}.npy"), labels_pred)

    return results


###############################################################################
# Plot data and results
# ---------

def plot_clustering(data_visualization, X_image, Y_image, X_ref, Y_ref, results, storage_path):

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
    figure.savefig(os.path.join(storage_path,"clustering_data.png"))


    for pipeline_name, labels_pred in results.items():
        figure, ax = plt.subplots(figsize=(5, 5))
        plt.pcolormesh(X_res, Y_res, labels_pred, cmap="tab20b")
        plt.xlim(X_image.min(), X_image.max())
        plt.ylim(Y_image.min(), Y_image.max())
        plt.title(f"Clustering with {pipeline_name}")
        plt.colorbar()
        ax.invert_yaxis()
        plt.xlabel("Range (m)")
        plt.ylabel("Azimuth (m)")
        plt.tight_layout()
        figure.savefig(os.path.join(storage_path,pipeline_name+"_clustering_results.png"))
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
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--data_path", type=str, default="data/Scene_1/Scene_1_0.npy")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--small_dataset", type=bool, default=True)
    parser.add_argument("--estimator", type=str, default="scm")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--number_run", "-n", type=str, default="")
    args = parser.parse_args()

    clustering(args.data_path, args.n_clusters, args.window, args.small_dataset, args.estimator, args.n_jobs, args.max_iter, args.storage_path, args.number_run)

    # plot_clustering(data_visualization, X_image, Y_image, X_res, Y_res, results, args.storage_path)
    
    print("Done")
    print(f"Results saved in {args.storage_path}")
    print("End of the script")