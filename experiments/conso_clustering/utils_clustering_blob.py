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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph

import argparse

###############################################################################
# Load data
# ---------
def load_data(n_clusters: int, data_seed: int):
    n_samples = 1000
    n_features = 5
    cluster_std = 5.0

    X, y = make_blobs(n_samples=n_samples,
                       n_features=n_features, 
                       centers=n_clusters,
                       cluster_std=cluster_std,
                       random_state=data_seed)

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=3, include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    return X, y, connectivity

###############################################################################
# Pipelines definition
# --------------------

def make_pipelines(n_clusters: int, random_seed: int,  connectivity: np.ndarray):

    pipelines = {
    # KMeans pipeline from [1]
    "k-means" : KMeans(
            n_clusters=n_clusters,
            init="random",
            random_state=random_seed,
        ),
    "k-means++" : KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=random_seed,
        ),
    "DBSCAN" : DBSCAN(
            eps=0.79,
        ),
    "HDBSCAN" : HDBSCAN(
            min_samples=3,
            min_cluster_size=15,
            allow_single_cluster=False,
        ),
    "OPTICS" : OPTICS(
            min_samples=30,
            xi=0.001,
            min_cluster_size=0.001,
        ),
    "AgglomerativeClustering" : AgglomerativeClustering(
            linkage="average",
            metric="cityblock",
            n_clusters=n_clusters,
            connectivity=connectivity,
        ),
    "Ward" : AgglomerativeClustering(
            linkage="ward",
            n_clusters=n_clusters,
            connectivity=connectivity,
        ),
    "GMM" : GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_seed,
        )
    }

    return pipelines

###############################################################################
# Perform clustering
# ------------------

def compute_clustering(pipeline, X, pipeline_name, repeat):

    print(f"\nStarting clustering with pipelines: {pipeline_name}")
    print("-"*60)
    print(f"Pipeline: {pipeline_name}")
    for i in range(max(repeat,0)):
        pipeline.fit(X)
        if hasattr(pipeline, "labels_"):
            results = pipeline.labels_.astype(int)
        else:
            results = pipeline.predict(X)
    print("-"*60)
    print("Done")
    return results

def clustering(n_clusters: int, model: str, data_seed: int, random_seed: int, storage_path: str, number_run: str, repeat: int):

    X, y, connectivity = load_data(n_clusters=n_clusters, data_seed=data_seed)

    pipeline = make_pipelines(n_clusters, random_seed, connectivity)[model]

    results = compute_clustering(X=X, pipeline=pipeline, pipeline_name=model, repeat=repeat)

    os.makedirs(os.path.join(storage_path, "output"), exist_ok=True)
    np.save(os.path.join(storage_path, "output", f"{number_run}.npy"), results)
    np.save(os.path.join(storage_path, "output", f"{number_run}_truth.npy"), y)
    
    return results

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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--number_run", "-n", type=str, default="")
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)

    args = parser.parse_args()

    from codecarbon import OfflineEmissionsTracker
    
    DIR_CARBON = os.path.join(args.storage_path,"codecarbon")
    os.makedirs(DIR_CARBON, exist_ok=True)
    tracker = OfflineEmissionsTracker(country_iso_code="FRA", output_dir=DIR_CARBON, output_file=f"emissions{args.number_run}.csv")
    tracker.start()

    clustering(n_clusters=args.n_clusters, 
               storage_path=args.storage_path, 
               number_run=args.number_run, 
               data_seed=args.data_seed, 
               random_seed=args.random_seed, 
               model=args.model,
               repeat=args.repeat)

    tracker.stop()
    
    print("Done")
    print(f"Results saved in {args.storage_path}")
    print("End of the script")