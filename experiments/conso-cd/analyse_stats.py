# -*- coding: utf-8 -*-
#
# This script is a qanat action executable analysing model performances from a run of the 'conso' experiment.
# Usage: qanat experiment action conso analyse-stats [RUN_ID] -r [GROUND_TRUTH_FILE_PATH] [-q [True/False]]
#
# Author: Matthieu Verlynde
# Email: matthieu.verlynde@univ-smb.fr
# Date: 25 Jun 2024
# Version: 1.0.0

import yaml
import argparse
import os
import re
import numpy as np
import pandas as pd
import subprocess
import plotly.express as px
from scipy import integrate
from skimage.metrics import structural_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from get_stats import get_stats

import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
from sklearn.metrics.pairwise import euclidean_distances

def analyse_stats(output):
    """Function to compute PCA on consumption and performance data.

    Parameters
    ----------
    output: str
        Path to the output csv file
    
    Returns
    -------
    eig: pandas dataframe
        Dataframe containing the eigenvalues
    data_pca: numpy array
        Data transformed by the PCA
    """
    data = pd.read_csv(output, header=0)
    n = data.shape[0] # nb individus
    p = data.shape[1] # nb variables

    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data))
    data_scaled.columns = data.columns

    pca = PCA()
    pca.fit(data_scaled)

    eig = pd.DataFrame(
        {
            "Dimension" : ["Dim" + str(x + 1) for x in range(p)], 
            "Variance expliquée" : pca.explained_variance_,
            "% variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
            "% cum. var. expliquée" : np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
        }
    )
    data_pca = pca.transform(data_scaled)

    eigval = (n-1) / n * pca.explained_variance_ # valeurs propres
    sqrt_eigval = np.sqrt(eigval) # racine carrée des valeurs propres
    corvar = np.zeros((p,p)) # matrice vide pour avoir les coordonnées
    for k in range(p):
        corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]
    # on modifie pour avoir un dataframe
    coordvar = pd.DataFrame({'id': data.columns, 'COR_1': corvar[:,0], 'COR_2': corvar[:,1], 'COR_3': corvar[:,2]})

    ccircle = []
    eucl_dist = []
    for i in range(len(data_scaled.T)):
        if np.std(data_scaled.T.iloc[i]) == 0:
            corr1 = 0
            corr2 = 0
        else:
            corr1 = np.corrcoef(data_scaled.T.iloc[i],data_pca[:,0])[0,1]
            corr2 = np.corrcoef(data_scaled.T.iloc[i],data_pca[:,1])[0,1]
        ccircle.append((corr1, corr2))
        eucl_dist.append(np.sqrt(corr1**2 + corr2**2))

    return eig, data_scaled, data_pca, coordvar, ccircle, eucl_dist


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--query", "-q", type=bool, default=True)
    args = parser.parse_args()

    results = os.path.join(args.storage_path, "results.txt")
    times = os.path.join(args.storage_path, "stdout.txt")
    output = os.path.join(args.storage_path, "output.csv")

    get_stats(results, times, args.storage_path, args.query).to_csv(output, index=False)
    
    eig,  data, data_pca, coordvar, ccircle, eucl_dist = analyse_stats(output)

    fig = px.scatter(data_pca, x=0, y=1, 
        title=f"Premier plan factoriel ({np.sum(eig['% variance expliquée'][0:1])})", 
        labels={"0": f"Dimension 1 ({eig['% variance expliquée'][0]}%)", "1": f"Dimension 2 ({eig['% variance expliquée'][1]}%)"})
    
    fig.write_html(os.path.join(args.storage_path, "pca.html"), include_mathjax='cdn')