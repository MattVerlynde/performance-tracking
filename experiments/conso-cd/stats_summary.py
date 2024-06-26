# -*- coding: utf-8 -*-
#
# This script is a python executable computing a statistical analysis of model performances and 
# consumption data from several runs of the 'conso' experiment,.
# Usage: python stats_summary.py --id [RUN_ID]
#
# Author: Matthieu Verlynde
# Email: matthieu.verlynde@univ-smb.fr
# Date: 26 Jun 2024
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

from get_stats import get_stats
from analyse_stats import analyse_stats 

import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, default="simulations/results_qanat/conso")
    parser.add_argument("--id", type=int, required=True, nargs='+')
    parser.add_argument("--query", "-q", type=bool, default=True)
    args = parser.parse_args()

    output_df = pd.DataFrame()
    for id in args.id:
        results = os.path.join(args.storage_path, f"run_{id}", "results.txt")
        times = os.path.join(args.storage_path, f"run_{id}", "times.txt")


        with open(os.path.join(args.storage_path, f"run_{id}", "group_info.yaml"), 'r') as f:
            paramYaml = yaml.load(f, Loader=yaml.FullLoader)

        window = int(paramYaml['parameters']['--window'])
        cores = int(paramYaml['parameters']['--cores'])
        T = int(paramYaml['parameters']['--image'][-7:-5])

        output_df_i = get_stats(results, times, os.path.join(args.storage_path, f"run_{id}"), args.query)
        output_df_i["Window size"] = window*np.ones(len(output_df_i))
        output_df_i["Threads"] = cores*np.ones(len(output_df_i))
        output_df_i["T"] = T*np.ones(len(output_df_i))
        output_df = pd.concat([output_df, output_df_i], ignore_index=True)
    
    output = os.path.join(args.storage_path, f"output_all.csv")
    output_df.to_csv(output, index=False)

    eig, data, data_pca, coordvar, ccircle, eucl_dist = analyse_stats(output)

    print(coordvar)

    
    with plt.style.context(('seaborn-v0_8-whitegrid')):
        fig, axs = plt.subplots(figsize=(8, 8))
        for i,j in enumerate(eucl_dist):
            # arrow_col = plt.cm.YlOrRd((eucl_dist[i] - np.array(eucl_dist).min())/\
            #         (np.array(eucl_dist).max() - np.array(eucl_dist).min()) )
            arrow_col = plt.cm.tab20(i)
            axs.arrow(0,0, # Arrows start at the origin
                    ccircle[i][0],  #0 for PC1
                    ccircle[i][1],  #1 for PC2
                    lw = 2, # line width
                    length_includes_head=True, 
                    color = arrow_col,
                    fc = arrow_col,
                    head_width=0.05,
                    head_length=0.05)
            axs.text(ccircle[i][0],ccircle[i][1], output_df.columns[i],fontsize=10)
        # Draw the unit circle, for clarity
        circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
        axs.add_patch(circle)
        axs.set_aspect('equal')
        axs.set_xlabel("PCA 1")
        axs.set_ylabel("PCA 2")
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.storage_path, "pca_circle.png"))

    print(eig)
    data_tot = pd.DataFrame.join(pd.DataFrame(data_pca), data)

    fig = px.scatter(data_tot, x=0, y=1, 
        title=f"Premier plan factoriel ({np.sum(eig['% variance expliquée'][0:1])})", 
        labels={0: f"Dimension 1 ({eig['% variance expliquée'][0]}%)", 1: f"Dimension 2 ({eig['% variance expliquée'][1]}%)"},
        hover_data=data_tot.columns[11:])
    
    fig.write_html(os.path.join(args.storage_path, "pca.html"), include_mathjax='cdn')
    fig.write_image(os.path.join(args.storage_path, "pca.png"))


