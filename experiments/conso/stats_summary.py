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
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from scipy import integrate
from skimage.metrics import structural_similarity

from get_stats import get_stats
from analyse_stats import analyse_stats 

import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, default="simulations/results_qanat/conso-change")
    parser.add_argument("--id", type=int, required=True, nargs='+')
    parser.add_argument("--grouped", "-g", type=bool, default=False)
    parser.add_argument("--query", "-q", type=bool, default=True)
    args = parser.parse_args()

    output_df = pd.DataFrame()

    if not args.grouped:
        for id in args.id:
            results = os.path.join(args.storage_path, f"run_{id}", "results.txt")
            times = os.path.join(args.storage_path, f"run_{id}", "times.txt")


            with open(os.path.join(args.storage_path, f"run_{id}", "group_info.yaml"), 'r') as f:
                paramYaml = yaml.load(f, Loader=yaml.FullLoader)

            window = int(paramYaml['parameters']['--window'])
            cores = int(paramYaml['parameters']['--cores'])
            output_df_i = get_stats(results, times, os.path.join(args.storage_path, f"run_{id}"), args.query)
            output_df_i["Window size"] = window*np.ones(len(output_df_i))
            output_df_i["Threads"] = cores*np.ones(len(output_df_i))

            # T = int(paramYaml['parameters']['--image'][-7:-5])
            # output_df_i["T"] = T*np.ones(len(output_df_i))

            output_df = pd.concat([output_df, output_df_i], ignore_index=True)
    else:
        id = args.id[0]
        list_group = sorted(os.listdir(os.path.join(args.storage_path, f"run_{id}")))[:-1]
        for group in list_group:
            results = os.path.join(args.storage_path, f"run_{id}", group, "results.txt")
            times = os.path.join(args.storage_path, f"run_{id}", group, "times.txt")

            with open(os.path.join(args.storage_path, f"run_{id}", group, "group_info.yaml"), 'r') as f:
                paramYaml = yaml.load(f, Loader=yaml.FullLoader)

            window = int(paramYaml['parameters']['--window'])
            cores = int(paramYaml['parameters']['--cores'])
            output_df_i = get_stats(results, times, os.path.join(args.storage_path, f"run_{id}", group), args.query)
            output_df_i["Window size"] = window*np.ones(len(output_df_i))
            output_df_i["Threads"] = cores*np.ones(len(output_df_i))

            # T = int(paramYaml['parameters']['--image'][-7:-5])
            # output_df_i["T"] = T*np.ones(len(output_df_i))

            output_df = pd.concat([output_df, output_df_i], ignore_index=True)

    
    output = os.path.join(args.storage_path, f"output_all.csv")
    output_df.to_csv(output, index=False)

    eig, data, data_pca, data_tsne, tsne_div, coordvar, ccircle, eucl_dist1, eucl_dist2 = analyse_stats(output)

    print(coordvar)

    
    with plt.style.context(('seaborn-v0_8-whitegrid')):
        fig, axs = plt.subplots(1,2,figsize=(16, 8))
        for iax,ax in enumerate(axs):
            eucl_dist = [eucl_dist1, eucl_dist2][iax]
            for i,j in enumerate(eucl_dist):
                # arrow_col = plt.cm.YlOrRd((eucl_dist[i] - np.array(eucl_dist).min())/\
                #         (np.array(eucl_dist).max() - np.array(eucl_dist).min()) )
                arrow_col = plt.cm.tab20(i)
                ax.arrow(0,0, # Arrows start at the origin
                        ccircle[i][iax],  #0 for PC1
                        ccircle[i][iax+1],  #1 for PC2
                        lw = 2, # line width
                        length_includes_head=True, 
                        color = arrow_col,
                        fc = arrow_col,
                        head_width=0.05,
                        head_length=0.05)
                ax.text(ccircle[i][iax],ccircle[i][iax+1], output_df.columns[i],fontsize=10)
            # Draw the unit circle, for clarity
            circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
            ax.add_patch(circle)
            ax.set_aspect('equal')
            ax.set_xlabel(f"PCA {iax+1}")
            ax.set_ylabel(f"PCA {iax+2}")
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.storage_path, "pca_circle.png"))

    # Plot PCA representation in correlation circle

    fig = make_subplots(rows=1, cols=2)
    for iax in range(2):
        eucl_dist = [eucl_dist1, eucl_dist2][iax]
        for i,j in enumerate(eucl_dist):
            fig.add_trace(go.Scatter(x=[0, ccircle[i][0]], y=[0, ccircle[i][iax+1]], name = output_df.columns[i], mode='lines+markers'), row=1, col=iax+1)
            #fixed color by variable
            fig.for_each_trace(lambda trace: trace.update(line=dict(color='red') if trace.name == 'AUC' or trace.name == 'SSIM' else dict(color='blue'),
                marker=dict(color='red') if trace.name == 'AUC' or trace.name == 'SSIM' else dict(color='blue')))

        # Add circles
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=-1, y0=-1, x1=1, y1=1,
            line_color="black",
            row=1, col=iax+1
        )

        fig.update_xaxes(range=[-1,1], title_text="PCA 1", row=1, col=iax+1)
        fig.update_yaxes(range=[-1,1], title_text=f"PCA {iax+2}", row=1, col=iax+1)

    fig.update_layout(
        title="Circle of correlations",
        showlegend=False,
        autosize=False,
        width=1400,
        height=700
    )

    fig.write_html(os.path.join(args.storage_path, "pca_circle.html"), include_mathjax='cdn')

    # Plot data in PCA representation

    print(eig)
    fig = px.scatter(pd.DataFrame.join(pd.DataFrame(data_pca), data), x=0, y=1, 
        title=f"Premier plan factoriel ({np.sum(eig['% variance expliquée'][0:1])})", 
        labels={0: f"Dimension 1 ({eig['% variance expliquée'][0]}%)", 1: f"Dimension 2 ({eig['% variance expliquée'][1]}%)"},
        hover_data=data.columns)
    
    fig.write_html(os.path.join(args.storage_path, "pca.html"), include_mathjax='cdn')

    # Plot data in t-SNE representation

    print(data_tsne.shape)
    fig = px.scatter(pd.DataFrame.join(pd.DataFrame(data_tsne), data), color="AUC", x=0, y=1, 
        title=f"t-SNE visualization (KL divergence: {tsne_div})", 
        labels={0: "Dimension 1", 1: "Dimension 2"},
        hover_data=data.columns)
    
    fig.write_html(os.path.join(args.storage_path, "tsne.html"), include_mathjax='cdn')

    # Plot PCA representation in correlation circle

    fig = px.scatter(data.loc[data["Energy"]!=0], x="AUC", y="Energy", 
        title=f"Energy consumption vs AUC",
        hover_data=data.columns)
    
    fig.write_html(os.path.join(args.storage_path, "auc_energy.html"), include_mathjax='cdn')

