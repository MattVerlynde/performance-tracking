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
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from get_stats import get_stats
from analyse_stats import analyse_stats 

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Patch
from matplotlib.colors import Colormap
import seaborn as sns

import warnings

def plot_frugality_score(perf, conso, legends, storage_path, title):
    w = np.linspace(0,1,100)
    legend_unique = legends.unique()
    df = pd.DataFrame()
    for legend in legend_unique:
        sample_perf = np.mean(perf[legends == legend])
        sample_conso = np.mean(conso[legends == legend])
        slope = 1 / (1 + 1/sample_conso)
        frugality_score = np.ones(w.shape)*sample_perf - w * slope
        df[legend] = frugality_score
        df["slope"] = slope

    fig = px.line(df, x = w, y = legend_unique, title = "Frugality score", hover_data = "slope")
    fig.update_xaxes(title_text="w")
    fig.update_yaxes(title_text="Frugality score")
    fig.update_layout(legend_title_text='Parameters')
    fig.write_image(os.path.join(storage_path, title)+".png")
    fig.write_html(os.path.join(storage_path, title)+".html", include_mathjax='cdn')
    

def plot_correlation_matrix(data, storage_path):

    data_sorted = data[sorted(data.columns.values.tolist())]
    data_corr = data_sorted.drop('Method', axis=1)
    
    fig = px.imshow(np.corrcoef(data_corr.values.T), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, 
                    x = data_corr.columns, y = data_corr.columns,
                    labels=dict(x="Variables", y="Variables", color="Correlation coefficient"))
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        title="Correlation matrix",
        autosize=False,
        width=1000,
        height=1000
    )
    fig.write_html(os.path.join(storage_path, "correlation_matrix.html"), include_mathjax='cdn')
    fig.write_image(os.path.join(storage_path, "correlation_matrix.png"))


    methods = sorted(data['Method'].unique())
    n_images = sorted(data['Number images'].unique())
    fig = make_subplots(rows=len(n_images), cols=len(methods),
                        column_titles = [f"Method {int(x)}" for x in methods],
                        row_titles = [f"{int(x)} images" for x in n_images])

    for i in range(len(n_images)):
        for j in range(len(methods)):
            data_corr = data_sorted.loc[(data['Number images'] == n_images[i]) & (data['Method'] == methods[j])].copy().drop(['Method', 'Number images'],axis=1)
            if not data_corr.empty:

                fig.add_trace(go.Heatmap(
                    z=np.corrcoef(data_corr.values.T), 
                    x=data_corr.columns, 
                    y=data_corr.columns,
                    colorscale="RdBu_r", 
                    zmin=-1, zmax=1, 
                    colorbar=dict(title="Correlation coefficient")
                    ), row=i+1, col=j+1)
                
                fig.update_xaxes(showticklabels=False, row=i+1, col=j+1)
                fig.update_yaxes(showticklabels=False, row=i+1, col=j+1)
    
    fig.for_each_annotation(lambda a:  a.update(y = -0.2) if a.text in n_images else a.update(x = -0.07) if a.text in methods else())
    for i in range(len(n_images)):
        fig.update_yaxes(showticklabels=True, row=i+1, col=1)
    for j in range(len(methods)):
        fig.update_xaxes(showticklabels=True, row=len(n_images), col=j+1)
    
    fig.write_html(os.path.join(storage_path, "correlation_matrices.html"), include_mathjax='cdn')
    fig.write_image(os.path.join(storage_path, "correlation_matrices.png"))

    



def plot_pca(eig, data, data_pca, data_legend, eucl_dist1, eucl_dist2, ccircle, storage_path, title_suffix):
    legend_labels = []
    with plt.style.context(('seaborn-v0_8-whitegrid')):
        fig, axs = plt.subplots(1,2, figsize=(16,9))
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
                # ax.text(ccircle[i][iax],ccircle[i][iax+1], data.columns[i],fontsize=10)
                if iax == 0:  # Add legend labels only once
                    legend_labels.append(data.columns[i])
            # Draw the unit circle, for clarity
            circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
            ax.add_patch(circle)
            ax.set_aspect('equal')
            ax.set_xlabel(f"PCA {iax+1}")
            ax.set_ylabel(f"PCA {iax+2}")
    axs[0].legend(legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(storage_path, f"pca_circle_{title_suffix}.png"))

    # Plot PCA representation in correlation circle

    fig = make_subplots(rows=1, cols=2)
    for iax in range(2):
        eucl_dist = [eucl_dist1, eucl_dist2][iax]
        for i,j in enumerate(eucl_dist):
            fig.add_trace(go.Scatter(x=[0, ccircle[i][0]], y=[0, ccircle[i][iax+1]], name = data.columns[i], mode='lines+markers'), row=1, col=iax+1)
            #fixed color by variable
            fig.for_each_trace(lambda trace: trace.update(line=dict(color='red') if trace.name in ['AUC', 'SSIM', 'Average Precision', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'] else dict(color='blue'),
                marker=dict(color='red') if trace.name in ['AUC', 'SSIM', 'Average Precision', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'] else dict(color='blue')))

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
        autosize=True
    )
    fig.write_html(os.path.join(storage_path, f"pca_circle_{title_suffix}.html"), include_mathjax='cdn')
    
    # Plot data in PCA representation

    print(eig)
    print("-"*27)
    correlation = pd.DataFrame(ccircle)
    correlation.columns = ["Dim1", "Dim2", "Dim3"]
    correlation.index = data.columns
    print(correlation)

    fig = px.scatter_3d(pd.DataFrame.join(pd.DataFrame(data_pca), pd.Series(data_legend, name="Parameters")), x=0, y=1, z=2,
        title=f"Premiers plans factoriels ({np.sum(eig['% variance expliquée'][0:2])})", 
        labels={0: f"Dimension 1 ({eig['% variance expliquée'][0]}%)", 1: f"Dimension 2 ({eig['% variance expliquée'][1]}%)", 2: f"Dimension 3 ({eig['% variance expliquée'][2]}%)"},
        hover_data="Parameters",
        color="Parameters")
    fig.write_html(os.path.join(storage_path, f"pca_{title_suffix}.html"), include_mathjax='cdn')


def plot_parameter_distrib(data, ids, storage_path, performance_metric):
    """Function to plot PCA and T-SNE on consumption and performance data.

    Parameters
    ----------
    output: str
        Path to the output csv file
    
    """
    
    # # Plot performance vs Energy

    # fig, ax = plt.subplots(1, 1, figsize=(16,8))
    # # ax.scatter(data["Energy"], data[performance_metric], s=10, c='white', alpha=0.01,
    # #            label='sample')
    
    # # Create parameters visualisation with matplotlib
    # parameters = pd.DataFrame()
    # for id in ids:
    #     with open(os.path.join(storage_path, f"run_{id}", "info.yaml"), 'r') as f:
    #                     paramYaml = yaml.load(f, Loader=yaml.FullLoader)
    #     parameters = pd.concat([parameters, pd.DataFrame(paramYaml['groups_of_parameters'])], axis=0, ignore_index=True)
    
    # for param_name in parameters.columns:
    #     if param_name in ["--window", "--cores", "--image", "--robust", "--riemann"]:
    #         param_values = parameters[param_name].unique()  
    #         param_name = "Window size" if param_name == "--window" else param_name      
    #         param_name = "Threads" if param_name == "--cores" else param_name
    #         param_name = "Method" if param_name in ["--robust", "--riemann"] else param_name
    #         if param_name == "--image":
    #             param_name = "Number images"
    #             for i in range(len(param_values)):
    #                 if "Scene_1" in param_values[i] or "Scene_2" in param_values[i]:
    #                     param_values[i] = 4.0
    #                 elif "Scene_3" in param_values[i]:
    #                     param_values[i] = 17.0
    #                 else:
    #                     param_values[i] = None
    #         for param_value in param_values:
    #             r = np.round(np.random.rand(),1)
    #             g = np.round(np.random.rand(),1)
    #             b = np.round(np.random.rand(),1)

    #             sample = data[data[param_name] == param_value][["Energy",performance_metric]]

    #             # Estimate mean and covariance matrix
    #             mean_est = np.mean(sample, axis=0)
    #             cov_est = np.cov(sample.T)
    #             # Compute eigenvalues and eigenvectors
    #             eig_val_est, eig_vec_est = np.linalg.eig(cov_est)
    #             # Compute angle of rotation
    #             angle_est = np.arctan2(
    #                     eig_vec_est[1, 0],
    #                     eig_vec_est[0, 0]) * 180 / np.pi

    #             # Plot estimated covariance matrix ellipsoid
    #             ell_est = Ellipse(xy=(mean_est["Energy"], mean_est[performance_metric]),
    #                         width=2 * np.sqrt(eig_val_est[0]),
    #                         height=2 * np.sqrt(eig_val_est[1]),
    #                         angle=angle_est,
    #                         label=f'Covariance for {param_name} = {param_value}')
    #             ell_est.set_facecolor([r,g,b,0.1])
    #             ell_est.set_edgecolor("none")
    #             ax.add_patch(ell_est)

    #             ax.scatter(mean_est["Energy"], mean_est[performance_metric], s=100, color=[r,g,b], marker='x',
    #                     label=f'Mean for {param_name} = {param_value}')

    # ax.legend()
    # ax.set_xlabel(r'$Energy$')
    # ax.set_ylabel(performance_metric)
    # ax.set_title(f"Plotting energy vs {performance_metric}", ha='left', fontsize=12, loc='left')

    # fig.savefig(os.path.join(storage_path, "perf_energy_ellipse.png"), bbox_inches='tight')

    # # Do the same plot but with one ellipse by total configuration
    # for n_images in data["Number images"].unique():
    #     sample = data[data["Number images"] == n_images].copy()
    #     sample = sample[sample["Energy"] != 0]
    #     sample["Legend"] = "Window size: " + sample["Window size"].astype(str) + ", Threads: " + sample["Threads"].astype(str) + ", Method: " + sample["Method"].astype(str)
        
    #     fig, ax = plt.subplots(1, 1, figsize=(16,8))
        
    #     colors = sns.color_palette("colorblind", len(sample["Legend"].unique()))
    #     for i, param_set in enumerate(sample["Legend"].unique()):

    #         sample_ellipse = sample[sample["Legend"] == param_set][["Energy", performance_metric]]
    #         # Estimate mean and covariance matrix
    #         mean_est = np.mean(sample_ellipse, axis=0)
    #         cov_est = np.cov(sample_ellipse.T)
    #         # Compute eigenvalues and eigenvectors
    #         eig_val_est, eig_vec_est = np.linalg.eig(cov_est)
    #         # Compute angle of rotation
    #         angle_est = np.arctan2(
    #                 eig_vec_est[1, 0],
    #                 eig_vec_est[0, 0]) * 180 / np.pi

    #         # Plot estimated covariance matrix ellipsoid
    #         ell_est = Ellipse(xy=(mean_est["Energy"], mean_est[performance_metric]),
    #                     width=2 * np.sqrt(eig_val_est[0]),
    #                     height=2 * np.sqrt(eig_val_est[1]),
    #                     angle=angle_est,
    #                     label=f'Covariance for {param_set}')
    #         ell_est.set_facecolor((colors[i], 0.2))
    #         ell_est.set_edgecolor("none")
    #         ax.add_patch(ell_est)

    #         ax.scatter(mean_est["Energy"], mean_est[performance_metric], s=100, color=colors[i], marker='x',
    #                 label=f'Mean for {param_set}')
    #         ax.scatter(sample_ellipse["Energy"], sample_ellipse[performance_metric], color=(colors[i], 0.2))

    #     ax.legend()
    #     ax.set_xlabel(r'$Energy$')
    #     ax.set_ylabel(performance_metric)
    #     ax.set_title(f"Plotting energy vs {performance_metric} for {int(n_images)} images", ha='left', fontsize=12, loc='left')

    #     fig.savefig(os.path.join(storage_path, f"perf_energy_ellipse_{int(n_images)}images.png"), bbox_inches='tight')

    

    # # Create parameters visualisation with matplotlib
    # fig, ax = plt.subplots(1, 1, figsize=(16,8))
    # colors = []
    # labels = []

    # for param_name in parameters.columns:
    #     if param_name in ["--window", "--cores", "--image", "--robust", "--riemann"]:
    #         param_values = parameters[param_name].unique()  
    #         param_name = "Window size" if param_name == "--window" else param_name      
    #         param_name = "Threads" if param_name == "--cores" else param_name
    #         param_name = "Method" if param_name in ["--robust", "--riemann"] else param_name
    #         if param_name == "--image":
    #             param_name = "Number images"
    #             for i in range(len(param_values)):
    #                 if "Scene_1" in param_values[i]:
    #                     param_values[i] = 2.0
    #                 elif "Scene_2" in param_values[i]:
    #                     param_values[i] = 4.0
    #                 elif "Scene_3" in param_values[i]:
    #                     param_values[i] = 17.0
    #                 else:
    #                     param_values[i] = None
    #         for param_value in param_values:
    #             r = np.round(np.random.rand(),1)
    #             g = np.round(np.random.rand(),1)
    #             b = np.round(np.random.rand(),1)

    #             sample = data[data[param_name] == param_value][["Energy",performance_metric]].copy()

    #             sns.kdeplot(x=sample["Energy"], y=sample[performance_metric], color=(r,g,b), fill=True, alpha = 0.4, warn_singular=False)
    #             colors.append((r,g,b, 0.4))
    #             labels.append(f'{param_name} = {param_value}')

    # for line in ax.get_lines():
    #     line.set_alpha(0)
    # handles = [Patch(facecolor=color, label=label)  for color,label in zip(colors,labels)]
    # plt.legend(handles=handles)
    # ax.set_xlabel(r'$Energy$')
    # ax.set_ylabel(performance_metric)
    # ax.set_title(f"Plotting energy vs {performance_metric}", ha='left', fontsize=12, loc='left')

    # fig.savefig(os.path.join(storage_path, "perf_energy_ellipse_seaborn.png"), bbox_inches='tight')

    # Create parameters visualisation with matplotlib

    for method in data["Method"].unique(): 
        sample_method = data.loc[data["Method"] == method].copy()
        for n_images in sample_method["Number images"].unique(): 
            sample = sample_method.loc[sample_method["Number images"] == n_images].copy()
            sample["Legend"] = "Window size: " + sample["Window size"].astype(str) + ", Threads: " + sample["Threads"].astype(str)

            fig, ax = plt.subplots(1, 1, figsize=(16,8))
            try:
                sns.kdeplot(x = sample["Energy (CodeCarbon)"], y = sample[performance_metric], hue = sample["Legend"], fill = True, alpha = 0.4, palette = "colorblind", warn_singular=False, ax=ax)
            except:
                warnings.warn("Data distribution does not allow KDE representation.")
            sns.scatterplot(sample, x = "Energy (CodeCarbon)", y = performance_metric, hue = "Legend", alpha = 1, palette = "colorblind")
            ax.set_xlabel("Energy (CodeCarbon)")
            ax.set_ylabel(performance_metric)
            ax.set_title(f"Plotting energy vs {performance_metric} for {int(n_images)} images with method {int(method)}", ha='left', fontsize=12, loc='left')
            fig.savefig(os.path.join(storage_path, f"perf_energy_ellipse_seaborn_{int(n_images)}images_method{int(method)}.png"), bbox_inches='tight')
            fig.show()

            fig_plotly = px.scatter(sample, x="Energy (CodeCarbon)", y=performance_metric, color='Legend', hover_data=sample.columns)
            fig_plotly.write_html(os.path.join(storage_path, f"perf_energy_{int(n_images)}images_method{int(method)}.html"), include_mathjax='cdn')

    sample = data.copy()
    sample["Legend"] = "Window size: " + sample["Window size"].astype(str) + ", Threads: " + sample["Threads"].astype(str) + ", Method: " + sample["Method"].astype(str) + ", Number of images: " + sample["Number images"].astype(str)
    fig, ax = plt.subplots(1, 1, figsize=(16,8))
    fig_plotly = px.scatter(sample, x="Energy (CodeCarbon)", y=performance_metric, color='Legend', hover_data=sample.columns)
    fig_plotly.write_html(os.path.join(storage_path, f"perf_energy_all.html"), include_mathjax='cdn')

def plot_stats(storage_path, ids, output_path):
    """Function to plot PCA and T-SNE on consumption and performance data.

    Parameters
    ----------
    output_path: str
        Path to the output csv file
    
    """
    print("-"*14)
    print("Analysing data")
    chunk = pd.read_csv(output_path, header=0, chunksize=1000)
    data = pd.concat(chunk)
    data = data.drop(['Average Precision'], axis=1)

    data_before_pca = data.drop(['Method'], axis=1)

    eig, data_pca, data_tsne, tsne_div, coordvar, ccircle, eucl_dist1, eucl_dist2 = analyse_stats(data_before_pca)
    data_legend = "Window size: " + data["Window size"].astype(str) + ", Threads: " + data["Threads"].astype(str) + ", Method: " + data["Method"].astype(str) + ", Number of images: " + data["Number images"].astype(str) 
    performance_metric = "AUC" if "AUC" in data.columns else "Calinski-Harabasz"
    
    print("-"*27)
    print("Plotting correlation matrix")

    plot_correlation_matrix(data, storage_path)
    
    print("-"*27)
    print("Plotting PCA representation")
    
    plot_pca(eig=eig, data=data_before_pca, data_pca=data_pca, data_legend=data_legend, 
             eucl_dist1=eucl_dist1, eucl_dist2=eucl_dist2, ccircle=ccircle, 
             storage_path=storage_path,
             title_suffix = 'all')
    
    print("-"*27)
    print("Plotting PCA representation for each method and dataset")

    for method in data["Method"].unique():
        sample_method = data.loc[data["Method"] == method].copy()
        for n_images in sample_method["Number images"].unique():
                sample = sample_method.loc[sample_method["Number images"] == n_images].copy()
                data_legend_sample = data_legend.loc[(data["Method"] == method) & (data["Number images"] == n_images)].copy()

                if sample.shape[0] > sample.shape[1]:
                    eig_sample, data_pca_sample, _, _, _, ccircle_sample, eucl_dist1_sample, eucl_dist2_sample = analyse_stats(sample)

                    plot_pca(eig=eig_sample, data=sample, data_pca=data_pca_sample, data_legend=data_legend_sample, 
                            eucl_dist1=eucl_dist1_sample, eucl_dist2=eucl_dist2_sample, ccircle=ccircle_sample, 
                            storage_path=storage_path,
                            title_suffix = f"{n_images}images_method{method}")
                else:
                    print(f"Not enough samples to plot PCA for method {method} with {n_images} images.")

    print("-"*29)
    print("Plotting T-SNE representation")

    fig = px.scatter(pd.DataFrame.join(pd.DataFrame(data_tsne), pd.Series(data_legend, name="Parameters")), color="Parameters", x=0, y=1, 
        title=f"t-SNE visualization (KL divergence: {tsne_div})", 
        labels={0: "Dimension 1", 1: "Dimension 2"})
    
    fig.write_html(os.path.join(storage_path, "tsne.html"), include_mathjax='cdn')    

    print("-"*31)
    print("Plotting parameter distribution")

    plot_parameter_distrib(data, ids, storage_path, performance_metric)

    print("Plotting duration vs energy")

    fig, ax = plt.subplots(1, 1, figsize=(16,8))
    sns.scatterplot(x = data["Duration"], y = data[performance_metric], hue = data_legend, alpha = 1, palette = "colorblind")
    ax.set_xlabel("Duration")
    ax.set_ylabel(performance_metric)
    ax.set_title(f"Plotting duration vs {performance_metric}", ha='left', fontsize=12, loc='left')
    fig.savefig(os.path.join(storage_path, f"duration_energy.png"), bbox_inches='tight')
    fig.show()

    fig, ax = plt.subplots(1, 1, figsize=(16,8))
    fig = px.scatter(x = data["Duration"], y = data[performance_metric], color = data_legend, color_discrete_sequence=px.colors.qualitative.Dark24,
               title = f"Plotting duration vs {performance_metric}",
               labels=dict(x="Duration (s)", y=performance_metric, color="Parameters"))
    fig.write_html(os.path.join(storage_path, f"duration_energy.html"), include_mathjax='cdn')

    print("Plotting energy measurement comparison")

    fig = px.scatter(x = data["Energy (plug)"]/(3.6*1e6), y = data["Energy (CodeCarbon)"]/(3.6*1e6),
               title = "Energy measurement comparison")
    fig.add_trace(go.Scatter(x = [0, np.max(data["Energy (plug)"])/(3.6*1e6)], y =[0, np.max(data["Energy (plug)"])/(3.6*1e6)], mode='lines', line_color="black"))
    fig.update_xaxes(title = "Energy measured by the plug (kWh)")
    fig.update_yaxes(title = "Energy measured by CodeCarbon (kWh)")
    fig.update_layout(showlegend=False) 
    fig.write_image(os.path.join(storage_path, f"energy_energy.png"))
    fig.write_html(os.path.join(storage_path, f"energy_energy.html"), include_mathjax='cdn')
    
    print("Plotting frugality score")

    plot_frugality_score(perf = data[performance_metric], conso = data["Energy (CodeCarbon)"], legends = data_legend, storage_path = storage_path,
                            title = "frugality_score")

    print("-"*24)
    print(" "*10+"DONE"+" "*10)
    print("-"*24)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, default="simulations/results_qanat/conso-change")
    parser.add_argument("--id", type=int, required=True, nargs='+')
    parser.add_argument("--grouped", "-g", type=int, required=True, nargs='+')
    parser.add_argument("--file", "-f", type=bool, default=False)
    args = parser.parse_args()



    output_path = os.path.join(args.storage_path, f"output_all.csv")
    if not args.file:
        output_df = pd.DataFrame()
        
        for group_i, id in enumerate(args.id):
            if not args.grouped[group_i]:
                results = os.path.join(args.storage_path, f"run_{id}", "results.txt")
                times = os.path.join(args.storage_path, f"run_{id}", "times.txt")


                with open(os.path.join(args.storage_path, f"run_{id}", "group_info.yaml"), 'r') as f:
                    paramYaml = yaml.load(f, Loader=yaml.FullLoader)

                window = int(paramYaml['parameters']['--window'])
                cores = int(paramYaml['parameters']['--cores'])
                name_images = paramYaml['parameters']['--image']

                if "Scene_1" in name_images:
                    n_images = 2
                elif"Scene_2" in name_images:
                    n_images = 4
                elif "Scene_3" in name_images:
                    n_images = 17
                else:
                    n_images = None

                output_df_i = get_stats(results, times, os.path.join(args.storage_path, f"run_{id}"), True)
                output_df_i["Window size"] = window*np.ones(len(output_df_i))
                output_df_i["Threads"] = cores*np.ones(len(output_df_i))
                output_df_i["Number images"] = n_images*np.ones(len(output_df_i))

                if "--robust" in paramYaml['parameters'].keys():
                    method = paramYaml['parameters']['--robust']
                    if method == 2:
                        output_df_i[["CPU","Memory","Energy (plug)","Temperature","Reads","Duration","Emissions","Energy (CodeCarbon)"]] = 1e-2*output_df_i[["CPU","Memory","Energy (plug)","Temperature","Reads","Duration","Emissions","Energy (CodeCarbon)"]]
                if "--riemann" in paramYaml['parameters'].keys():
                    method = paramYaml['parameters']['--riemann']
                output_df_i["Method"] = int(method)*np.ones(len(output_df_i))

                output_df = pd.concat([output_df, output_df_i], ignore_index=True)
            else:
                list_group = sorted(os.listdir(os.path.join(args.storage_path, f"run_{id}")))[:-1]
                for group in list_group:
                    results = os.path.join(args.storage_path, f"run_{id}", group, "results.txt")
                    times = os.path.join(args.storage_path, f"run_{id}", group, "times.txt")

                    with open(os.path.join(args.storage_path, f"run_{id}", group, "group_info.yaml"), 'r') as f:
                        paramYaml = yaml.load(f, Loader=yaml.FullLoader)

                    window = int(paramYaml['parameters']['--window'])
                    cores = int(paramYaml['parameters']['--cores'])
                    name_images = paramYaml['parameters']['--image']
                    if "Scene_1" in name_images:
                        n_images = 2
                    elif "Scene_2" in name_images:
                        n_images = 4
                    elif "Scene_3" in name_images:
                        n_images = 17
                    else:
                        n_images = None

                    output_df_i = get_stats(results, times, os.path.join(args.storage_path, f"run_{id}", group), True)
                    output_df_i["Window size"] = window*np.ones(len(output_df_i))
                    output_df_i["Threads"] = cores*np.ones(len(output_df_i))
                    output_df_i["Number images"] = n_images*np.ones(len(output_df_i))

                    if "--robust" in paramYaml['parameters'].keys():
                        method = paramYaml['parameters']['--robust']
                        if method == 2:
                            output_df_i[["CPU","Memory","Energy (plug)","Temperature","Reads","Duration","Emissions","Energy (CodeCarbon)"]] = 1e-2*output_df_i[["CPU","Memory","Energy (plug)","Temperature","Reads","Duration","Emissions","Energy (CodeCarbon)"]]
                    if "--riemann" in paramYaml['parameters'].keys():
                        method = paramYaml['parameters']['--riemann']
                    output_df_i["Method"] = int(method)*np.ones(len(output_df_i))

                    output_df = pd.concat([output_df, output_df_i], ignore_index=True)

        output_df.to_csv(output_path, index=False)
    
    plot_stats(args.storage_path, args.id, output_path)
