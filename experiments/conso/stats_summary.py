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
# import tikzplotlib

from get_stats import get_stats
from analyse_stats import analyse_stats 

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Patch
from matplotlib.colors import Colormap
import seaborn as sns

# Set default font size
plt.rcParams.update({'font.size': 18})

from sklearn.linear_model import LinearRegression

import warnings

def plot_frugality_score(perf, conso, legends, storage_path, title, factor=1, norm_perf=1):
    """Function to plot data distribution along the performance and energy consumptions axes.

    Parameters
    ----------
    perf: pandas Series
        Series containing the performance data
    conso: pandas Series
        Series containing the consumption data
    legends: pandas Series
        Series containing the labels with the settings of all parameters for each point
    storage_path: str
        Path to the folder to save the different plots
    title: str
        Title for the plot 
    """
    w = np.linspace(0,1,100)
    legend_unique = legends.unique()
    df = pd.DataFrame()
    slopes = []
    max_frugality = []
    min_frugality = []
    frugality_ws = []
    frugality_hm = []
    epsilon_ws = 0.5
    epsilon_hm = 1
    samples_perf = []
    samples_conso = []

    

    for legend in legend_unique:
        sample_perf = np.mean(perf[legends == legend])
        sample_conso = np.mean(conso[legends == legend])*factor
        samples_perf.append(sample_perf)
        samples_conso.append(sample_conso)
        sample_perf_norm = sample_perf/norm_perf
        sample_conso_norm = sample_conso/(np.max(conso)*factor)
        slope = 1 / (1 + 1/sample_conso)
        frugality_score = np.ones(w.shape)*sample_perf - w * slope
        df[legend] = frugality_score
        slopes.append(slope)
        max_frugality.append(frugality_score[0])
        min_frugality.append(frugality_score[-1])
        frug_ws = epsilon_ws * sample_perf_norm + (1 - epsilon_ws) * (1 - sample_conso_norm)
        frug_hm = (1 + epsilon_hm**2) * (sample_perf_norm * (1 - sample_conso_norm))/(sample_perf_norm + epsilon_hm**2*(1 - sample_conso_norm))
        frugality_ws.append(frug_ws)
        frugality_hm.append(frug_hm)

    fig = px.line(df, x = w, y = legend_unique)
    fig.update_xaxes(title_text="w")
    fig.update_yaxes(title_text="Frugality score")
    fig.update_layout(legend_title_text='Parameters')
    fig.write_html(os.path.join(storage_path, title+ str(factor))+".html", include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')

    plt.figure(figsize=(16,8))
    pal = sns.color_palette('colorblind', legend_unique.shape[0])
    pal = pal.as_hex()
    
    for i,legend in enumerate(legend_unique):
        print(f"Legend: {legend}")
        print(df[legend].shape)
        plt.plot(w, df[legend], color=pal[i], label=legend, linewidth=2)
    print("***************done")
    plt.xlabel("w")
    plt.ylabel("Frugality score")
    # change colors to palette colorblind

    plt.legend(legend_unique)
    # tikzplotlib.save(os.path.join(storage_path, title+ str(factor))+".tex")
    plt.savefig(os.path.join(storage_path, title+ str(factor))+".png")
    plt.show()

    #give metrics on the slopes
    frugality = pd.DataFrame()
    frugality["Parameters"] = legend_unique
    frugality["Mean performance"] = samples_perf
    frugality["Mean consumption"] = samples_conso
    frugality["alpha x beta"] = sample_conso_norm * sample_perf_norm
    frugality["alpha"] = sample_perf_norm
    frugality["beta"] = sample_conso_norm
    frugality["Frugality score max perf"] = max_frugality
    frugality["Frugality score max eco"] = min_frugality
    frugality["Slope frugality score"] = slopes
    frugality["Frugality score (weighted sum)"] = frugality_ws
    frugality["Frugality score (harmonic mean)"] = frugality_hm
    frugality.to_csv(os.path.join(storage_path, title+ str(factor)+".csv"), index=False, float_format='%.3f')

    frugality = pd.DataFrame()
    frugality["Frugality score (harmonic mean)"] = frugality_hm
    frugality.to_csv(os.path.join(storage_path, title+ str(factor)+"_harmonic.csv"), index=False)



    
    

def plot_correlation_matrix(data, storage_path):
    """Function to plot data distribution along the performance and energy consumptions axes.

    Parameters
    ----------
    data: pandas DataFrame
        Dataframe containing the recordings
    storage_path: str
        Path to the folder to save the different plots*
    """
    data_sorted = data[sorted(data.columns.values.tolist())]
    data_corr = data_sorted.drop('Method', axis=1)
    
    fig = px.imshow(np.corrcoef(data_corr.values.T), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, 
                    x = data_corr.columns, y = data_corr.columns,
                    labels=dict(x="Variables", y="Variables", color="Correlation coefficient"))
    fig.update_xaxes(tickangle=90)
    fig.update_layout(
        # title="Correlation matrix",
        autosize=False,
        width=1000,
        height=1000,
        font=dict(size=18)
    )
    fig.write_html(os.path.join(storage_path, "correlation_matrix.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')
    fig.write_image(os.path.join(storage_path, "correlation_matrix.png"))

    methods_names = ["GLRT", "Robust GLRT", "Log difference"]
    methods = sorted(data['Method'].unique())
    n_images = sorted(data['Number images'].unique())
    fig = make_subplots(rows=len(n_images), cols=len(methods),
                        # column_titles = [f"Method {int(x)}" for x in methods],
                        column_titles = methods_names,
                        row_titles = [f"Image {int(x+1)}" for x in range(len(n_images))])

    for i in range(len(n_images)):
        for j in range(len(methods)):
            data_corr = data_sorted.loc[(data['Number images'] == n_images[i]) & (data['Method'] == methods[j])].copy().drop(['Method', 'Number images'],axis=1)
            if not data_corr.empty:

                fig.add_trace(go.Heatmap(
                    z=np.corrcoef(data_corr.values.T), 
                    x=data_corr.columns, 
                    y=data_corr.columns,
                    colorscale="YlOrRd", 
                    # colorscale="RdBu_r", 
                    zmin=-1, zmax=1, 
                    colorbar=dict(title="Correlation coefficient")
                    ), row=i+1, col=j+1)
                
                fig.update_xaxes(showticklabels=False, row=i+1, col=j+1)
                fig.update_yaxes(showticklabels=False, row=i+1, col=j+1)
            else:
                fig.add_annotation(
                    text="No data",
                    xref="paper", yref="paper",
                    x=0.66, y=0.33,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_xaxes(showticklabels=False, row=i+1, col=j+1)
                fig.update_yaxes(showticklabels=False, row=i+1, col=j+1)

    
    fig.for_each_annotation(lambda a:  a.update(y = -0.2) if a.text in n_images else a.update(x = -0.07) if a.text in methods else())
    for i in range(len(n_images)):
        fig.update_yaxes(showticklabels=True, row=i+1, col=1)
    for j in range(len(methods)):
        fig.update_xaxes(showticklabels=True, row=len(n_images), col=j+1, tickangle=45)
    
    fig.update_layout(
        width=2000,
        height=2400,
        font=dict(size=35))
    fig.update_annotations(font_size=50)
    fig.write_html(os.path.join(storage_path, "correlation_matrices.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')
    fig.write_image(os.path.join(storage_path, "correlation_matrices.png"))

    



def plot_pca(eig, data, data_pca, data_legend, eucl_dist1, eucl_dist2, ccircle, storage_path, title_suffix):
    """Function to plot the PCA .

    Parameters
    ----------
    eig: pandas DataFrame
        Dataframe containing the recordings
    data: pandas DataFrame
        Dataframe containing the recordings
    data_pca: pandas DataFrame
        Dataframe containing the recordings
    data_legend: array
        Label with the setting of all parameters for each point
    eucl_dist1: array
        Array containing the euclidian distances in the PCA first dimensions 1 and 2
    eucl_dist2: array
        Array containing the euclidian distances in the PCA first dimensions 2 and 3
    ccircle: pandas DataFrame
        Dataframe containing the correlations along the PCA dimensions
    storage_path: str
        Path to the folder to save the different plots
    title_suffix: str
        Dataframe containing the recordings
    """
    legend_labels = []
    legend_colors = []
    with plt.style.context(('seaborn-v0_8-whitegrid')):
        fig, axs = plt.subplots(1,3, figsize=(25,10))

        for iax,ax in enumerate(axs):
            if iax == 0:
                ax.bar(range(1, 11), eig["% variance expliquée"][:10], color="#005556")
                # chart formatting
                ax.set_xlabel("PCA dimensions")
                ax.set_ylabel("Explained variance (%)")
                #add values on top of bars
                for i, v in enumerate(eig["% variance expliquée"][:10]):
                    ax.text(i+1, v + 0.5, str(round(v, 2)) + "%", ha='center', va='bottom')
            else:
                eucl_dist = [eucl_dist1, eucl_dist2][iax-1]
                for i,j in enumerate(eucl_dist):
                    # arrow_col = plt.cm.YlOrRd((eucl_dist[i] - np.array(eucl_dist).min())/\
                    #         (np.array(eucl_dist).max() - np.array(eucl_dist).min()) )
                    arrow_col = plt.cm.tab20(i)
                    ax.arrow(0,0, # Arrows start at the origin
                            ccircle[i][iax-1],  #0 for PC1
                            ccircle[i][iax],  #1 for PC2
                            lw = 2, # line width
                            length_includes_head=True, 
                            color = arrow_col,
                            fc = arrow_col,
                            head_width=0.05,
                            head_length=0.05,
                            label = data.columns[i])
                    legend_colors.append(arrow_col)
                    if iax == 1:  # Add legend labels only once
                        legend_labels.append(data.columns[i])
                # Draw the unit circle, for clarity
                circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
                ax.add_patch(circle)
                ax.set_xlabel(f"Dimension {iax}")
                ax.set_ylabel(f"Dimension {iax+1}")
                
    # axs[1].legend(legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    # Create a single legend for the entire figure
    plt.subplots_adjust(bottom=0.3, left=0.05, right=1, top=1)
    handles = [Patch(facecolor=color, edgecolor='k', label=label) for color, label in zip(legend_colors, legend_labels)]
    fig.legend(handles = handles, loc='lower center', bbox_to_anchor=(0.68, 0.03), ncol=4, prop={'size': 20})
    # plt.tight_layout()
    plt.show()
    # tikzplotlib.save(os.path.join(storage_path, f"pca_circle_{title_suffix}.tex"))
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
        # title="Circle of correlations",
        showlegend=False,
        autosize=True
    )
    fig.write_html(os.path.join(storage_path, f"pca_circle_{title_suffix}.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')
    
    # Plot data in PCA representation

    print(eig)
    print("-"*27)
    correlation = pd.DataFrame(ccircle)
    correlation.columns = ["Dim1", "Dim2", "Dim3"]
    correlation.index = data.columns
    print(correlation)

    data_pca = pd.DataFrame(data_pca).set_index(data.index)

    data_3d = pd.concat([data_pca, data_legend], axis = 1)
    col = pd.DataFrame(data_pca).columns.values.tolist()
    col.append("Parameters")

    data_3d.columns = col

    print(data_3d.head)

    fig = px.scatter_3d(data_3d, x=0, y=1, z=2,
                        hover_data="Parameters",
                        color="Parameters",
                        labels={0: f"Dimension 1 ({eig['% variance expliquée'][0]}%)", 1: f"Dimension 2 ({eig['% variance expliquée'][1]}%)", 2: f"Dimension 3 ({eig['% variance expliquée'][2]}%)"},)
    fig.update_layout(
        # title=f"Premiers plans factoriels ({np.sum(eig['% variance expliquée'][0:3])})", 
        legend=dict(title="Parameters")
    )
    fig.write_html(os.path.join(storage_path, f"pca_{title_suffix}.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')


def plot_parameter_distrib(data, storage_path, performance_metric):
    """Function to plot data distribution along the performance and energy consumptions axes.

    Parameters
    ----------
    data: pandas DataFrame
        Dataframe containing the recordings
    storage_path: str
        Path to the folder to save the different plots
    performance_metric: str
        Label of the performance metric used
    """
    for method in data["Method"].unique(): 
        sample_method = data.loc[data["Method"] == method].copy()
        for n_images in sample_method["Number images"].unique(): 
            sample = sample_method.loc[sample_method["Number images"] == n_images].copy()
            sample["Legend"] = "Window size: " + sample["Window size"].astype(str) + ", Threads: " + sample["Threads"].astype(str)

            fig, ax = plt.subplots(1, 1, figsize=(16,8))
            try:
                sns.kdeplot(x = sample["Energy (CodeCarbon)"]/(3.6*1e6), y = sample[performance_metric], hue = sample["Legend"], fill = True, alpha = 0.4, palette = "colorblind", warn_singular=False, ax=ax)
            except:
                warnings.warn("Data distribution does not allow KDE representation.")
            sns.scatterplot(sample, x = "Energy (CodeCarbon)", y = performance_metric, hue = "Legend", alpha = 1, palette = "colorblind")
            ax.set_xlabel("Energy measured with CodeCarbon (kWh)")
            ax.set_ylabel(performance_metric)
            # ax.set_title(f"Plotting energy vs {performance_metric} for {int(n_images)} images with method {int(method)}", ha='left', fontsize=16, loc='left')
            # Get current axis limits
            x_min, x_max = plt.gca().get_xlim()
            y_min, y_max = plt.gca().get_ylim()

            # Add space by extending the limits
            x_padding = (x_max - x_min) * 0.05  # 5% padding
            y_padding = (y_max - y_min) * 0.05  # 5% padding

            plt.gca().set_xlim(x_min - x_padding, x_max + x_padding)
            plt.gca().set_ylim(y_min - y_padding, y_max + y_padding)

            fig.savefig(os.path.join(storage_path, f"perf_energy_ellipse_seaborn_{int(n_images)}images_method{int(method)}.png"), bbox_inches='tight')
            # tikzplotlib.save(os.path.join(storage_path, f"perf_energy_ellipse_seaborn_{int(n_images)}images_method{int(method)}.tex"))
            fig.show()

            grouped = sample.groupby("Legend")
            means = grouped.mean()
            errors = grouped.std()
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            # Scatterplot for the mean values with error bars
            pal = sns.color_palette('colorblind', means.shape[0])
            pal.as_hex()
            # Create legend
            color_map = dict(zip(means.index, pal))
            for i, row in means.iterrows():
                ax.errorbar(
                    row["Energy (CodeCarbon)"] / (3.6 * 1e6),  # Convert energy to kWh
                    row[performance_metric],
                    xerr=errors.loc[i]["Energy (CodeCarbon)"] / (3.6 * 1e6),  # Error in x (energy)
                    yerr=errors.loc[i][performance_metric],  # Error in y (performance metric)
                    fmt='',  # 'o' for scatter plot points
                    ecolor="black",  # Error bar color
                    elinewidth=2,  # Error bar line width
                    capsize=8,  # Error bar cap size
                    color=color_map[i],  # Point color
                    label=None,  # No label for the points
                    markersize=10,  # Point size
                )
                ax.scatter(row["Energy (CodeCarbon)"] / (3.6 * 1e6), row[performance_metric], color=color_map[i], s=100, label=i, zorder=10)
            ax.legend(title="Parameters")

            # Set labels
            ax.set_xlabel("Energy measured with CodeCarbon (kWh)")
            ax.set_ylabel(performance_metric)
            # Extend the axis limits slightly for better visibility
            x_min, x_max = plt.gca().get_xlim()
            y_min, y_max = plt.gca().get_ylim()
            x_padding = (x_max - x_min) * 0.05  # 5% padding
            y_padding = (y_max - y_min) * 0.05  # 5% padding
            plt.gca().set_xlim(x_min - x_padding, x_max + x_padding)
            plt.gca().set_ylim(y_min - y_padding, y_max + y_padding)
            # Save the figure
            fig.savefig(os.path.join(storage_path, f"perf_energy_{int(n_images)}images_method{int(method)}.png"), bbox_inches='tight')
            # tikzplotlib.save(os.path.join(storage_path, f"perf_energy_{int(n_images)}images_method{int(method)}.tex"))
            fig.show()

            fig_plotly = px.scatter(x=sample["Energy (CodeCarbon)"]/(3.6*1e6), y=sample[performance_metric], color=sample['Legend'])
            fig_plotly.update_xaxes(title="Energy measured with CodeCarbon (kWh)")
            fig_plotly.update_yaxes(title=performance_metric)
            fig_plotly.update_layout(legend_title_text='Parameters')
            fig_plotly.write_html(os.path.join(storage_path, f"perf_energy_{int(n_images)}images_method{int(method)}.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')

    sample = data.copy()
    sample["Legend"] = "Window size: " + sample["Window size"].astype(str) + ", Threads: " + sample["Threads"].astype(str) + ", Method: " + sample["Method"].astype(str) + ", Number of images: " + sample["Number images"].astype(str)
    fig, ax = plt.subplots(1, 1, figsize=(16,8))
    fig_plotly = px.scatter(x=sample["Energy (CodeCarbon)"]/(3.6*1e6), y=sample[performance_metric], color=sample['Legend'])
    fig_plotly.update_xaxes(title="Energy measured with CodeCarbon (kWh)")
    fig_plotly.write_html(os.path.join(storage_path, f"perf_energy_all.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')

def plot_stats(storage_path, output_path):
    """Main function to plot all statistics.

    Parameters
    ----------
    storage_path: str
        Path to the folder to save the different plots
    output_path: str
        Path to the output csv file containing results
    """
    print("-"*14)
    print("Analysing data")
    chunk = pd.read_csv(output_path, header=0, chunksize=1000)
    data = pd.concat(chunk)
    if 'Average Precision' in data.columns:
        data = data.drop(['Average Precision'], axis=1)
    if 'Silhouette' in data.columns:
        data = data.drop(['Silhouette'], axis=1)

    data_before_pca = data.drop(['Method'], axis=1)

    eig, data_pca, data_tsne, tsne_div, coordvar, ccircle, eucl_dist1, eucl_dist2 = analyse_stats(data_before_pca)
    data_legend = pd.DataFrame()
    data_legend['Parameters'] = "Window size: " + data["Window size"].astype(int).astype(str) + ", Threads: " + data["Threads"].astype(int).astype(str) + ", Method: " + data["Method"].astype(int).astype(str) + ", Number of images: " + data["Number images"].astype(int).astype(str) 
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
        sample_method = data.loc[data["Method"] == method]
        for n_images in sample_method["Number images"].unique():
                sample = sample_method.loc[sample_method["Number images"] == n_images]
                data_legend_sample = data_legend.loc[(data["Method"] == method) & (data["Number images"] == n_images)]

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

    fig = px.scatter(pd.DataFrame.join(pd.DataFrame(data_tsne), data_legend['Parameters']), color="Parameters", x=0, y=1, 
        # title=f"t-SNE visualization (KL divergence: {tsne_div})", 
        labels={0: "Dimension 1", 1: "Dimension 2"})
    
    fig.write_html(os.path.join(storage_path, "tsne.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')    

    print("-"*31)
    print("Plotting parameter distribution")

    plot_parameter_distrib(data, storage_path, performance_metric)

    print("Plotting duration vs performance")
    fig, ax = plt.subplots(1, 1, figsize=(16,8))
    data_dp = data[data["Window size"] != 1]
    data_dp['Threads'] = data_dp['Threads'].astype(int)
    data_dp['Window size'] = data_dp['Window size'].astype(int)
    data_dp['Method'] = data_dp['Method'].replace({0: "GLRT", 1: "Robust GLRT", 2: "Log difference"})
    sns.scatterplot(x = data_dp["Duration"], y = data_dp[performance_metric], hue = data_dp['Threads'], style = data_dp['Method'], size = data_dp['Window size'],
                    sizes={1:10, 5:50, 7:300, 21:800}, alpha = 1, palette = "colorblind")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel(performance_metric)
    h, l = ax.get_legend_handles_labels()
    print("******************************************")
    print("Legend handles")
    print(h)
    print(l)
    # hi = h[-1]
    # hi.size = 50
    # h[-1] = 
    # hi = h[-2]
    # hi.size = 50
    # h[-2] = hi

    # hi = h[1]
    # hi.size = 50
    # hi.marker = "square"
    # h[1] = hi
    # hi = h[2]
    # hi.size = 50
    # hi.marker = "square"
    # h[2] = hi

    # h[1].set_sizes([50])
    # h[2].set_sizes([50])
    # h[-1].set_sizes([50])
    # h[-2].set_sizes([50])

    lgnd = plt.legend(handles=h, labels=l)
    lgnd.legend_handles[1].set_markersize(25)
    lgnd.legend_handles[1].set_marker('s')
    lgnd.legend_handles[2].set_markersize(25)
    lgnd.legend_handles[2].set_marker('s')
    lgnd.legend_handles[-1].set_markersize(15)
    lgnd.legend_handles[-2].set_markersize(15)
    # ax.set_title(f"Plotting duration vs {performance_metric}", ha='left', fontsize=16, loc='left')
    fig.savefig(os.path.join(storage_path, "duration_performance_small.png"), bbox_inches='tight')
    # tikzplotlib.save(os.path.join(storage_path, "duration_performance.tex"))
    fig.show()


    print("Plotting duration vs performance")
    fig, ax = plt.subplots(1, 1, figsize=(16,8))
    sns.scatterplot(x = data["Duration"], y = data[performance_metric], hue = data_legend['Parameters'], alpha = 1, palette = "colorblind")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel(performance_metric)
    # ax.set_title(f"Plotting duration vs {performance_metric}", ha='left', fontsize=16, loc='left')
    fig.savefig(os.path.join(storage_path, "duration_performance.png"), bbox_inches='tight')
    # tikzplotlib.save(os.path.join(storage_path, "duration_performance.tex"))
    fig.show()

    fig, ax = plt.subplots(1, 1, figsize=(16,8))
    fig = px.scatter(x = data["Duration"], y = data[performance_metric], color = data_legend['Parameters'], color_discrete_sequence=px.colors.qualitative.Dark24,
<<<<<<< HEAD
                    #  title = f"Plotting duration vs {performance_metric}",
                     labels=dict(x="Duration (s)", y=performance_metric, color="Parameters"))
    fig.write_html(os.path.join(storage_path, f"duration_performance.html"), include_mathjax='cdn')
=======
            #    title = f"Plotting duration vs {performance_metric}",
               labels=dict(x="Duration (s)", y=performance_metric, color="Parameters"))
    fig.write_html(os.path.join(storage_path, f"duration_performance.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')
>>>>>>> a2c0c27f3526725ce1370733e5be04c945819ebc

    print("Plotting energy measurement comparison")

    errors = 3*data["Duration"]/(3.6*1e6)

    conso_plug = data["Energy (plug)"]/(3.6*1e6)
    conso_codecarbon = data["Energy (CodeCarbon)"]/(3.6*1e6)    

    model_regress = LinearRegression()
    model_regress.fit(X = np.array(conso_plug).reshape(-1, 1), y = np.array(conso_codecarbon).reshape(-1, 1))
    regress = model_regress.predict(np.array(conso_plug).reshape(-1, 1))
    

    equation_list = [f"{coef[0]}x^{i+1}" for i, coef in enumerate(model_regress.coef_.round(3))]
    equation_intercept = str(model_regress.intercept_.round(3)[0]) + " + " if model_regress.intercept_.round(3)[0] != 0 else ""
    equation = "$" + equation_intercept + " + ".join(equation_list) + "$"
    replace_map = {"x^1": "x", '+ -': '- '}
    for old, new in replace_map.items():
        equation = equation.replace(old, new)
    score = model_regress.score(np.array(conso_codecarbon).reshape(-1, 1),regress)
    score = "$R^2 = " + str(round(score,3)) + "$"

    print(score)

    fig = px.scatter(x = conso_plug, y = conso_codecarbon,
            #    title = "Energy measurement comparison " + score,
               error_x = errors)
    fig.add_trace(go.Scatter(x = conso_plug, y = regress.reshape(-1), mode = 'lines', line_color = "black", name = equation))
    fig.update_xaxes(title = "Energy measured by the plug (kWh)")
    fig.update_yaxes(title = "Energy measured by CodeCarbon (kWh)")
    fig.write_image(os.path.join(storage_path, f"energy_energy.png"))
    fig.write_html(os.path.join(storage_path, f"energy_energy.html"), include_mathjax = 'cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')
    
    print("Plotting frugality score")

    if performance_metric == "AUC":
        norm_perf = 1
    else:
        norm_perf = np.max(data[performance_metric])
    for n_images in data["Number images"].unique():
        data_legend_frugality = pd.DataFrame()
        data_legend_frugality['Parameters'] = "Method: " + data["Method"].replace({0: "GLRT", 1: "Robust GLRT", 2: "Log difference"}).astype(str) + ", Window size: " + data["Window size"].astype(int).astype(str) + ", Threads: " + data["Threads"].astype(int).astype(str)
        data_legend_frugality = data_legend_frugality[data["Number images"] == n_images]
        data_frugality = data[data["Number images"] == n_images]
        for factor in [1, 1e-3, 1e-6]:
            plot_frugality_score(perf = data_frugality[performance_metric], conso = data_frugality["Energy (CodeCarbon)"], legends = data_legend_frugality['Parameters'], storage_path = storage_path,
                                 title = f"frugality_score_{n_images}images_", factor = factor, norm_perf = norm_perf)

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



    output_path = os.path.join(args.storage_path, f"output_all2.csv")
    
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
                elif "Scene_2" in name_images:
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
    
    plot_stats(args.storage_path, output_path)
