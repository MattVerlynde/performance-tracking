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
import tikzplotlib

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

def plot_frugality_score(perf, conso, legends, storage_path, title, factor=1):
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
    for legend in legend_unique:
        sample_perf = np.mean(perf[legends == legend])
        sample_conso = np.mean(conso[legends == legend])*factor
        slope = 1 / (1 + 1/sample_conso)
        frugality_score = np.ones(w.shape)*sample_perf - w * slope
        df[legend] = frugality_score
        df["slope"] = slope

    fig = px.line(df, x = w, y = legend_unique, hover_data = "slope")
    fig.update_xaxes(title_text="w")
    fig.update_yaxes(title_text="Frugality score")
    fig.update_layout(legend_title_text='Parameters')
    fig.write_html(os.path.join(storage_path, title+ str(factor))+".html", include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')

    for legend in df.columns:
        plt.plot(w, df[legend])
    plt.xlabel("w")
    plt.ylabel("Frugality score")
    plt.legend(legend_unique)
    tikzplotlib.save(os.path.join(storage_path, title+ str(factor))+".tex")
    plt.savefig(os.path.join(storage_path, title+ str(factor))+".png")

    
    
    

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
    data_corr = data_sorted.drop('Model', axis=1)
    
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


    model = sorted(data['Model'].unique())
    n_images = [1]
    fig = make_subplots(rows=len(n_images), cols=len(model),
                        column_titles = [x for x in model])

    for i in range(len(n_images)):
        for j in range(len(model)):
            data_corr = data_sorted.loc[data['Model'] == model[j]].copy().drop(['Model'],axis=1)
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
    
    fig.for_each_annotation(lambda a:  a.update(y = -0.2) if a.text in n_images else a.update(x = -0.07) if a.text in model else())
    for i in range(len(n_images)):
        fig.update_yaxes(showticklabels=True, row=i+1, col=1)
    for j in range(len(model)):
        fig.update_xaxes(showticklabels=True, row=len(n_images), col=j+1, tickangle=45)
    
    fig.update_layout(
        width=2000,
        height=500,
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
    with plt.style.context(('seaborn-v0_8-whitegrid')):
        fig, axs = plt.subplots(1,3, figsize=(25,10))

        for iax,ax in enumerate(axs):
            if iax == 0:
                ax.bar(range(1, 11), eig["% variance expliquée"][:10], color="#005556")
                # chart formatting
                ax.set_xlabel("PCA dimensions")
                ax.set_ylabel("Explained variance (%)")
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
                            head_length=0.05)
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
    fig.legend(legend_labels, loc='lower center', bbox_to_anchor=(0.68, 0.03), ncol=4, prop={'size': 20})
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
            fig.for_each_trace(lambda trace: trace.update(line=dict(color='red') if trace.name in ['macro_precision', 'macro_recall', 'macro_f1', 'macro_f2', 'micro_precision', 'micro_recall', 'micro_f1', 'micro_f2'] else dict(color='blue'),
                marker=dict(color='red') if trace.name in ['macro_precision', 'macro_recall', 'macro_f1', 'macro_f2', 'micro_precision', 'micro_recall', 'micro_f1', 'micro_f2'] else dict(color='blue')))

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
    for model in data["Model"].unique(): 
        sample = data.loc[data["Model"] == model].copy()
        sample["Legend"] = "Learning rate: " + sample["Learning rate"].astype(str) + ", Batch size: " + sample["Batch size"].astype(str) + ", Epochs: " + sample["Epochs"].astype(str)

        fig, ax = plt.subplots(1, 1, figsize=(16,8))
        try:
            sns.kdeplot(x = sample["Energy (CodeCarbon)"]/(3.6*1e6), y = sample[performance_metric], hue = sample["Legend"], fill = True, alpha = 0.4, palette = "colorblind", warn_singular=False, ax=ax)
        except:
            warnings.warn("Data distribution does not allow KDE representation.")
        sns.scatterplot(sample, x = "Energy (CodeCarbon)", y = performance_metric, hue = "Legend", alpha = 1, palette = "colorblind")
        ax.set_xlabel("Energy measured with CodeCarbon (kWh)")
        ax.set_ylabel(performance_metric)
        # Get current axis limits
        x_min, x_max = plt.gca().get_xlim()
        y_min, y_max = plt.gca().get_ylim()

        # Add space by extending the limits
        x_padding = (x_max - x_min) * 0.05  # 5% padding
        y_padding = (y_max - y_min) * 0.05  # 5% padding

        plt.gca().set_xlim(x_min - x_padding, x_max + x_padding)
        plt.gca().set_ylim(y_min - y_padding, y_max + y_padding)

        fig.savefig(os.path.join(storage_path, f"perf_energy_ellipse_seaborn_model{model}.png"), bbox_inches='tight')
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
        fig.savefig(os.path.join(storage_path, f"perf_energy_images_model_{model}.png"), bbox_inches='tight')
        tikzplotlib.save(os.path.join(storage_path, f"perf_energy_images_model_{model}.tex"))
        fig.show()

        fig_plotly = px.scatter(x=sample["Energy (CodeCarbon)"]/(3.6*1e6), y=sample[performance_metric], color=sample['Legend'])
        fig_plotly.update_xaxes(title="Energy measured with CodeCarbon (kWh)")
        fig_plotly.update_yaxes(title=performance_metric)
        fig_plotly.update_layout(legend_title_text='Parameters')
        fig_plotly.write_html(os.path.join(storage_path, f"perf_energy_images_model_{model}.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')

    sample = data.copy()
    sample["Legend"] = "Model: " + sample["Model"].astype(str) + ", Learning rate: " + sample["Learning rate"].astype(str) + ", Batch size: " + sample["Batch size"].astype(str) + ", Epochs: " + sample["Epochs"].astype(str)
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

    # data_before_pca = data.drop(['Model'], axis=1)

    # eig, data_pca, data_tsne, tsne_div, coordvar, ccircle, eucl_dist1, eucl_dist2 = analyse_stats(data_before_pca)
    data_legend = pd.DataFrame()
    data_legend['Parameters'] = "Moded: " + data["Model"].astype(str) + ", Batch size: " + data["Batch size"].astype(str) + ", Epochs: " + data["Epochs"].astype(str) + ", Learning rate: " + data["Learning rate"].astype(str) 
    performance_metric = "macro_f1"
    
    print("-"*27)
    print("Plotting correlation matrix")

    # plot_correlation_matrix(data, storage_path)
    
    # print("-"*27)
    # print("Plotting PCA representation")
    
    # plot_pca(eig=eig, data=data_before_pca, data_pca=data_pca, data_legend=data_legend, 
    #          eucl_dist1=eucl_dist1, eucl_dist2=eucl_dist2, ccircle=ccircle, 
    #          storage_path=storage_path,
    #          title_suffix = 'all')
    
    # print("-"*27)
    # print("Plotting PCA representation for each model and dataset")

    # for model in data["Model"].unique():
    #     sample = data.loc[data["Model"] == model]
    #     data_legend_sample = data_legend.loc[data["Model"] == model]

    #     if sample.shape[0] > sample.shape[1]:
    #         eig_sample, data_pca_sample, _, _, _, ccircle_sample, eucl_dist1_sample, eucl_dist2_sample = analyse_stats(sample)

    #         plot_pca(eig=eig_sample, data=sample, data_pca=data_pca_sample, data_legend=data_legend_sample, 
    #                 eucl_dist1=eucl_dist1_sample, eucl_dist2=eucl_dist2_sample, ccircle=ccircle_sample, 
    #                 storage_path=storage_path,
    #                 title_suffix = f"model{model}")
    #     else:
    #         print(f"Not enough samples to plot PCA for model {model}.")

    # print("-"*29)
    # print("Plotting T-SNE representation")

    # fig = px.scatter(pd.DataFrame.join(pd.DataFrame(data_tsne), data_legend['Parameters']), color="Parameters", x=0, y=1, 
    #     # title=f"t-SNE visualization (KL divergence: {tsne_div})", 
    #     labels={0: "Dimension 1", 1: "Dimension 2"})
    
    # fig.write_html(os.path.join(storage_path, "tsne.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')    

    print("-"*31)
    print("Plotting parameter distribution")

    plot_parameter_distrib(data, storage_path, performance_metric)

    print("Plotting duration vs performance")
    fig, ax = plt.subplots(1, 1, figsize=(16,8))
    sns.scatterplot(x = data["Duration"], y = data[performance_metric], hue = data['Model'], style = data['Learning rate'], size = data['Epochs'],
                    sizes={10:50, 20:150, 30:250}, alpha = 1, palette = "colorblind")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel(performance_metric)
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
            #    title = f"Plotting duration vs {performance_metric}",
               labels=dict(x="Duration (s)", y=performance_metric, color="Parameters"))
    fig.write_html(os.path.join(storage_path, f"duration_performance.html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')

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

    for factor in [1, 1e-3, 1e-6]:
        plot_frugality_score(perf = data[performance_metric], conso = data["Energy (CodeCarbon)"], legends = data_legend['Parameters'], storage_path = storage_path,
                                title = "frugality_score", factor = factor)

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
                
                batch = int(paramYaml['parameters']['--batch'])
                epochs = int(paramYaml['parameters']['--epochs'])
                lr = float(paramYaml['parameters']['--lr'])
                model = paramYaml['parameters']['--model']
                rgb = int(paramYaml['parameters']['--rgb'])
                fine_tune = int(paramYaml['parameters']['--finetune'])

                if model == "ShortCNN":
                    if rgb:
                        model="S-CNN-RGB"
                    else:
                        model="S-CNN-All"
                elif model == "InceptionV3":
                    if fine_tune == 1:
                        model+=" (fine tuned)"
                    elif fine_tune == 2:
                        model+=" (from scratch)"
                    else:
                        model=" (transfer learning)"

                output_df_i = get_stats(results, times, os.path.join(args.storage_path, f"run_{id}"), True)
                output_df_i["Batch size"] = batch*np.ones(len(output_df_i))
                output_df_i["Epochs"] = epochs*np.ones(len(output_df_i))
                output_df_i["Learning rate"] = lr*np.ones(len(output_df_i))
                output_df_i["Model"] = [model for m in range(len(output_df_i))]

                output_df = pd.concat([output_df, output_df_i], ignore_index=True)
            else:
                list_group = sorted(os.listdir(os.path.join(args.storage_path, f"run_{id}")))[:-1]
                for group in list_group:
                    results = os.path.join(args.storage_path, f"run_{id}", group, "results.txt")
                    times = os.path.join(args.storage_path, f"run_{id}", group, "times.txt")

                    with open(os.path.join(args.storage_path, f"run_{id}", group, "group_info.yaml"), 'r') as f:
                        paramYaml = yaml.load(f, Loader=yaml.FullLoader)

                    batch = int(paramYaml['parameters']['--batch'])
                    epochs = int(paramYaml['parameters']['--epochs'])
                    lr = float(paramYaml['parameters']['--lr'])
                    model = paramYaml['parameters']['--model']
                    rgb = int(paramYaml['parameters']['--rgb'])
                    fine_tune = int(paramYaml['parameters']['--finetune'])

                    if model == "ShortCNN":
                        if rgb:
                            model="S-CNN-RGB"
                        else:
                            model="S-CNN-All"
                    elif model == "InceptionV3":
                        if fine_tune == 1:
                            model+=" (fine tuned)"
                        elif fine_tune == 2:
                            model+=" (from scratch)"
                        else:
                            model=" (transfer learning)"

                    output_df_i = get_stats(results, times, os.path.join(args.storage_path, f"run_{id}", group), True)
                    output_df_i["Batch size"] = batch*np.ones(len(output_df_i))
                    output_df_i["Epochs"] = epochs*np.ones(len(output_df_i))
                    output_df_i["Learning rate"] = lr*np.ones(len(output_df_i))
                    output_df_i["Model"] = model*np.ones(len(output_df_i))

                    output_df = pd.concat([output_df, output_df_i], ignore_index=True)

        output_df.to_csv(output_path, index=False)
    
    plot_stats(args.storage_path, output_path)
