# -*- coding: utf-8 -*-
#
# This script is a qanat action executable analysing model performances from a run of the 'conso' experiment.
# Usage: qanat experiment action conso get-perf [RUN_ID] --plot [True/False] -r [GROUND_TRUTH_FILE_PATH]
#
# Author: Matthieu Verlynde
# Email: matthieu.verlynde@univ-smb.fr
# Date: 20 Jun 2024
# Version: 1.0.0

import yaml
import argparse
import os
import re
import numpy as np
import argparse
import plotly.express as px
from sklearn import metrics
from skimage.metrics import structural_similarity


def get_perf_change(storage_path, result):
    """Function to 
    
    Parameters
    ----------
    reference: str
        Path to the ground reference file in .npy format
    results: str
        Path to the result file in .npy format
    
    Returns
    -------
    tpr: numpy array
        True Positive Rate for every threshold
    fpr: numpy array
        False Positive Rate for every threshold
    auc: float
        Area Under the Curve
    ssim: float
        Structural Similarity index
    acc: numpy array
        True Positive Rate for every threshold
    """
    with open(os.path.join(storage_path, "group_info.yaml"), 'r') as f:
            paramYaml = yaml.load(f, Loader=yaml.FullLoader)
    ref_path = paramYaml['parameters']['--image'] + "_truth.npy"
    reference = np.load(ref_path)

    x,y = result.shape
    x_ref, y_ref = reference.shape
    gapx, gapy = int((x_ref - x)/2), int((y_ref - y)/2)
    reference = reference[gapx:(x_ref-gapx), gapy:(y_ref-gapy)]

    ref_thresh = np.where(reference != 0, 1, 0)

    fpr, tpr, threshold = metrics.roc_curve(y_true=ref_thresh.flatten(), y_score=result.flatten())
    auc = metrics.auc(fpr, tpr)
    average_precision_score = metrics.average_precision_score(ref_thresh.flatten(), result.flatten())

    perf = {}
    perf["AUC"] = auc
    perf["Average Precision"] = average_precision_score

    return tpr, fpr, perf

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()

    for file in os.listdir(os.path.join(args.storage_path, "output")):
        result = np.load(os.path.join(args.storage_path, "output", file))

        tpr, fpr, auc, ssim = get_perf(args.storage_path, result)
            
        print(f"AUC: {np.mean(auc)}")
        print(f"SSIM: {np.mean(ssim)}")

        if args.plot:
            if not os.path.exists(os.path.join(args.storage_path, "plots")):
                os.mkdir(os.path.join(args.storage_path, "plots"))
            fig = px.scatter(x=fpr, y=tpr, title="ROC Curve", width=600, height=600)
            fig.add_trace(px.line(x=[0, 1], y=[0, 1], color_discrete_sequence=["rgb(0, 0, 0)"]).data[0])
            fig.update_xaxes(title_text="False Positive Rate")
            fig.update_yaxes(title_text="True Positive Rate")
            fig.add_annotation(
            x=0.75,
            y=0.25,
            showarrow=False,
            text="AUC: {:.5f}".format(auc),
            font=dict(
                family="Courier New, monospace",
                size=25,
                color="rgb(0, 0, 0)"
                )
            )
            fig.write_html(os.path.join(args.storage_path, "plots", "roc_curve_" + file[25:-4] + ".html"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')
            fig.write_image(os.path.join(args.storage_path, "plots", "roc_curve_" + file[25:-4] + ".png"))
            fig.show()