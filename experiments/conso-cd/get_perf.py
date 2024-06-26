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
from scipy import integrate
from skimage.metrics import structural_similarity


def get_perf(storage_path, result):
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

    tpr = []
    fpr = []
    acc = []
    ref_thresh = np.where(reference != 0, 1, 0)
    for t in np.linspace(0, 1, 100):
        result_thresh = np.where(result >= t, 1, 0)
        TP = np.sum(np.logical_and(ref_thresh == 1, result_thresh == 1))
        FP = np.sum(np.logical_and(ref_thresh == 0, result_thresh == 1))
        FN = np.sum(np.logical_and(ref_thresh == 1, result_thresh == 0))
        TN = np.sum(np.logical_and(ref_thresh == 0, result_thresh == 0))
        # print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))
        acc.append((TP + TN) / (TP + TN + FP + FN))
    
    #Calculating AUC
    auc = abs(integrate.trapezoid(tpr, fpr))

    ssim = structural_similarity(ref_thresh, result, data_range=1.0)

    return tpr, fpr, auc, ssim, acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()

    for file in os.listdir(os.path.join(args.storage_path, "output")):
        result = np.load(os.path.join(args.storage_path, "output", file))

        tpr, fpr, auc, ssim, acc = get_perf(args.storage_path, result)
            
        # print(f"TPR: {tpr}")
        # print(f"FPR: {fpr}")
        print(f"ACC: {np.mean(acc)}")
        print(f"AUC: {np.mean(auc)}")
        print(f"SSIM: {np.mean(ssim)}")
        # print(f"Threshold: {threshold}")

        if args.plot:
            if not os.path.exists(os.path.join(args.storage_path, "plots")):
                os.mkdir(os.path.join(args.storage_path, "plots"))
                os.mkdir(os.path.join(args.storage_path, "plots/accuracy"))
                os.mkdir(os.path.join(args.storage_path, "plots/roc"))
            fig = px.line(x=fpr, y=tpr, title="ROC Curve", width=600, height=600)
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
            fig.write_html(os.path.join(args.storage_path, "plots/roc", "roc_curve_" + file[25:-4] + ".html"), include_mathjax='cdn')
            fig.show()

            fig = px.line(x=np.linspace(0, 1, 100), y=acc, title="Accuracy", width=600, height=600)
            fig.update_xaxes(title_text="Threshold")
            fig.update_yaxes(title_text="Accuracy")
            fig.write_html(os.path.join(args.storage_path, "plots/accuracy", "acc_curve_" + file[25:-4] + ".html"), include_mathjax='cdn')
            fig.show()