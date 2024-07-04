# -*- coding: utf-8 -*-
#
# This script is a qanat action executable analysing model performances from a run of the 'conso' experiment.
# Usage: qanat experiment action conso get-stats [RUN_ID] -r [GROUND_TRUTH_FILE_PATH] [-q [True/False]]
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

from get_conso import make_table, query_data, filter_time, get_score 
from get_perf import get_perf

def get_time(times):
    """
    """
    durations = []
    with open(times, 'r') as file:
        time_list = file.read().split('\n')
    for i in range(2,len(time_list)-2):
        t0, t1 = pd.to_datetime(time_list[i]).value, pd.to_datetime(time_list[i+1]).value
        durations.append((t1-t0)/1e9)
    return durations



def get_stats(results, times, storage_path, query=True):
    """Function to get the statistics from the results and the performance metrics.
    
    Parameters
    ----------
    results: str
        Path to the results.txt file
    times: str
        Path to the stdout.txt file
    output: str
        Path to the output csv file
    ref_path: str
        Path to the reference file in .npy format
    query: bool
        If True, query the results.txt file

    Returns
    -------
    stats: pandas dataframe
        Dataframe containing the statistics
    """
    stats = pd.DataFrame()

    if query:
        if os.path.exists(results):
            os.remove(results)
        query_data(results, times)
    tab_results = make_table(results)
    list_integrals = get_score(tab_results,times)

    params = ["CPU", "Memory", "Energy", "Temperature", "Reads"]
    for i in range(len(list_integrals)):
        param = params[i]
        list_val = list_integrals[i]
        stats[param] = pd.DataFrame(list_val)

    list_auc = []
    list_ssim = []
    for file in os.listdir(os.path.join(os.path.dirname(results), "output")):
        result = np.load(os.path.join(os.path.dirname(results), "output", file))
        tpr, fpr, auc, ssim = get_perf(storage_path, result)
        list_auc.append(auc)
        list_ssim.append(ssim)
    
    stats["AUC"] = pd.DataFrame(list_auc)
    stats["SSIM"] = pd.DataFrame(list_ssim)

    stats["Duration"] = pd.DataFrame(get_time(times))

    list_carbon = []
    for file in os.listdir(os.path.join(os.path.dirname(results), "codecarbon")):
        result = os.path.join(os.path.dirname(results), "codecarbon", file)
        carbon = pd.read_csv(result, header=0)["emissions"].values[-1]
        list_carbon.append(carbon)
        
    stats["Emissions"] = pd.DataFrame(list_carbon)

    return stats

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--query", "-q", type=bool, default=True)
    args = parser.parse_args()

    results = os.path.join(args.storage_path, "results.txt")
    times = os.path.join(args.storage_path, "times.txt")
    output = os.path.join(args.storage_path, "output.csv")

    get_stats(results, times, args.storage_path, args.query).to_csv(output, index=False)
