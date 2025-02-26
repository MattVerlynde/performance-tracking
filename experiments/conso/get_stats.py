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

import sys
sys.path.append('/home/verlyndem/Documents/Tests_change_detection/SAR-change-detection/performance-tracking/experiments')
from get_conso import make_table, query_data, filter_time, get_score 
from conso_change.get_perf import get_perf_change
from conso_clustering.get_perf import get_perf_clustering
from conso_classif_deep.get_perf import get_perf_classif_deep
from conso_clustering.get_perf_blob import get_perf_clustering_blob

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

    params = ["CPU", "Memory", "Energy (plug)", "Temperature", "Reads"]
    for i in range(len(list_integrals)):
        param = params[i]
        list_val = list_integrals[i]
        stats[param] = pd.DataFrame(list_val)

    df_perf = pd.DataFrame()
    for file in os.listdir(os.path.join(os.path.dirname(results), "output")):
        if "change" in storage_path:
            result = np.load(os.path.join(os.path.dirname(results), "output", file))
            _, _, perf = get_perf_change(storage_path, result)
<<<<<<< HEAD
        elif "clustering-blob" in storage_path:
            perf = get_perf_clustering_blob(storage_path)
=======
            df_perf = pd.concat([df_perf, pd.DataFrame([perf])], axis=0, ignore_index=True)
>>>>>>> a2c0c27f3526725ce1370733e5be04c945819ebc
        elif "clustering" in storage_path:
            # result = np.load(os.path.join(os.path.dirname(results), "output", file))
            perf = get_perf_clustering(storage_path)
            df_perf = pd.concat([df_perf, pd.DataFrame([perf])], axis=0, ignore_index=True)
        elif "classif" in storage_path:
            result = pd.read_csv(os.path.join(os.path.dirname(results), "output", file), index_col=0)
            perf = result
            # perf = get_perf_classif_deep(storage_path)
            print(perf)
            df_perf = pd.concat([df_perf, perf], axis=0, ignore_index=True)
        
    
    stats = pd.concat([stats, df_perf], axis=1)
    print(stats)
    stats["Duration"] = pd.DataFrame(get_time(times))

    list_carbon = []
    list_energy_code_carbon = []
    for file in os.listdir(os.path.join(os.path.dirname(results), "codecarbon")):
        codecarbon_path = os.path.join(os.path.dirname(results), "codecarbon", file)
        codecarbon_results = pd.read_csv(codecarbon_path, header=0).iloc[-1,:]
        carbon = codecarbon_results["emissions"]
        list_carbon.append(carbon)
        energy_code_carbon = codecarbon_results["energy_consumed"]*(3.6*1e6)
        list_energy_code_carbon.append(energy_code_carbon)
        
    stats["Emissions"] = pd.Series(list_carbon)
    stats["Energy (CodeCarbon)"] = pd.Series(list_energy_code_carbon)    

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
