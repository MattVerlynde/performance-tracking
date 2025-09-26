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

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_conso import make_table, query_data, filter_time, get_score 
# from conso_change.get_perf import get_perf_change
# from conso_clustering.get_perf import get_perf_clustering
# from conso_classif_deep.get_perf import get_perf_classif_deep
# from conso_clustering.get_perf_blob import get_perf_clustering_blob

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
    stats_not_filtered = pd.DataFrame()

    if query:
        if os.path.exists(results):
            os.remove(results)
        query_data(results, times)
        tab_results = make_table(results)
        list_integrals, list_integrals_not_filtered = get_score(tab_results,times)

        params = ["CPU", "Memory", "Energy (plug)", "Temperature", "Reads"]
        for i in range(len(list_integrals)):
            param = params[i]
            list_val = list_integrals[i]
            list_val_not_filtered = list_integrals_not_filtered[i]
            print(list_val)
            stats[param] = pd.DataFrame(list_val)
            stats_not_filtered[param] = pd.DataFrame(list_val_not_filtered)

    df_perf = pd.DataFrame()
    # for file in os.listdir(os.path.join(storage_path, "output")):
    #     if "change" in storage_path:
    #         result = np.load(os.path.join(storage_path, "output", file))
    #         _, _, perf = get_perf_change(storage_path, result)
    #         df_perf = pd.concat([df_perf, pd.DataFrame([perf])], axis=0, ignore_index=True)
    #     elif "clustering-blob" in storage_path:
    #         perf = get_perf_clustering_blob(storage_path)
    #         df_perf = pd.concat([df_perf, pd.DataFrame([perf])], axis=0, ignore_index=True)
    #     elif "clustering" in storage_path:
    #         # result = np.load(os.path.join(os.path.dirname(results), "output", file))
    #         perf = get_perf_clustering(storage_path)
    #         df_perf = pd.concat([df_perf, pd.DataFrame([perf])], axis=0, ignore_index=True)
    #     elif "classif" in storage_path:
    #         result = pd.read_csv(os.path.join(storage_path, "output", file), index_col=0)
    #         perf = result
    #         # perf = get_perf_classif_deep(storage_path)
    #         df_perf = pd.concat([df_perf, perf], axis=0, ignore_index=True)
        
    if query:
        stats = pd.concat([stats, df_perf], axis=1)
        stats_not_filtered = pd.concat([stats_not_filtered, df_perf], axis=1)
    else:
        stats[df_perf.columns] = df_perf
        stats_not_filtered[df_perf.columns] = df_perf
    stats["Duration"] = pd.DataFrame(get_time(times))
    stats_not_filtered["Duration"] = pd.DataFrame(get_time(times))

    # list_carbon = []
    # list_carbon_process = []
    # list_carbon_filtered = []
    # list_energy_code_carbon = []
    # list_energy_code_carbon_process = []
    # list_energy_code_carbon_filtered = []
    # for file in os.listdir(os.path.join(os.path.dirname(results), "codecarbon")):
    #     codecarbon_path = os.path.join(os.path.dirname(results), "codecarbon", file)
    #     codecarbon_results = pd.read_csv(codecarbon_path, header=0).iloc[1,:]
    #     codecarbon_results_process = pd.read_csv(codecarbon_path, header=0).iloc[0,:]
    #     codecarbon_results_back = pd.read_csv(os.path.join(os.path.dirname(results), "..", "output-back-gpu", "codecarbon", "emissions_1.csv"), header=0).iloc[1,:]
    #     carbon = codecarbon_results["emissions"]
    #     carbon_process = codecarbon_results_process["emissions"]
    #     carbon_filtered = codecarbon_results["emissions"] - (codecarbon_results_back["emissions"] / codecarbon_results_back["duration"]) * codecarbon_results["duration"]
    #     list_carbon.append(carbon)
    #     list_carbon_process.append(carbon_process)
    #     list_carbon_filtered.append(carbon_filtered)
    #     energy_code_carbon = codecarbon_results["energy_consumed"]*(3.6*1e6) # Joules
    #     energy_code_carbon_process = codecarbon_results_process["energy_consumed"]*(3.6*1e6) # Joules
    #     energy_code_carbon_filtered = codecarbon_results["energy_consumed"]*(3.6*1e6) - (codecarbon_results_back["energy_consumed"]*(3.6*1e6) / codecarbon_results_back["duration"]) * codecarbon_results["duration"]
    #     list_energy_code_carbon.append(energy_code_carbon)
    #     list_energy_code_carbon_process.append(energy_code_carbon_process)
    #     list_energy_code_carbon_filtered.append(energy_code_carbon_filtered)
        
    # stats["Emissions"] = pd.Series(list_carbon)
    # stats["Emissions (process)"] = pd.Series(list_carbon_process)
    # # stats["Energy (CodeCarbon)"] = pd.Series(list_energy_code_carbon)    
    # # stats["Energy (CodeCarbon process)"] = pd.Series(list_energy_code_carbon_process)
    # stats["Emissions (filtered)"] = pd.Series(list_carbon_filtered)
    # stats["Energy (CodeCarbon filtered)"] = pd.Series(list_energy_code_carbon_filtered)
    stats["Energy (plug not filtered)"] = pd.Series(stats_not_filtered["Energy (plug)"])

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
