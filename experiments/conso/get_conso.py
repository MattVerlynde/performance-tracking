# -*- coding: utf-8 -*-
#
# This script is a qanat action executable analysing consumption data from a run of the 'conso' experiment.
# Usage: qanat experiment action conso get-conso [RUN_ID]
#
# Author: Matthieu Verlynde
# Email: matthieu.verlynde@univ-smb.fr
# Date: 20 Jun 2024
# Version: 1.0.0

import argparse
import os
import pandas as pd
import argparse
import numpy as np
import subprocess

def query_data(result_file, time_file):
    """Function to query a pandas dataframe.
    
    Parameters
    ----------
    df: pandas dataframe
        Input database
    query: str
        Query to apply to the database
    """

    command = f"bash performance-tracking/experiments/conso/query_influx.sh {time_file} {result_file}"
    subprocess.run(command, shell=True)
    return
    

def make_table(results):
    """Function to read the results.txt file and return a list of dataframes, corresponding to the different queried variables.
    
    Parameters
    ----------
    results: str
        Path to the results.txt file
    
    Returns
    -------
    df_list: list
        List of pandas dataframes
    """
    with open(results, 'r') as file:
        results_list = file.read().split('#####')
        results_list.pop()
            
    df_list =[]
    for result in results_list:
        df = pd.DataFrame([x.split(',') for x in result.split('\n') if x.strip()])
        df.drop(df.columns[0:5],axis=1,inplace=True)
        df[5] = pd.to_datetime(df[5]).dt.strftime("%d/%m/%Y %H:%M:%S")
        df_list.append(df)
    
    command = f"rm {results}"
    subprocess.run(command, shell=True)

    return df_list

def filter_time(df,time_path):
    """Function to slice an input database using the input timestamps.
    
    Parameters
    ----------
    df: pandas dataframe
        Input database
    time_path: str
        Path to the file containing the timestamps

    Returns
    -------
    df_list: list
        List of pandas dataframes for each time interval
    """
    with open(time_path, 'r') as file:
        time_list = file.read().split('\n')
    
    df_list = []
    for i in range(0,len(time_list)-2):
        # if i != 1:
        t0, t1 = pd.to_datetime(time_list[i]).value, pd.to_datetime(time_list[i+1]).value
        df_time = pd.to_datetime(df.iloc[:,0], dayfirst=True).apply(lambda x: x.value)
        df_inter = df.loc[t0 <= df_time].loc[df_time < t1] 
        # print(df_inter)
        df_list.append(df_inter)
     
    return df_list

def get_integral(t, y, d):
    """Function to slice an input database using the input timestamps.

    Parameters
    ----------
    t: pandas series
        Timestamps
    y: pandas series
        Variable to integrate
    d: float
        Standardisation factor

    Returns
    -------
    integral: float
        Integral of the variable over the time interval
    """
    T = t.iloc[1]-t.iloc[0]
    return T*(float(y.iloc[1])+float(y.iloc[0])-2*d)/2

def get_score(df_list,time_path):
    """Function to slice an input database using the input timestamps.

    Parameters
    ----------
    df_list: pandas dataframe
        Input database
    time_path: str
        Path to the file containing the timestamps

    Returns
    -------
    list_integrals: list
        Integrals of the variables over the time intervals
    """
    with open(time_path, 'r') as file:
        time_list = file.read().split('\n')
        t0, t1 = pd.to_datetime(time_list[0]).value, pd.to_datetime(time_list[1]).value
        T = t1-t0 # Cold time interval in nanoseconds
    
    list_integrals = []

    i_var = 0
    while i_var < len(df_list):
        df_list_var = filter_time(df_list[i_var],time_path)
        i_df = 0
        list_integrals_var = []
        P0 = np.array([[0.1]])
        while i_df < len(df_list_var):
            i_row = 0
            d = 0
            integral = 0

            # df_list_var[i_df].iloc[:,1], P0 = kalmann_filter(df_list_var[i_df].iloc[:,1].astype('float'), P0)

            while i_row < len(df_list_var[i_df])-1:
                x = pd.to_datetime(df_list_var[i_df].iloc[i_row:i_row+2,0], format="%d/%m/%Y %H:%M:%S", dayfirst=True).astype('int')
                y = df_list_var[i_df].iloc[i_row:i_row+2,1]
                i_row += 1
                integral += get_integral(x,y,d)
            if i_df == 0:
                d = integral/T
            elif i_df > 1:
                list_integrals_var.append(integral/1e9) # Converting nanoseconds to seconds
            i_df += 1
        list_integrals.append(list_integrals_var)
        i_var += 1


    return list_integrals

def kalmann_filter(y, P0):
    """
    Function to apply a Kalman filter to a pandas dataframe.

    Parameters
    ----------
    list: list
        Input list with values to filter

    Returns
    -------
    list_filtered: list
        Filtered list
    """

    T = len(y) - 1

    # Define the state transition matrix
    F = np.array([[1]])

    # Define the observation matrix
    H = np.array([[1]])

    # Define the process noise covariance
    Q = np.array([[0.1]])

    # Define the observation noise covariance
    R = np.array([[1]])

    # Define the initial state
    y0 = np.array([y[0]])

    y_filtered = [y0]
    P = [P0]

    for t in range(T):
        # Prediction
        y_filtered.append(F@y[t])
        P.append(F@P[t]@F.T + Q)

        # Update
        K = H @ P[t+1] @ H.T @ np.linalg.inv(H @ P[t+1] @ H.T + R)
        y_filtered[t+1] = y_filtered[t+1] + K @ (y[t+1] - H @ y_filtered[t+1])
        P[t] = (np.eye(1) - K @ H) @ P[t+1]

    return y_filtered, P[-1]





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--query", "-q", type=bool, required=True)
    args = parser.parse_args()

    results = os.path.join(args.storage_path, "results.txt")
    times = os.path.join(args.storage_path, "times.txt")

    if os.path.exists(results):
        os.remove(results)
    query_data(results, times)

    tab_results = make_table(results)

    list_integrals, list_durations = get_score(tab_results,times)
    # print(list_integrals)

    params = ["CPU", "Memory", "Energy", "Temperature", "Reads", "Duration"]
    for i in range(len(list_integrals)):
        param = params[i]
        list_val = list_integrals[i]
        mean, std = np.mean(list_val), np.std(list_val)
        print(f"{param}: {mean} +- {std}")
        np.savetxt(os.path.join(args.storage_path, "conso_metrics.txt"), list_integrals)