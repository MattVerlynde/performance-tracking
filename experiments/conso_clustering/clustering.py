import os
import argparse
import subprocess
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", '-i', type=str, default='/home/verlyndem/Data/Selection/Scene_1/Scene_1_0.npy')
    parser.add_argument("--window", '-w', type=int, required=True)
    parser.add_argument("--cores", '-c', type=float, required=True)
    parser.add_argument("--n_clusters", type=int, default=2)
    parser.add_argument("--number_run", '-n', type=int, default=1)
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--riemann", type=float, required=True)
    args = parser.parse_args()

    results_path = os.path.join(args.storage_path, "results.txt")
    times_path = os.path.join(args.storage_path, "times.txt")

    os.makedirs(os.path.join(args.storage_path, "output"), exist_ok=True)

    command = f"bash performance-tracking/experiments/conso/simulation_metrics_exec.sh {results_path} {times_path} {args.number_run} python3 performance-tracking/experiments/conso_clustering/utils_clustering.py --data_path {args.image} --window {args.window} --n_jobs {int(args.cores)} --riemann {int(args.riemann)} --storage_path {args.storage_path}"

    result = subprocess.run(command, shell=True)

