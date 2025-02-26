import os
import argparse
import subprocess
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--number_run", "-n", type=str, default="1")
    parser.add_argument("--data_seed", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    results_path = os.path.join(args.storage_path, "results.txt")
    times_path = os.path.join(args.storage_path, "times.txt")

    os.makedirs(os.path.join(args.storage_path, "output"), exist_ok=True)

    command = f"bash performance-tracking/experiments/conso/simulation_metrics_exec.sh {results_path} {times_path} {args.number_run} python performance-tracking/experiments/conso_clustering/utils_clustering_blob.py --storage_path {args.storage_path} --model {args.model} --n_clusters {args.n_clusters} --number_run {args.number_run} --data_seed {args.data_seed} --random_seed {args.random_seed} --repeat {args.repeat}"

    result = subprocess.run(command, shell=True)