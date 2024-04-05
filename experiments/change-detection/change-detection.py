import os
import argparse
import subprocess


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='500x500x4')
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--cores", type=int, default=12)
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--storage_path", type=str, required=True)
    args = parser.parse_args()

    results_path = os.path.join(args.storage_path, "results.txt")

    command = f"bash experiments/change-detection/simulation_metrics_exec.sh {results_path} {args.password} python3 experiments/change-detection/cd_sklearn_pair_var.py {args.image} {args.window} {args.cores}"

    result = subprocess.run(command, shell=True)