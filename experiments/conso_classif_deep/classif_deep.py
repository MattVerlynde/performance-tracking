import os
import argparse
import subprocess
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", '-m', type=str, default='ShortCNN')
    parser.add_argument("--epochs", '-e', type=float, default=10)
    parser.add_argument("--optim", '-o', type=str, default='SGD')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", '-l', type=str, default='CrossEntropy')
    parser.add_argument("--batch", '-b', type=float, default=256)
    parser.add_argument("--count", type=float, default=0)
    parser.add_argument("--rgb", type=float, default=1)
    parser.add_argument("--finetune", type=float, default=0)
    parser.add_argument("--seed", type=float, default=42)
    parser.add_argument("--storage_path", type=str, required=True)
    args = parser.parse_args()

    args.epochs = int(args.epochs)
    args.batch = int(args.batch)
    args.finetune = int(args.finetune)
    args.seed = int(args.seed)
    args.count = "--count" if args.count else "--no-count"
    args.rgb = "--rgb" if args.rgb else "--no-rgb"

    results_path = os.path.join(args.storage_path, "results.txt")
    times_path = os.path.join(args.storage_path, "times.txt")

    os.makedirs(os.path.join(args.storage_path, "output"), exist_ok=True)

    command = f"bash performance-tracking/experiments/conso_classif_deep/simulation_metrics_exec.sh {results_path} {times_path} python3 train.py --model {args.model} --epochs {args.epochs} --optim {args.optim} --lr {args.lr} --loss {args.loss} --batch {args.batch} --seed {args.seed} {args.count} {args.rgb} --storage_path {args.storage_path}"

    result = subprocess.run(command, shell=True)