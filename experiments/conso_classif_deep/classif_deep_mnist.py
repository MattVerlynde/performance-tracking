import os
import argparse
import subprocess
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ImageNet Classification with PyTorch')
    parser.add_argument('--num_epochs', type=float, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=float, default=512, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training and validation')
    parser.add_argument('--data_root', type=str, default='/media/HDD/MNIST', help='Root directory for MNIST dataset')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture to use (e.g., resnet18, resnet50, vgg16, etc.)')
    parser.add_argument('--pretrained', type=str, default='True', help='Use pretrained model weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cuda, cpu)')
    parser.add_argument('--storage_path', type=str, default='./results', help='Path to store results and logs')
    parser.add_argument('--num_workers', type=float, default=0, help='Number of workers for data loading')
    parser.add_argument('--seed', type=float, default=37, help='Random seed for reproducibility')
    parser.add_argument('--prefetch_factor', type=float, default=2, help='Number of batches to prefetch')
    args = parser.parse_args()

    args.num_epochs = int(args.num_epochs)
    args.batch_size = int(args.batch_size)
    args.seed = int(args.seed)
    args.prefetch_factor = int(args.prefetch_factor)
    args.num_workers = int(args.num_workers)
    args.learning_rate = float(args.learning_rate)

    results_path = os.path.join(args.storage_path, "results.txt")
    times_path = os.path.join(args.storage_path, "times.txt")

    os.makedirs(os.path.join(args.storage_path, "output"), exist_ok=True)

    command = f"bash performance-tracking/experiments/conso_classif_deep/simulation_metrics_exec.sh {results_path} {times_path} python classif_torch_mnist.py --model {args.model} --num_epochs {args.num_epochs} --batch_size {args.batch_size} --data_root {args.data_root} --pretrained {args.pretrained} --device {args.device} --num_workers {args.num_workers} --seed {args.seed} --prefetch_factor {args.prefetch_factor} --learning_rate {args.learning_rate} --storage_path {args.storage_path}"

    result = subprocess.run(command, shell=True)