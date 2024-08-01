import os
import argparse
import subprocess


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--image", '-i', type=str, default='/home/verlyndem/Data/Selection/Scene_1/Scene_1_0.npy')
    # parser.add_argument("--window", '-w', type=int, required=True)
    # parser.add_argument("--cores", '-c', type=float, required=True)
    # parser.add_argument("--n_clusters", type=int, default=2)
    parser.add_argument("--number_run", '-n', type=int, default=1)
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument('--configs', type=str, help= 'json config file')

    args = parser.parse_args()

    results_path = os.path.join(args.storage_path, "results.txt")
    times_path = os.path.join(args.storage_path, "times.txt")

    os.makedirs(os.path.join(args.storage_path, "output"))

    command = 'bash performance-tracking/experiments/conso_classif_deep/simulation_metrics_exec.sh {} {} {} python performance-tracking/experiments/conso_classif_deep/train.py --configs {} --storage_path {}'.format(results_path, times_path, args.number_run, args.configs, args.storage_path)
    os.system(command)

    command = 'python performance-tracking/experiments/conso_classif_deep/eval.py --settings {} --storage_path {}'.format(args.configs, args.storage_path)
    os.system(command)