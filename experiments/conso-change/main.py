# Example code of change detection in a time series of SAR images

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from functions import SARImageTimeSeriesReader, GaussianChangeDetector

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAR Image Time Series Change Detection")
    parser.add_argument("--dataset_path", type=str, required=True,
        help="Path to data location. Assumption is one .npy file per date."
            " Ordered by date.")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Size of the Sliding Window")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of threads for the simulation")
    parser.add_argument("--storage_path", type=str, default="./temp",
                        help="Where to store the results")
    parser.add_argument("--preload", action='store_true',
                        help="Whether to preload all the data.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level.")
    args = parser.parse_args()

    # Print configuration
    log_str = f"Change detection over {args.dataset_path} with parameters:\n"
    for param_name, param_value in vars(args).items():
        log_str += f"    * {param_name}: {param_value}\n"
    print(log_str)

    # Data reader
    dataset = SARImageTimeSeriesReader(args.dataset_path,
                                       preload=args.preload,
                                       verbose=args.verbose,
                                       crop_indexes=None)

    # Gaussian change detection
    detector = GaussianChangeDetector(window_size=args.window_size,
                                      n_jobs=args.n_jobs, verbose=args.verbose)
    pval = detector.fit_predict(dataset)

    # Showing time series
    for t in range(len(dataset)):
        plt.figure()
        plt.imshow(20*np.log10(np.sum(np.abs(dataset[t]**2), axis=2)),
                   aspect="auto", cmap="gray")
        plt.colorbar()
        plt.title(f"Date: {t+1}")

    # Showing results raw
    fig = plt.figure()
    plt.imshow(-detector.lnQ_all, aspect="auto")
    plt.colorbar()
    plt.savefig(os.path.join(args.storage_path, "lnQ_all.png"))
    plt.show()

    fig = plt.figure()
    plt.imshow(pval, aspect="auto")
    plt.colorbar()
    plt.savefig(os.path.join(args.storage_path, "pval.png"))
    plt.show()
    
