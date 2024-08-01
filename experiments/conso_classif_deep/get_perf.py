# -*- coding: utf-8 -*-
#
# This script is a qanat action executable analysing model performances from a run of the 'conso' experiment.
# Usage: qanat experiment action conso get-perf [RUN_ID] --plot [True/False] -r [GROUND_TRUTH_FILE_PATH]
#
# Author: Matthieu Verlynde
# Email: matthieu.verlynde@univ-smb.fr
# Date: 20 Jun 2024
# Version: 1.0.0

import yaml
import argparse
import os
import argparse
import subprocess
import json


def get_perf_deep_classif(storage_path, ):
    """Function to 
    
    Parameters
    ----------
    storage_path: str
        Path to the experiment run
    
    Returns
    -------
    f1_score: numpy array
        True Positive Rate for every threshold
    f2_score: numpy array
        False Positive Rate for every threshold
    precision: float
        Area Under the Curve
    recall: float
        Structural Similarity index
    """
    with open(os.path.join(storage_path, "group_info.yaml"), 'r') as f:
            paramYaml = yaml.load(f, Loader=yaml.FullLoader)
    config_path = paramYaml['parameters']['--configs']
    
    command = 'python performance-tracking/experiments/conso_classif_deep/eval.py --settings {} --storage_path {}'.format(config_path, storage_path)

    subprocess.run(command, shell=True)

    with open(os.path.join(storage_path, 'eval_result.json'), 'rb') as f:
        results = json.load(f)

    return results['sample_f1_score'], results['sample_f2_score'], results['sample_precision'], results['sample_recall']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    parser_args = parser.parse_args()

    f1_score, f2_score, precision, recall = get_perf_deep_classif(parser_args.storage_path)

    
    