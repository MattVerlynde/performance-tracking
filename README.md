# Performance tracking

This repository is used to track the performances of different models in terms of hardware usage and inference time. These comparisons are made using data fetched on InfluxDB, a time-series database, and visualized on Grafana, a data visualization tool.
This repository is made to be used as a submodule to apply on your data after construction of your pipeline. 

## Repository structure

```bash
.
├── experiments
│   ├── conso
│   │   ├── analyse_stats.py
│   │   ├── get_conso.py
│   │   ├── get_stats.py
│   │   ├── query_influx.sh
│   │   ├── simulation_metrics_exec.sh
│   │   ├── stats_summary_blob.py
│   │   ├── stats_summary_deep.py
│   │   └── stats_summary.py
│   ├── conso_change
│   │   ├── cd_sklearn_pair_var.py
│   │   ├── change-detection.py
│   │   ├── functions.py
│   │   ├── get_perf.py
│   │   ├── helpers
│   │   │   └── multivariate_images_tool.py
│   │   └── main.py
│   ├── conso_classif_deep
│   │   ├── classif_deep.py
│   │   ├── get_perf.py
│   │   ├── get_scores.py
│   │   ├── read_event.py
│   │   ├── read_events.py
│   │   └── simulation_metrics_exec.sh
│   └── conso_clustering
│       ├── clustering_blob.py
│       ├── clustering.py
│       ├── get_perf_blob.py
│       ├── get_perf.py
│       ├── helpers
│       │   └── processing_helpers.py
│       ├── plot_clustering.py
│       ├── utils_clustering_blob.py
│       └── utils_clustering.py
├── plot_usage.py
├── README.md
└── simulation_metrics_exec.sh
```

## Installation

### Dependencies

python 3.11.8, codecarbon 2.3.4, numpy 1.26.4, pandas 2.2.1, scikit-learn 1.4.1
```bash
conda env create -f environment.yml
conda activate frugal-score
```

## Usage

| File | Associated command | Description |
| ---- | ------------------ | ----------- |
| [change-detection.py](experiments/conso_change/change-detection.py)  | `python change-detection.py --storage_path [PATH_TO_FOLDER_TO_STORE_RESULTS] --image [PATH_TO_FOLDER_WITH_IMAGES] --window [WINDOW_SIZE] --cores [NUMBER_OF_CORES_USED] --number_run [NUMBER_OF_RUNS] --robust [ROBUSTNESS ID]` | Runs change detection algorithms on UAVSAR data |
| [clustering_blob.py](experiments/conso_clustering/clustering_blob.py)  | `python clustering_blob.py --storage_path [PATH_TO_FOLDER_TO_STORE_RESULTS] --data_seed [SEED] --random_seed [SEED] --n_clusters [NUMBER_OF_CLUSTERS] --model [CLUSTERING_METHOD] --repeat [NUMBER_OF_MODEL_RUNS] --number_run/-n [NUMBER_OF_RUNS]` | Runs clustering algorithms on toy data |

## Authors

* [Matthieu Verlynde](https://github.com/MattVerlynde) ([matthieu.verlynde@univ-smb.fr](mailto:matthieu.verlynde@univ-smb.fr))
* [Ammar Mian](https://ammarmian.github.io/) ([ammar.mian@univ-smb.fr](mailto:ammar.mian@univ-smb.fr))
* [Yajing Yan](https://www.univ-smb.fr/listic/en/presentation_listic/membres/enseignants-chercheurs/yajing-yan-fr/) ([yajing.yan@univ-smb.fr](mailto:yajing.yan@univ-smb.fr))
