# Change detection project

This project repository attest of the first phase of my internship in the LISTIC lab in Annecy, under the supervision of Ammar Mian (https://ammarmian.github.io/).

## Objectives

The objective of this project is to understand and create a strategy to follow hardware performances for remote sensing data algorithms. We are interested in following the energy consumption and the execution time of the algorithms. The task used to test the algorithms is change detection in remote sensing data, in particular between SAR images. We consider a method based on the paper of Conradsen et al. (2015) using covariances within images.

## Content 

Content of this project :
* previous work by Conradsen et al. (2015) in `Conradsen_change_detection` 
* change detection algorithms in `change_detection`
* simulations and retrieval of the performances in `simulations`.

<!--
## Installation

To install the project, you need to clone the repository and install the requirements. You can do this by running the following commands:

```bash
git clone
cd change_detection_project
pip install -r requirements.txt
```
-->

## Running simulations

To run the simulations, you need to run the following command:

```bash
sh simulations/simulation_metrics_exec.sh python3 [path_to_script]
```

The script `simulation_metrics_exec.sh` will run the script `path_to_script` and retrieve the performances of the algorithm in InlfuxDB. The performances are stored in the `simulations` folder.