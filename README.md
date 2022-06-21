# The Road to Explainability is Paved with Bias: Measuring the Fairness of Explanations <!-- omit in toc -->
An examination of the approximation quality of post-hoc explainability methods between subgroups. For more details please see our [FaCCT 2022 paper](https://arxiv.org/abs/2205.03295).

## Contents <!-- omit in toc -->
- [Setting Up](#setting-up)
  - [1. Environment and Prerequisites](#1-environment-and-prerequisites)
  - [2. Obtaining the Data](#2-obtaining-the-data)
- [Main Experimental Grid](#main-experimental-grid)
  - [1. Running Experiments](#1-running-experiments)
  - [2. Aggregating Results](#2-aggregating-results)
- [Auxiliary Experiments](#auxiliary-experiments)
  - [Law School Admissions Simulation](#law-school-admissions-simulation)
  - [Group Information in Feature Representations](#group-information-in-feature-representations)
- [Citation](#citation)


## Setting Up
### 1. Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment:

```
git clone git@github.com:MLforHealth/ExplanationsSubpopulations.git
cd ExplanationsSubpopulations/
conda env create -f environment.yml
conda activate fair_exp
```

### 2. Obtaining the Data
We provide the `adult`, `recidivism`, `lsac` datasets as .csv files in this repository. For access to the `mimic` dataset, see [ProcessMIMIC.md](ProcessMIMIC.md).


## Main Experimental Grid
### 1. Running Experiments
To reproduce the experiments in the paper which involve training grids of models using different hyperparameters, use `sweep.py` as follows:

```
python sweep.py launch \
    --experiment {experiment_name} \
    --output_dir {output_root} \
    --command_launcher {launcher} 
```

where:
- `experiment_name` corresponds to experiments defined as classes in `experiments.py`
- `output_root` is a directory where experimental results will be stored.
- `launcher` is a string corresponding to a launcher defined in `launchers.py` (i.e. `slurm` or `local`).

Sample bash scripts showing the command can also be found in `bash_scripts/`.

Alternatively, a single model can also be trained at once by calling `run.py` with the appropriate arguments, for example:

```
python run.py \
    --dataset adult \
    --blackbox_model lr \
    --explanation_type local \
    --explanation_model lime \
    --n_features 5 \
    --model_type sklearn \
    --evaluate_val \
    --seed 1
```

### 2. Aggregating Results
We aggregate results and generate tables used in the paper similar to the `notebooks/localmodels.ipynb` notebook. Another example has been shown in the `notebooks/agg_results.ipynb` notebook.

## Auxiliary Experiments
### Law School Admissions Simulation
We provide the notebook used to simulate the real-world impact of biased explanations on law school admissions in `notebooks/simulate_explanation_impact.ipynb`.

### Group Information in Feature Representations
To reproduce the experiment described in Section 6.2 of the paper, run the `DatasetDists` experiment, and aggregate results using the `notebooks/agg_dist_results.ipynb` notebook.

## Citation
If you use this code in your research, please cite the following publication:
```
@article{balagopalan2022road,
  title={The Road to Explainability is Paved with Bias: Measuring the Fairness of Explanations},
  author={Balagopalan, Aparna and Zhang, Haoran and Hamidieh, Kimia and Hartvigsen, Thomas and Rudzicz, Frank and Ghassemi, Marzyeh},
  journal={arXiv preprint arXiv:2205.03295},
  year={2022}
}

```
