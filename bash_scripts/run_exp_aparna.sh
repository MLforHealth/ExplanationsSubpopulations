#!/bin/bash

set -e
cd ../

slurm_pre="--partition t4v2,t4v1,p100,rtx6000 --gres gpu:1 --mem 8gb -c 1 --job-name ${1} --output /scratch/ssd001/home/aparna/explanations-subpopulations/logs/${1}_%A.log"

output_root="/scratch/ssd001/home/aparna/explanations-subpopulations/output_reproduction"

python sweep.py launch \
    --experiment ${1} \
    --output_root "${output_root}" \
    --slurm_pre "${slurm_pre}" \
    --command_launcher "slurm"
