#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --job-name=Mario_Long_test_run
#SBATCH --output=job-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu

# Load the Python version that has been used to construct the virtual environment
# we are using below
module load Python/3.10.4-GCCcore-11.3.0

# Activate the virtual environment
source $HOME/venvs/mario-env/bin/activate

# Start the jupyter server, using the hostname of the node as the way to connect to it
python -u $HOME/venvs/mario-env/environment.py
