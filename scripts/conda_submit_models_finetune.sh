#!/bin/bash
# Load the Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate dnabert-2

if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
  echo "No Conda environment is activated."
else
  echo "Conda environment '$CONDA_DEFAULT_ENV' is activated."
fi

echo "Running DNABERT 6mer"
sh scripts/run_dnabert1.sh /path/to/GUE/ 6

echo "Running DNABERT 3mer"
sh scripts/run_dnabert1.sh /path/to/GUE/ 3

echo "Running Nucleotide Transformer 1000g"
sh scripts/run_nt.sh /path/to/GUE/ 0

echo "Running Nucleotide Transformer Multi-Species"
sh scripts/run_nt.sh /path/to/GUE/ 1

echo "Running DNABERT2"
sh scripts/run_dnabert2.sh /path/to/GUE/ zhihan1996/DNABERT-2-117M dnabert2
