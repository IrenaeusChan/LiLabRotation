#!/bin/bash
# Load the Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate dnabert-2

if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
  echo "No Conda environment is activated."
else
  echo "Conda environment '$CONDA_DEFAULT_ENV' is activated."
fi

# Run your Python script
sh scripts/run_dnabert2.sh /path/to/GUE/ zhihan1996/DNABERT-2-117M dnabert2
sh scripts/run_dnabert2.sh /path/to/GUE/ /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_9_1/ dnabert9_1
sh scripts/run_dnabert2.sh /path/to/GUE/ /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_40_10/ dnabert40_10
sh scripts/run_dnabert2.sh /path/to/GUE/ /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_100_20/ dnabert100_20
