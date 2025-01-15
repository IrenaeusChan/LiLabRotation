#!/bin/bash
# Load the Conda environment
source /opt/conda/etc/profile.d/conda.sh
#conda activate dnabert-2
conda activate pretrain

if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
  echo "No Conda environment is activated."
else
  echo "Conda environment '$CONDA_DEFAULT_ENV' is activated."
fi

# Run your Python script
#/storage1/fs1/yeli/Active/chani/.conda/envs/dnabert-2/bin/python /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/check_dnabert-2_install.py

#sh scripts/run_dnabert1.sh 6
#sh scripts/run_dnabert1.sh 3
#sh scripts/run_nt.sh /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/ 0
#sh scripts/run_dnabert2.sh /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/

echo "Trying to run script..."
python $LI/Projects/PracticePreTrain/transformers/examples/pytorch/language-modeling/run_mlm.py
echo "This should be the help message for the script above."

TOKENIZERS_PARALLELISM=true python $LI/Projects/PracticePreTrain/transformers/examples/pytorch/language-modeling/run_mlm.py --model_name_or_path zhihan1996/DNABERT-2-117M \
    --train_file /storage2/fs1/btc/Active/yeli/chani/100_train.txt \
    --validation_file /storage2/fs1/btc/Active/yeli/chani/20_val.txt \
    --trust_remote_code True \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --do_train \
    --do_eval \
    --line_by_line True \
    --output_dir /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_100_20/ \
    --run_name 100_20 \
    --logging_steps="5000" \
    --save_steps="5000" \
    --eval_steps="5000" \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 4
echo "Script should have run."