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
# /storage1/fs1/yeli/Active/chani/.conda/envs/dnabert-2/bin/python /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/DNABERT_2/finetune/train.py \
#     --model_name_or_path zhihan1996/DNABERT-2-117M \
#     --data_path /storage1/fs1/yeli/Active/chani/Data/Leuthardt_WGS_GBM_gVCFs/FineTuneData/Cancer1 \
#     --kmer -1 \
#     --run_name DNABERT2__3e-5_Cancer1 \
#     --model_max_length 128 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 3e-5 \
#     --num_train_epochs 3 \
#     --fp16 \
#     --save_steps 200 \
#     --output_dir output/dnabert2/ \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 50 \
#     --logging_steps 100000 \
#     --overwrite_output_dir True \
#     --log_level info \
#     --find_unused_parameters False

# /storage1/fs1/yeli/Active/chani/.conda/envs/dnabert-2/bin/python /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/DNABERT_2/finetune/train.py \
#     --model_name_or_path zhihan1996/DNABERT-2-117M \
#     --data_path /storage1/fs1/yeli/Active/chani/Data/Leuthardt_WGS_GBM_gVCFs/FineTuneData/Cancer2 \
#     --kmer -1 \
#     --run_name DNABERT2__3e-5_Cancer2 \
#     --model_max_length 128 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 3e-5 \
#     --num_train_epochs 3 \
#     --fp16 \
#     --save_steps 200 \
#     --output_dir output/dnabert2/ \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 50 \
#     --logging_steps 100000 \
#     --overwrite_output_dir True \
#     --log_level info \
#     --find_unused_parameters False

# /storage1/fs1/yeli/Active/chani/.conda/envs/dnabert-2/bin/python /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/DNABERT_2/finetune/train.py \
#     --model_name_or_path /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_9_1/ \
#     --data_path /storage1/fs1/yeli/Active/chani/Data/Leuthardt_WGS_GBM_gVCFs/FineTuneData/Cancer1 \
#     --kmer -1 \
#     --run_name DNABERT2__3e-5_Cancer1 \
#     --model_max_length 128 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 3e-5 \
#     --num_train_epochs 3 \
#     --fp16 \
#     --save_steps 200 \
#     --output_dir output/dnabert9_1/ \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 50 \
#     --logging_steps 100000 \
#     --overwrite_output_dir True \
#     --log_level info \
#     --find_unused_parameters False

# /storage1/fs1/yeli/Active/chani/.conda/envs/dnabert-2/bin/python /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/DNABERT_2/finetune/train.py \
#     --model_name_or_path /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_9_1/ \
#     --data_path /storage1/fs1/yeli/Active/chani/Data/Leuthardt_WGS_GBM_gVCFs/FineTuneData/Cancer2 \
#     --kmer -1 \
#     --run_name DNABERT2__3e-5_Cancer2 \
#     --model_max_length 128 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 3e-5 \
#     --num_train_epochs 3 \
#     --fp16 \
#     --save_steps 200 \
#     --output_dir output/dnabert9_1/ \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 50 \
#     --logging_steps 100000 \
#     --overwrite_output_dir True \
#     --log_level info \
#     --find_unused_parameters False

/storage1/fs1/yeli/Active/chani/.conda/envs/dnabert-2/bin/python /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/DNABERT_2/finetune/train.py \
    --model_name_or_path /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_40_10/ \
    --data_path /storage1/fs1/yeli/Active/chani/Data/Leuthardt_WGS_GBM_gVCFs/FineTuneData/Cancer1 \
    --kmer -1 \
    --run_name DNABERT2__3e-5_Cancer1 \
    --model_max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --fp16 \
    --save_steps 200 \
    --output_dir output/dnabert40_10/ \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100000 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False

/storage1/fs1/yeli/Active/chani/.conda/envs/dnabert-2/bin/python /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/DNABERT_2/finetune/train.py \
    --model_name_or_path /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_40_10/ \
    --data_path /storage1/fs1/yeli/Active/chani/Data/Leuthardt_WGS_GBM_gVCFs/FineTuneData/Cancer2 \
    --kmer -1 \
    --run_name DNABERT2__3e-5_Cancer2 \
    --model_max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --fp16 \
    --save_steps 200 \
    --output_dir output/dnabert40_10/ \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100000 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False

/storage1/fs1/yeli/Active/chani/.conda/envs/dnabert-2/bin/python /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/DNABERT_2/finetune/train.py \
    --model_name_or_path /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_100_20/ \
    --data_path /storage1/fs1/yeli/Active/chani/Data/Leuthardt_WGS_GBM_gVCFs/FineTuneData/Cancer1 \
    --kmer -1 \
    --run_name DNABERT2__3e-5_Cancer \
    --model_max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --fp16 \
    --save_steps 200 \
    --output_dir output/dnabert100_20/ \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100000 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False

/storage1/fs1/yeli/Active/chani/.conda/envs/dnabert-2/bin/python /storage1/fs1/yeli/Active/chani/Projects/DNABERT-2/DNABERT_2/finetune/train.py \
    --model_name_or_path /storage2/fs1/btc/Active/yeli/chani/Output_DNABERT2_100_20/ \
    --data_path /storage1/fs1/yeli/Active/chani/Data/Leuthardt_WGS_GBM_gVCFs/FineTuneData/Cancer2 \
    --kmer -1 \
    --run_name DNABERT2__3e-5_Cancer2 \
    --model_max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --fp16 \
    --save_steps 200 \
    --output_dir output/dnabert100_20/ \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100000 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False
