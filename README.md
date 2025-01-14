# Li Lab Rotation
Tutorial to reproduce work done during Li Lab rotation

## Contents
1. [Model Comparisons: DNABERT (6mer, 3mer), DNABERT2, Nucleotide-Transformer (Multi-Species, 1000g)](https://github.com/IrenaeusChan/LiLabRotation?tab=readme-ov-file#1-model-comparisons)
2. [Prepare Data for Pre-Training](https://github.com/IrenaeusChan/LiLabRotation)
3. [Pre-Train DNABERT-2](https://github.com/IrenaeusChan/LiLabRotation)
4. [Fine-Tune DNABERT-2](https://github.com/IrenaeusChan/LiLabRotation)
5. [Compare](https://github.com/IrenaeusChan/LiLabRotation)

## 1. Model Comparisons
For each of the foundation models used in this overview, please refer to their individual Github repositories for installation and usage:
[DNABERT](https://github.com/jerryji1993/DNABERT)  
[DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2)  
[Nucleotide-Transformer](https://github.com/instadeepai/nucleotide-transformer)

However, for most of the work done here we will be using the already pre-trained models which are available on Huggingface

### 1.1. Setup conda environment

There will be two different conda environments that are required for performing the tasks shown here. The first of which will be dnabert-2 which is simply following the installation instructions provided by DNABERT_2 Github README page.
```
# Create and activate the virtual python environment
conda create -n dnabert-2 python=3.8
conda activate dnabert-2

# Install required packages (the requirements.txt will be provided from https://github.com/MAGICS-LAB/DNABERT_2)
python3 -m pip install -r requirements.txt

# Uninstall Triton - For some reason the Flash Attention from Triton is no longer compatible so we just uninstall it
# Refer to issue: https://github.com/MAGICS-LAB/DNABERT_2/issues/123
python3 -m pip uninstall triton

# Update transformer package to the most recent transformer
python3 -m pip install upgrade transformers

# Test to see if the conda environment works properly
python3 check_dnabert-2_install.py
```

### 1.2. To compare and evaluate the models performance, we will be evaluating the data on GUE

Please download the GUE dataset from [here](https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2/view?usp=sharing). 

We will need to do some pre-processing on the GUE inputs for DNABERT which expects the data to be organized as kmers and JSON files (the current inputs from the GUE download are organized as CSV). 

Here is an example to generate a 3mer JSON for all of the CSV inputs. Replace 3 with whichever k-mer you want (e.g. 6).
```
for dir in $(ls -d GUE/); do
  for subdir in $(ls -d $dir/*); do
    for csv in $(ls $subdir/*.csv); do
      python $scripts/generate_kmer_inputs.py $csv 3;
    done;
  done;
done
```
Then run the scripts to evaluate on all of the tasks.
The following scripts are adapted from [DNABERT-2 Finetune Step](https://github.com/MAGICS-LAB/DNABERT_2?tab=readme-ov-file#6-finetune) but adapted to the current LSF infrastructure that is current being used at WashU. Additionally there are some modifications for the train.py script to match the GUE dataset
```
# Finetune ALL models on the GUE dataset
LSF_DOCKER_PRESERVE_ENVIRONMENT=false \
PATH="/opt/conda/bin:/opt/coda/bin:$PATH" \
bsub \
-q subscription \
-G compute-yeli-t2 \
-sla yeli_t2 \
-n 8 \
-M 32 \
-R "select[gpuhost] span[hosts=1] rusage[mem=32G]" \
-gpu "num=1:gmodel=TeslaV100_SXM2_32GB" \
-o out_dnabert1_finetune.log \
-a "docker(kboltonlab/dnabert-2:1.0)" \
/bin/bash conda_submit_models_finetune.sh
```

### 1.3. Compare Results
In an interactive session activate the conda environment dnabert-2. You will probably have to install additional packages like pandas and matplotlib, I can't actually remember which ones you will need but install whichever packages are missing and then run
```
python3 scripts/summarise_results.py /path/to/output/directories/ finetune
```
[The result will look like this](https://github.com/IrenaeusChan/LiLabRotation/blob/ee95ef96ef3dfa47c0ab62fd46f243061482b6ef/pdf/models_matthews_correlation_plot.pdf)
