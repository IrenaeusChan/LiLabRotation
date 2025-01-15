#!/bin/bash

# Check if samtools is installed
if ! command -v samtools &> /dev/null; then
    echo "samtools could not be found. Please install samtools and ensure it is in your PATH."
    exit 1
fi

# Check if bwa is installed
if ! command -v bwa &> /dev/null; then
    echo "bwa could not be found. Please install bwa and ensure it is in your PATH."
    exit 1
fi

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_bam> <output_bam> <ref_fa>"
    exit 1
fi

in_bam=${1}
out_bam=${2}
reference=${3:-"/storage1/fs1/bga/Active/gmsroot/gc2560/core/model_data/2887491634/build21f22873ebe0486c8e6f69c15435aa96/all_sequences.fa"}

# Check if reference genome file exists
if [ ! -f "${reference}" ]; then
    echo "Reference genome file ${reference} does not exist."
    exit 1
fi

samtools collate -Oun128 "${in_bam}" |
    samtools fastq -OT RG,BC - |
    bwa mem -pt8 -C "${HG38_REF}" - |
    samtools sort -@4 -m4g -o "${out_bam}" -

# Check if the pipeline executed successfully
if [ $? -ne 0 ]; then
    echo "An error occurred during the execution of the pipeline."
    exit 1
fi

echo "Pipeline executed successfully. Output BAM file: ${out_bam}"
