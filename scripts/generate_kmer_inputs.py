import os
import csv
import json
import sys
from typing import List

input_csv_file_path = sys.argv[1]
kmer = int(sys.argv[2])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(input_csv_file_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = input_csv_file_path.replace(".csv", f"_{k}mer.json")
    print(f"Generating k-mer...")
    kmer = [generate_kmer_str(text, k) for text in texts]
    with open(kmer_path, "w") as f:
        print(f"Saving k-mer to {kmer_path}...")
        json.dump(kmer, f)
        
    return kmer

with open(input_csv_file_path, "r") as f:
    data = list(csv.reader(f))[1:]
if len(data[0]) == 2:
    # data is in the format of [text, label]
    print("Perform single sequence classification...")
    texts = [d[0] for d in data]
    labels = [int(d[1]) for d in data]
elif len(data[0]) == 3:
    # data is in the format of [text1, text2, label]
    print("Perform sequence-pair classification...")
    texts = [[d[0], d[1]] for d in data]
    labels = [int(d[2]) for d in data]
else:
    raise ValueError("Data format not supported.")

texts = load_or_generate_kmer(input_csv_file_path, texts, kmer)