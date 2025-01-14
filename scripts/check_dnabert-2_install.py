import torch
import transformers
from transformers import AutoModel
from transformers import AutoTokenizer

print("Package transformers version (should be 4.29.2):", transformers.__version__) # Make sure for this code we are using 4.29.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("CPU or CUDA? --> ", device)
print("Num GPUs: ", n_gpu)

print("------------ Example Embedding Calculation ------------")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
print(f"DNA sequence is: {dna}")
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0]
embedding_mean = torch.mean(hidden_states[0], dim=0)
print("This should be 768 --> ", embedding_mean.shape)
embedding_max = torch.max(hidden_states[0], dim=0)[0]
print("This should be 768 --> ", embedding_max.shape)
