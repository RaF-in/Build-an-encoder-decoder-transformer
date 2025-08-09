import torch
import numpy as np
from model import Config
from model import Model
import math
import inspect
import torch.nn as nn
from model import LayerNormalization
from datasets import Dataset

# Load the prepared data
# encoder_inputs = np.load("transformer_data/encoder_inputs.npy")
# decoder_inputs = np.load("transformer_data/decoder_inputs.npy")
# decoder_targets = np.load("transformer_data/decoder_targets.npy")
ds = Dataset.load_from_disk("processed/translation_dataset")

# Convert to tensors (PyTorch example)
enc_inputs = torch.stack([torch.tensor(x['encoder_input'], dtype=torch.long) for x in ds])
dec_inputs = torch.stack([torch.tensor(x['decoder_input'], dtype=torch.long) for x in ds])
dec_targets = torch.stack([torch.tensor(x['decoder_output'], dtype=torch.long) for x in ds])

print(f"lengths of encoder inputs = {enc_inputs.shape}, and decoder inputs = {dec_inputs.shape}, decoder outputs = {dec_targets.shape}")