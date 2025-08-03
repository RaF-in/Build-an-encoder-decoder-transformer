import numpy as np

# Load the prepared data
encoder_inputs = np.load("transformer_data/encoder_inputs.npy")
decoder_inputs = np.load("transformer_data/decoder_inputs.npy")
decoder_targets = np.load("transformer_data/decoder_targets.npy")

# Convert to tensors (PyTorch example)
import torch
enc_inputs = torch.tensor(encoder_inputs, dtype=torch.long)
dec_inputs = torch.tensor(decoder_inputs, dtype=torch.long)
dec_targets = torch.tensor(decoder_targets, dtype=torch.long)


print(dec_targets[57])
