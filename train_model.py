import torch
import numpy as np
from model import Config
from model import Model
import math
import inspect

# Load the prepared data
encoder_inputs = np.load("transformer_data/encoder_inputs.npy")
decoder_inputs = np.load("transformer_data/decoder_inputs.npy")
decoder_targets = np.load("transformer_data/decoder_targets.npy")

# Convert to tensors (PyTorch example)
enc_inputs = torch.tensor(encoder_inputs, dtype=torch.long)
dec_inputs = torch.tensor(decoder_inputs, dtype=torch.long)
dec_targets = torch.tensor(decoder_targets, dtype=torch.long)

def train_test_split(train_ratio: float, val_ratio: float): 
    total_len = len(enc_inputs)
    train_num = int(total_len * train_ratio)
    val_num = int(total_len * val_ratio)
    return enc_inputs[:train_num], dec_inputs[:train_num], dec_targets[:train_num], enc_inputs[train_num:train_num + val_num], dec_inputs[train_num:train_num + val_num], dec_targets[train_num:train_num + val_num]

enc_train, dec_in_train, dec_tar_train, enc_val, dec_in_val, dec_tar_val = train_test_split(0.95, 0.05)
# print("total amount of data ===== ")
# print("train data")
# print(len(enc_train), len(dec_in_train), len(dec_tar_train))
# print("val data")
# print(len(enc_val), len(dec_in_val), len(dec_tar_val))

class DataLoaderLite: 
    def __init__(self, enc_data, dec_data, tar_data, config): 
        self.enc_data = enc_data
        self.dec_data = dec_data
        self.tar_data = tar_data
        self.current_pos = 0
        self.config = config
    def reset(self): 
        self.current_pos = 0
    def next_batch(self): 
        enc_inputs = self.enc_data[self.current_pos: self.current_pos + self.config.batch_size]
        dec_inputs = self.dec_data[self.current_pos: self.current_pos + self.config.batch_size]
        dec_outputs = self.tar_data[self.current_pos: self.current_pos + self.config.batch_size]
        self.current_pos += self.config.batch_size
        if self.current_pos + self.config.batch_size > len(self.enc_data): 
            self.reset()
        return enc_inputs, dec_inputs, dec_outputs

config = Config()

train_loader, val_loader = DataLoaderLite(enc_train, dec_in_train, dec_tar_train, config), DataLoaderLite(enc_val, dec_in_val, dec_tar_val, config)
train_loader.reset()
val_loader.reset()

torch.set_float32_matmul_precision('high')
def configure_optimizer(model, lr, weight_decay):
    param_group = {pn:p for pn, p in model.named_parameters()}
    param_group = {pn: p for pn, p in param_group.items() if p.requires_grad}

    decay_params = [p for pn, p in param_group.items() if p.dim() >= 2]
    non_decay_params = [p for pn, p in param_group.items() if p.dim() < 2]

    optim_group = [
        {"params": decay_params, "weight_decay": weight_decay}, 
        {"params": non_decay_params, "weight_decay": 0.0}
    ]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_fused = "cuda" in device and fused_available
    optimizer = torch.optim.AdamW(optim_group, lr=lr, fused=is_fused)
    return optimizer

max_lr = 3e-4
min_lr = max_lr * 0.001
warmup_steps = 4
max_steps = config.max_steps
total_grad_steps = 1 << 17
weight_decay = 0.1
grad_accum_steps = total_grad_steps // (Config().block_size * Config().batch_size)

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

raw_model = Model(Config())
optimizer = configure_optimizer(raw_model, max_lr, 0.1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device={device}")
raw_model.to(device)

model = raw_model

def test_model(): 
    val_loader.reset()
    model.eval()
    avg_loss = 0
    total_test_steps = len(val_loader.enc_data) // config.batch_size
    print(f"config batch size = {config.batch_size}")
    print(f"total test steps = {total_grad_steps}")
    with torch.no_grad():
        for i in range(total_test_steps): 
            encoder_data, decoder_data, targets = val_loader.next_batch()
            encoder_data = encoder_data.to(device)
            decoder_data = decoder_data.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(encoder_data, decoder_data, targets)
            avg_loss += loss.detach()
    print(f"total val loss = {avg_loss/total_test_steps}")

def train_model(): 
    train_loader.reset()
    model.train()
    print(f"total grad accum steps = {grad_accum_steps}")
    for i in range(config.max_steps): 
        if i % 100 == 0: 
            test_model()
            model.train()
        optimizer.zero_grad()
        avg_loss = 0.0
        for step in range(grad_accum_steps): 
            encoder_data, decoder_data, targets = train_loader.next_batch()
            encoder_data = encoder_data.to(device)
            decoder_data = decoder_data.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(encoder_data, decoder_data, targets)
            loss = loss / grad_accum_steps
            avg_loss += loss.detach() 
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(i)
        for param in optimizer.param_groups: 
            param['lr'] = lr
        
        optimizer.step()
        print(f"total loss at step {i} = {avg_loss}")
        with open('log.txt', 'a+') as f:
            f.write('\n' + str(avg_loss.item()))

if __name__ == "__main__":
    train_model()
    test_model()
    torch.save({
        'model_state_dict': model.state_dict(), 
        'config': Config()
    }, 'trained_model2.pth')
            
            


