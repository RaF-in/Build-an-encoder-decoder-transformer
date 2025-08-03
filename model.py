import torch
import math
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class Config: 
    n_embd: int = 768
    block_size: int = 64
    batch_size: int = 64
    vocab_size: int = 50304
    no_of_head: int = 6
    no_of_layers: int = 12
    dropout: float = 0.2
    max_steps: int = 100


class TokenEmbeddings(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.n_embd)
    def forward(self, x): 
        return self.embedding(x) * math.sqrt(self.config.n_embd)


class PositionalEmbeddings(nn.Module): 
    def __init__(self, config, device): 
        super().__init__()
        self.config = config
        pe = torch.zeros(self.config.block_size, self.config.n_embd, dtype=torch.float, device=device)
        pos = torch.arange(0, self.config.block_size, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.config.n_embd, 2, device=device).float() * (-math.log(10000)/self.config.n_embd))
        pe[:, 0::2] = torch.sin(pos * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(pos * div_term) # cos(position * (10000 ** (2i / d_model))
        pe = pe.unsqueeze(0) # 1, block_size, n_embd
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(self.config.dropout)
    def forward(self, x): 
        return self.dropout(x + (self.pe[:, :x.size(1), :]).requires_grad_(False))
    
class LayerNormalization(nn.Module): 
    def __init__(self, config, eps=1e-6): 
        self.config = config
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(self.config.n_embd))
        self.beta = nn.Parameter(torch.zeros(self.config.n_embd))
        self.eps = eps
    def forward(self, x): 
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.beta

class MultiHeadAttention(nn.Module): 
    def __init__(self, config, is_casual=True):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(self.config.n_embd, self.config.n_embd * 3)
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)
        self.casual = is_casual
        if is_casual: 
            self.register_buffer('bias', torch.tril(torch.ones(self.config.block_size, self.config.block_size).view(1, 1, self.config.block_size, self.config.block_size)))
        else: 
            self.bias = None
    def forward(self, x, encoder_output = None): 
        B, T, C = x.shape
        if encoder_output is not None:
            k, v = encoder_output, encoder_output
            _, T_kv, _ = k.shape  # instead of reusing T
        else:
            T_kv = T

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embd, dim=-1) # B, T, C
        if encoder_output != None: 
            k, v = encoder_output, encoder_output

        q = q.view(B, T, self.config.no_of_head, C// self.config.no_of_head).transpose(1, 2) # B, nh, T, C 
        k = k.view(B, T_kv, self.config.no_of_head, C// self.config.no_of_head).transpose(1, 2)
        v = v.view(B, T_kv, self.config.no_of_head, C// self.config.no_of_head).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1))) # B, nh, T, T
        if self.casual:
            wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # B, nh, T, C
        return self.c_proj(out.transpose(1, 2).contiguous().view(B, T, C))
    
    
class Mlp(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(self.config.n_embd, self.config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(self.config.n_embd * 4, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)
    def forward(self, x): 
        return self.c_proj(self.dropout(self.gelu(self.c_fc(x))))
    
class EncoderBlock(nn.Module): 
    def __init__(self, config): 
        self.config = config
        super().__init__()
        self.ln1 = LayerNormalization(self.config)
        self.c_attn = MultiHeadAttention(self.config, is_casual=False)
        self.ln2 = LayerNormalization(self.config)
        self.mlp = Mlp(self.config)
    def forward(self, x): 
        x = x + self.c_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class DecoderBlock(nn.Module): 
    def __init__(self, config): 
        self.config = config
        super().__init__()
        self.ln1 = LayerNormalization(self.config)
        self.c_attn = MultiHeadAttention(self.config)
        self.ln2 = LayerNormalization(self.config)
        self.mlp = Mlp(self.config)
        self.cross_attn = MultiHeadAttention(self.config, False)
        self.ln3 = LayerNormalization(self.config)
    def forward(self, x, encoder_output): 
        x = x + self.c_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), encoder_output)
        x = x + self.mlp(self.ln3(x))
        return x
    
class Model(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transformer = nn.ModuleDict(dict(
            wte = TokenEmbeddings(self.config), 
            wpe = PositionalEmbeddings(self.config, self.device), 
            encoders = nn.ModuleList([EncoderBlock(self.config) for _ in range(self.config.no_of_layers)]), 
            decoders = nn.ModuleList([DecoderBlock(self.config) for _ in range(self.config.no_of_layers)])
        ))
        self.ln_f = LayerNormalization(self.config)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

        # Weights sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
    
    def forward(self, encoder_input, decoder_input, targets=None): 
        B, T = encoder_input.shape
        encoder_input = self.transformer.wte(encoder_input)
        encoder_input = self.transformer.wpe(encoder_input)

        decoder_input = self.transformer.wte(decoder_input)
        decoder_input = self.transformer.wpe(decoder_input)

        encoder_output = None
        for i in range(len(self.transformer.encoders)): 
            encoder_input = self.transformer.encoders[i](encoder_input)

        encoder_output = encoder_input.clone()
        for i in range(len(self.transformer.decoders)): 
            decoder_input = self.transformer.decoders[i](decoder_input, encoder_output)
        decoder_input = self.ln_f(decoder_input)
        logits = self.lm_head(decoder_input)
        loss = None
        if targets != None: 
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(-1))
        return logits, loss