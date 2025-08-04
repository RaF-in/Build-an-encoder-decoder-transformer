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
    pad_token_id: int = 0

def create_mask(x): 
    return x != Config().pad_token_id

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
    def forward(self, x, encoder_output = None, src_mask=None, tgt_mask=None, kv_cache=None, pos=None): 
        B, T, C = x.shape
        mask = None
        if encoder_output is not None:
            k, v = encoder_output, encoder_output
            _, T_kv, _ = k.shape  # instead of reusing T
        else:
            T_kv = T

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embd, dim=-1) # B, T, C
        if encoder_output != None: 
            k, v = encoder_output, encoder_output

        # Handle KV cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache  # kv_cache should be tuple (k_cache, v_cache)
            if k_cache is not None and v_cache is not None:
                k = torch.cat([k_cache, k], dim=1)
                v = torch.cat([v_cache, v], dim=1)
                T_kv = k.size(1)

        # Update cache
        new_kv_cache = (k, v) if kv_cache is not None else None

        q = q.view(B, T, self.config.no_of_head, C// self.config.no_of_head).transpose(1, 2) # B, nh, T, C 
        k = k.view(B, T_kv, self.config.no_of_head, C// self.config.no_of_head).transpose(1, 2)
        v = v.view(B, T_kv, self.config.no_of_head, C// self.config.no_of_head).transpose(1, 2)


        wei = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1))) # B, nh, T, T
        if self.casual:
            wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            mask = tgt_mask.unsqueeze(1).unsqueeze(1)
            mask_key = mask.unsqueeze(1).unsqueeze(1) # B, 1, 1, T
            mask_query = mask.unsqueeze(1).unsqueeze(-1) # B, 1, T, 1
            mask = mask_key * mask_query
        else: 
            mask = src_mask.unsqueeze(1).unsqueeze(1)

        wei = wei.masked_fill(mask==0, float("-inf"))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # B, nh, T, C
        return self.c_proj(out.transpose(1, 2).contiguous().view(B, T, C)), new_kv_cache
    
    
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
    def forward(self, x, mask): 
        x = x + self.c_attn(self.ln1(x), src_mask=mask, kv_cache=None)[0]
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
    def forward(self, x, encoder_output, src_mask, tgt_mask, kv_cache=None): 
        xx, new_kv_cache = self.c_attn(self.ln1(x), src_mask=src_mask, kv_cache=kv_cache)
        x = xx + x
        x = x + self.cross_attn(self.ln2(x), encoder_output, src_mask=src_mask, tgt_mask=tgt_mask, kv_cache=None)[0]
        x = x + self.mlp(self.ln3(x))
        return x, new_kv_cache
    
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
        
    
    def forward(self, encoder_input, decoder_input, targets=None, kv_cache=None, inference=False, encoder_output_previous=None): 
        B, T = encoder_input.shape
        src_mask, tgt_mask = create_mask(encoder_input), create_mask(decoder_input)
        encoder_input = self.transformer.wte(encoder_input)
        encoder_input = self.transformer.wpe(encoder_input)

        if inference and kv_cache is None: 
            kv_cache = {
                "decoder": [None] * len(self.transformer.decoders)
            }

        decoder_input = self.transformer.wte(decoder_input)
        decoder_input = self.transformer.wpe(decoder_input)

        if encoder_output_previous is not None: 
            encoder_output = encoder_output_previous
        else:
            for i in range(len(self.transformer.encoders)): 
                encoder_input= self.transformer.encoders[i](encoder_input, src_mask=src_mask)
            encoder_output = encoder_input.clone()

        for i, layer in enumerate(self.transformer.decoders):
            layer_cache = kv_cache["decoder"][i] if inference else None
            decoder_input, new_cache = layer(decoder_input, encoder_output, src_mask, tgt_mask, kv_cache=layer_cache)
            if inference:
                kv_cache["decoder"][i] = new_cache

        decoder_input = self.ln_f(decoder_input)
        logits = self.lm_head(decoder_input)
        loss = None
        if targets is not None:
            # Replace pad tokens with -100 in the target so they are ignored in loss
            targets_masked = targets.clone()
            targets_masked[targets_masked == self.config.pad_token_id] = -100

            loss = F.cross_entropy(logits.view(B * T, -1), targets_masked.view(-1), ignore_index=-100)
        return logits, loss, encoder_output, kv_cache