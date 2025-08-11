import torch
import math
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
# from Mixture_of_experts import MOE
from config import Config

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
    def forward(self, x, start_pos=0): 
        pos_emb = self.pe[:, start_pos:start_pos + x.size(1), :].requires_grad_(False)
        return self.dropout(x + pos_emb)

    
class LayerNormalization(nn.Module): 
    def __init__(self, config, eps=1e-6): 
        super().__init__()
        self.config = config
        self.alpha = nn.Parameter(torch.ones(self.config.n_embd))
        self.beta = nn.Parameter(torch.zeros(self.config.n_embd))
        self.eps = eps

    def forward(self, x): 
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False) 
        return self.alpha * ((x - mean) / (std + self.eps)) + self.beta


class MultiHeadAttention(nn.Module): 
    def __init__(self, config, is_casual=True):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(self.config.n_embd, self.config.n_embd * 3)
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.c_proj.INIT_FLAG=1
        self.dropout = nn.Dropout(self.config.dropout)
        self.casual = is_casual
        if is_casual: 
            self.register_buffer('bias', torch.tril(torch.ones(self.config.block_size, self.config.block_size).view(1, 1, self.config.block_size, self.config.block_size)))
        else: 
            self.bias = None
    def forward(self, x, encoder_output = None, src_mask=None, tgt_mask=None, kv_cache=None): 
        B, T, C = x.shape
        mask = None
        if encoder_output is not None:
            k, v = encoder_output, encoder_output
            _, T_kv, _ = k.shape  # instead of reusing T
        else:
            T_kv = T

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embd, dim=-1) # B, T, C
        if encoder_output is not None: 
            k, v = encoder_output, encoder_output

        # Handle KV cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache  # kv_cache should be tuple (k_cache, v_cache)
            if k_cache is not None and v_cache is not None:
                k = torch.cat([k_cache, k], dim=1)
                v = torch.cat([v_cache, v], dim=1)
                T_kv = k.size(1)
                

        # Update cache
        new_kv_cache = (k, v) 

        q = q.view(B, T, self.config.no_of_head, C// self.config.no_of_head).transpose(1, 2) # B, nh, T, C 
        k = k.view(B, T_kv, self.config.no_of_head, C// self.config.no_of_head).transpose(1, 2)
        v = v.view(B, T_kv, self.config.no_of_head, C// self.config.no_of_head).transpose(1, 2)
        wei = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1))) # B, nh, T, T
        if self.casual:
            past_len = kv_cache[0].size(1) if kv_cache and kv_cache[0] is not None else 0
            wei = wei.masked_fill(self.bias[:, :, past_len:past_len+T, :T_kv] == 0, float('-inf'))
            mask = tgt_mask.unsqueeze(1).unsqueeze(1)
            mask = mask.expand(B, 1, T, T_kv)
        elif src_mask is not None: 
            mask = src_mask.unsqueeze(1).unsqueeze(1)

        if mask is not None:
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
        self.c_proj.INIT_FLAG=1
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
        # self.mlp = Mlp(self.config)
        self.mlp = Mlp(config)
    def forward(self, x, src_mask): 
        x = x + self.c_attn(self.ln1(x), src_mask=src_mask, kv_cache=None)[0]
        # x = x + self.mlp(self.ln2(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class DecoderBlock(nn.Module): 
    def __init__(self, config): 
        self.config = config
        super().__init__()
        self.ln1 = LayerNormalization(self.config)
        self.c_attn = MultiHeadAttention(self.config)
        self.ln2 = LayerNormalization(self.config)
        # self.mlp = Mlp(self.config)
        self.mlp = Mlp(config)
        self.cross_attn = MultiHeadAttention(self.config, False)
        self.ln3 = LayerNormalization(self.config)
    def forward(self, x, encoder_output, src_mask, tgt_mask, kv_cache=None): 
        xx, new_kv_cache = self.c_attn(self.ln1(x), src_mask=src_mask, tgt_mask=tgt_mask, kv_cache=kv_cache)
        x = xx + x
        x = x + self.cross_attn(self.ln2(x), encoder_output, src_mask=src_mask, kv_cache=None)[0]
        # x = x + self.mlp(self.ln3(x))
        x = x + self.mlp(self.ln3(x))
        return x, new_kv_cache
    

class MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(config.n_embd * 2, config.n_embd) for _ in range(config.no_head_in_mtp)])
        self.transformer = nn.ModuleList([EncoderBlock(config) for _ in range(config.no_head_in_mtp)])
        self.unembd = nn.Linear(config.n_embd, config.vocab_size)
        self.token_embedding = TokenEmbeddings(config)
        self.config = config

    def rmsnorm(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True)) + 1e-8
        return x / rms

    def forward(self, hidden, input_tokens=None, inference_mode=False):

        # For inference, use simple single token prediction
        if inference_mode:
            # Use the simple linear head for single token prediction
            return self.unembd(hidden)  # [batch, seq_len, vocab_size]
        
        batch_size = hidden.size(0)
        
        mtp_upper = hidden.shape[1] - self.config.no_head_in_mtp
        
        # Pre-allocate output tensor to avoid dynamic memory allocation
        all_logits = torch.empty(
            batch_size, mtp_upper, self.config.no_head_in_mtp, self.config.vocab_size,
            dtype=hidden.dtype, device=hidden.device
        )
        for i in range(mtp_upper):
            h_prev = hidden[:, i, :].clone()
            for k in range(self.config.no_head_in_mtp):
                future_token = i + k + 1
                if input_tokens is not None:
                    # Use the actual future token embeddings
                    future_emb = self.token_embedding(input_tokens[:, future_token])
                else:
                    # Fallback to using the input embeddings
                    future_emb = hidden[:, future_token, :]
                h_prev = self.rmsnorm(h_prev)
                future_emb = self.rmsnorm(future_emb)
                concatenated_res = torch.cat([future_emb, h_prev], dim=-1)
                curr_h = self.projections[k](concatenated_res)

                # Transformer forward pass with error handling
                transformer_input = curr_h.unsqueeze(1)
                curr_res = self.transformer[k](transformer_input, None)
                
                # Safe reshape with validation
                expected_size = batch_size * curr_res.size(-1)
                if curr_res.numel() != expected_size:
                    curr_res = curr_res.contiguous().view(batch_size, -1)
                else:
                    curr_res = curr_res.view(batch_size, -1)
                
                # Generate logits with gradient checkpointing to save memory
                logits = self.unembd(curr_res)
                
                # Bounds checking before storing
                if (i < all_logits.size(1) and k < all_logits.size(2) and 
                    logits.size(0) == all_logits.size(0) and 
                    logits.size(1) == all_logits.size(3)):
                    all_logits[:, i, k, :] = logits
                else:
                    raise IndexError(f"Tensor size mismatch at position [{i}, {k}]")
                
                # Update h_prev and clean up intermediate tensors
                h_prev = curr_h.detach()  # Detach to prevent gradient accumulation

                # Clear intermediate tensors to free memory
                del concatenated_res, curr_h, curr_res, logits

        return all_logits
    
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
        self.lm_head = MTP(config)
        self.lm_head.unembd.weight = self.transformer.wte.embedding.weight
        self.lm_head.token_embedding.embedding.weight = self.transformer.wte.embedding.weight


        # Weights sharing scheme
        # self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.weight_init)

    def weight_init(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'INIT_FLAG'):
                std *= (2 * self.config.no_of_layers) ** -0.5
            torch.nn.init.normal_(module.weight, std=std, mean=0.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=std, mean=0.0)
        
    
    def forward(self, encoder_input, decoder_input, targets=None, kv_cache=None, inference=False, encoder_output_previous=None): 
        B, T = encoder_input.shape
        raw_decoder_input = decoder_input.clone()
        src_mask, tgt_mask = create_mask(encoder_input), create_mask(decoder_input)
        encoder_input = self.transformer.wte(encoder_input)
        encoder_input = self.transformer.wpe(encoder_input)

        if inference and kv_cache is None: 
            kv_cache = {
                "decoder": [None] * len(self.transformer.decoders)
            }

        if inference:
            start_pos = kv_cache['decoder'][0][0].shape[1] if kv_cache['decoder'][0] else 0
            decoder_input = self.transformer.wte(decoder_input)
            decoder_input = self.transformer.wpe(decoder_input, start_pos=start_pos)
        else:
            decoder_input = self.transformer.wte(decoder_input)
            decoder_input = self.transformer.wpe(decoder_input)


        if encoder_output_previous is not None: 
            encoder_output = encoder_output_previous
        else:
            for i in range(len(self.transformer.encoders)): 
                encoder_input = self.transformer.encoders[i](encoder_input, src_mask=src_mask)
            encoder_output = encoder_input.clone()
        
        for i, layer in enumerate(self.transformer.decoders):
            layer_cache = kv_cache["decoder"][i] if inference else None
            decoder_input, new_cache = layer(decoder_input, encoder_output, src_mask, tgt_mask, kv_cache=layer_cache)
            if inference:
                kv_cache["decoder"][i] = new_cache
        
        decoder_input = self.ln_f(decoder_input)
        logits = self.lm_head(decoder_input, raw_decoder_input, inference)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss, encoder_output, kv_cache