from dataclasses import dataclass
@dataclass
class Config: 
    n_embd: int = 768
    block_size: int = 32
    batch_size: int = 64
    vocab_size: int = 50304
    no_of_head: int = 6
    no_of_layers: int = 12
    dropout: float = 0.1
    max_steps: int = 500
    pad_token_id: int = 0
    num_of_experts: int = 8
    num_of_shared_experts: int = 2
    activated_experts: int = 6
    expert_dim: int = 192