from dataclasses import dataclass

@dataclass
class Llama3Config:
    dim: int = 4096  # embedding dimension
    n_layers: int = 32 # number of layers
    n_heads: int = 32  # number of attention heads
    n_kv_heads: int = 8 # number of key/value heads (for Grouped Query Attention)
    vocab_size: int = 128256 # Llama 3's tokenizer vocabulary size
    multiple_of: int = 1024 # for FeedForward layer dimension calculation
    ffn_dim_multiplier: float = 1.3 # multiplier for feed forward hidden dim (approximated)
    norm_eps: float = 1e-5 # epsilon for RMSNorm
    max_seq_len: int = 8192 # max sequence length (context window)
    rope_theta: float = 500000.0 # for RoPE (Llama 3 specific)

# Llama 3 8B specific configuration
LLAMA3_8B_CONFIG = Llama3Config()