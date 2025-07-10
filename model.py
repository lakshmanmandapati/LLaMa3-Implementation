# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from config import Llama3Config # Import your configuration

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) layer, used in Llama models.
    It normalizes the input based on the root mean square, which is faster than LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable gain parameter initialized to ones
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate RMS for each row (last dimension)
        # x.pow(2).mean(-1, keepdim=True) is mean of squares
        # torch.rsqrt is 1 / sqrt()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Cast to float32 for normalization to avoid precision issues, then cast back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: str = "cpu"):
    """
    Precomputes the complex frequencies for Rotary Position Embeddings (RoPE).
    RoPE injects positional information by rotating query and key vectors.
    """
    # Inverse frequencies: (1 / (theta^(2i/dim))) for i=0, 2, ..., dim-2
    # Shape: (dim // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)).to(device)
    
    # Position indices: 0, 1, ..., end-1
    # Shape: (end,)
    t = torch.arange(end, device=device)
    
    # Outer product to get (end, dim // 2) matrix of frequencies
    # freqs[j] = 1 / (theta^(2j/dim))
    # t[i] = i
    # torch.outer(t, freqs)[i,j] = t[i] * freqs[j] = i / (theta^(2j/dim))
    freqs_outer = torch.outer(t, freqs).float()
    
    # Create complex numbers: e^(i * freq_val) = cos(freq_val) + i * sin(freq_val)
    # torch.polar(magnitude, angle) -> magnitude * (cos(angle) + i * sin(angle))
    # Here magnitude is 1.0
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer) # Complex tensor (end, dim // 2)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequencies for broadcasting during RoPE application.
    Makes freqs_cis compatible with (bsz, n_heads, seqlen, head_dim) or (bsz, seqlen, n_heads, head_dim)
    by adding singleton dimensions.
    """
    ndim = x.ndim # Example: 4 (bsz, n_heads, seqlen, head_dim)
    assert 0 <= 1 < ndim # Ensure seqlen is at dim 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) # freqs_cis is (seqlen, head_dim // 2)
    
    # Example: if x is (bsz, seqlen, n_heads, head_dim)
    # We want freqs_cis to be (1, seqlen, 1, head_dim // 2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Applies Rotary Position Embeddings to query (xq) and key (xk) tensors.
    """
    # View query and key as complex numbers for efficient rotation
    # Reshape (..., head_dim) to (..., head_dim // 2, 2)
    # then view as complex (..., head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape frequencies for broadcasting with xq_ and xk_
    # freqs_cis is (seqlen, head_dim // 2). We want (1, seqlen, 1, head_dim // 2)
    # for application on (bsz, n_heads, seqlen, head_dim // 2)
    freqs_cis_reshaped = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
    
    # Perform complex multiplication for rotation
    # (bsz, n_heads, seqlen, head_dim // 2) * (1, seqlen, 1, head_dim // 2)
    xq_rotated = xq_ * freqs_cis_reshaped
    xk_rotated = xk_ * freqs_cis_reshaped
    
    # Convert back to real numbers and flatten the last two dimensions
    # From (..., head_dim // 2) complex to (..., head_dim // 2, 2) real, then flatten to (..., head_dim)
    xq_out = torch.view_as_real(xq_rotated).flatten(3)
    xk_out = torch.view_as_real(xk_rotated).flatten(3)
    
    # Ensure output type matches input type
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads # Number of query heads
        self.n_kv_heads = config.n_kv_heads # Number of key/value heads (for GQA)
        self.head_dim = config.dim // config.n_heads # Dimension of each head
        
        # How many query heads share the same Key/Value head in GQA
        self.n_rep = self.n_heads // self.n_kv_heads 

        # Linear layers for Query, Key, Value, and Output projection
        # Input dimension: config.dim (embedding size)
        # Output dimension: n_heads * head_dim (for Q), n_kv_heads * head_dim (for K, V)
        # Output dimension for Wo: n_heads * head_dim (combined attention output)
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        # KV Cache: Stores past Key and Value states to avoid recomputing them
        # Shape: (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        # These will be initialized/resized dynamically during the first forward pass.
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,       # Input tensor (bsz, seqlen, dim)
        start_pos: int,        # Current starting position for KV cache (number of tokens already in cache)
        freqs_cis: torch.Tensor, # RoPE frequencies for current sequence (seqlen, head_dim // 2)
        mask: Optional[torch.Tensor], # Causal mask (bsz, 1, seqlen, total_len_with_history)
    ):
        bsz, seqlen, dim = x.shape

        # Project input x to Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # Reshape Q, K, V for multi-head attention
        # (bsz, seqlen, num_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        # (bsz, seqlen, num_kv_heads, head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply Rotary Position Embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Transpose to (bsz, num_heads, seqlen, head_dim) for batch matrix multiplication
        xq = xq.transpose(1, 2) 
        xk = xk.transpose(1, 2) 
        xv = xv.transpose(1, 2) 

        # Initialize or resize KV cache for the current batch size
        # This allows for dynamic batching up to max_seq_len
        if self.cache_k is None or self.cache_k.shape[0] < bsz:
             self.cache_k = torch.empty(
                (bsz, self.config.max_seq_len, self.n_kv_heads, self.head_dim),
                device=x.device, dtype=xk.dtype # Ensure cache is on the correct device and dtype
            )
             self.cache_v = torch.empty(
                (bsz, self.config.max_seq_len, self.n_kv_heads, self.head_dim),
                device=x.device, dtype=xv.dtype
            )

        # Update cache for the current sequence
        # Store current K and V vectors at their respective positions (start_pos to start_pos + seqlen)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        
        # Retrieve full key/value history from cache
        # From start of sequence up to current end (start_pos + seqlen)
        keys = self.cache_k[:bsz, : start_pos + seqlen] # (bsz, total_len_with_history, n_kv_heads, head_dim)
        values = self.cache_v[:bsz, : start_pos + seqlen] # (bsz, total_len_with_history, n_kv_heads, head_dim)

        # Grouped Query Attention (GQA): Repeat K and V heads for each query head
        # if n_kv_heads < n_heads. This is an optimization for inference.
        # Original: (bsz, history_len, n_kv_heads, head_dim)
        # Repeat `n_rep` times along n_kv_heads dimension (dim=2)
        # Transpose to (bsz, n_heads, history_len, head_dim) for matmul
        keys = keys.repeat_interleave(self.n_rep, dim=2).transpose(1, 2) 
        values = values.repeat_interleave(self.n_rep, dim=2).transpose(1, 2)

        # Calculate attention scores: Query @ Key_T
        # (bsz, n_heads, seqlen, head_dim) @ (bsz, n_heads, head_dim, history_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Apply causal mask: for tokens in current sequence, mask out future tokens
            # by setting their scores to -infinity before softmax.
            scores = scores + mask 

        scores = F.softmax(scores.float(), dim=-1).type_as(x) # Softmax over attention scores
        
        # Apply scores to Values: scores @ Value
        # (bsz, n_heads, seqlen, history_len) @ (bsz, n_heads, history_len, head_dim)
        output = torch.matmul(scores, values) 
        
        # Concatenate and reshape back to (bsz, seqlen, dim)
        # Transpose n_heads and seqlen, then flatten last two dimensions
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1) 

        return self.wo(output) # Final output projection


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) with SwiGLU activation.
    This expands the dimension, applies activation, and then projects back.
    """
    def __init__(self, config: Llama3Config):
        super().__init__()
        # Calculate hidden dimension as per Llama 3 specification (SwiGLU)
        # It's usually 2 * dim, rounded up to nearest multiple_of
        hidden_dim = int(config.dim * config.ffn_dim_multiplier)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        # Linear layers for the SwiGLU activation pattern
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False) # Maps input to the "gate" for SiLU
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False) # Maps activated output back to original dim
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False) # Maps input to the "linear" part of SiLU

    def forward(self, x):
        # SwiGLU: w2( SiLU(x @ w1) * (x @ w3) )
        # F.silu(self.w1(x)) is the SiLU (Sigmoid Linear Unit) activation on one branch
        # self.w3(x) is the other branch, which acts as a gate
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    A single Transformer block, consisting of Attention and Feed-Forward layers,
    each preceded by RMSNorm and followed by a residual connection.
    """
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps) # Pre-attention norm
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)      # Pre-FFN norm

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # Residual connection around attention
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        # Residual connection around FFN
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Llama3Model(nn.Module):
    """
    The main Llama 3 model architecture.
    Comprises token embeddings, a stack of Transformer blocks, a final RMSNorm,
    and a linear output layer to project to vocabulary logits.
    """
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.config = config
        
        # Token embeddings layer: maps token IDs to dense vectors
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Stack of Transformer blocks
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config))
            
        # Final normalization layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # Output layer: projects the final hidden state to vocabulary logits
        # Weight tying: In Llama, this often shares weights with tok_embeddings for efficiency
        # For this from-scratch, we keep it separate for clarity, but you can tie them if you wish.
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Precompute RoPE frequencies once, move to device
        self.freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads, # Dim for RoPE is head_dim
            self.config.max_seq_len * 2, # Generate frequencies for a length beyond max_seq_len to be safe
            theta=self.config.rope_theta,
            device="mps" if torch.backends.mps.is_available() else "cpu" # Use MPS for M2
        )

        # Apply basic weight initialization to all layers (will be overwritten by loaded weights)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization for the model's layers.
        """
        if isinstance(module, nn.Linear):
            # Kaiming uniform initialization is common for ReLU-like activations
            # Here it's used generally for linear layers.
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            # Normal distribution for embedding layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            # RMSNorm weights are typically initialized to 1
            nn.init.ones_(module.weight)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Forward pass through the Llama 3 model.
        Args:
            tokens (torch.Tensor): Input token IDs (bsz, seqlen).
            start_pos (int): The starting position in the sequence, used for KV cache indexing.
                             0 for the initial prompt, then increments for each generated token.
        Returns:
            torch.Tensor: Logits for the next token (bsz, seqlen, vocab_size).
        """
        _bsz, seqlen = tokens.shape
        # Get token embeddings
        h = self.tok_embeddings(tokens)

        # Select the relevant RoPE frequencies for the current sequence segment
        # freqs_cis is precomputed for max_seq_len, here we slice for current seqlen
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # Causal mask for attention:
        # Applies to the attention mechanism to prevent tokens from attending to future tokens.
        # For the initial prompt (seqlen > 1), we need a mask to ensure causality within the prompt itself.
        # For subsequent single-token generation (seqlen == 1), the KV cache implicitly handles causality,
        # so an explicit mask is often not needed, or it's a simple (1,1) mask.
        mask = None
        if seqlen > 1:
            # Create a mask that is negative infinity in the upper triangle (future tokens)
            # The mask needs to consider the entire history length (start_pos + seqlen)
            # Shape: (1, 1, seqlen, total_history_len) for broadcasting
            mask = torch.full(
                (1, 1, seqlen, start_pos + seqlen), # Query length is `seqlen`, Key/Value length is `start_pos + seqlen`
                float("-inf"), # Set future tokens to -inf
                device=tokens.device,
                dtype=h.dtype # Match input dtype for consistency
            )
            # Use torch.triu to set values in the upper triangle (relative to `diagonal`) to -inf
            # `diagonal=start_pos + 1` shifts the effective diagonal based on past tokens in cache
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h) 
            
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h) # Final normalization
        output = self.output(h).float() # Project to vocabulary logits, ensure float32 for softmax
        return output