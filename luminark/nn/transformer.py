"""
LUMINARK - NanoGPT-Style Transformer
Character-level language model with toroidal attention and SAP stage modulation

Based on Andrej Karpathy's nanoGPT architecture with LUMINARK enhancements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ToroidalAttentionLayer(nn.Module):
    """
    Toroidal attention mechanism - connects sequence endpoints
    
    Creates circular attention pattern where:
    - Each token attends to itself
    - Previous token (with wraparound)
    - Next token (with wraparound)
    """
    
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: Optional attention mask
            
        Returns:
            (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Create toroidal attention mask
        if mask is None:
            mask = torch.zeros(seq_len, seq_len, device=x.device)
            for i in range(seq_len):
                # Attend to: previous (wraparound), self, next (wraparound)
                mask[i, (i - 1) % seq_len] = 1
                mask[i, i] = 1
                mask[i, (i + 1) % seq_len] = 1
            
            # Convert to attention mask format (0 = attend, -inf = ignore)
            mask = torch.where(mask == 0, float('-inf'), 0.0)
        
        # Project input
        x_proj = self.proj(x)
        
        # Apply multi-head attention with toroidal mask
        attn_out, _ = self.attn(x_proj, x_proj, x_proj, attn_mask=mask)
        
        return self.dropout(attn_out)


class TransformerBlock(nn.Module):
    """
    Standard transformer block with:
    - Toroidal attention
    - Layer normalization
    - Feed-forward network
    - Residual connections
    """
    
    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = ToroidalAttentionLayer(dim, heads, dropout)
        self.ln1 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, sap_stage: int = 4) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            sap_stage: Current SAP stage (0-9) for modulation
            
        Returns:
            (batch, seq_len, dim)
        """
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        
        # SAP stage modulation
        if sap_stage <= 3:
            # Early stages: Amplify signal (exploration)
            x = x * 1.15
        elif sap_stage >= 7:
            # Late stages: Constrain signal (refinement)
            x = torch.tanh(x) * 0.85
        
        return x


class LuminarkTransformer(nn.Module):
    """
    Complete character-level transformer for text generation
    
    Features:
    - Toroidal attention mechanism
    - SAP stage-aware modulation
    - Positional embeddings
    - Multi-layer architecture
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim = dim
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        idx: torch.Tensor,
        sap_stage: int = 4,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx: (batch, seq_len) token indices
            sap_stage: Current SAP stage (0-9)
            targets: Optional (batch, seq_len) for loss calculation
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Optional scalar loss if targets provided
        """
        batch_size, seq_len = idx.shape
        
        # Ensure sequence length doesn't exceed block size
        if seq_len > self.block_size:
            idx = idx[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]
            seq_len = self.block_size
        
        # Token embeddings + positional embeddings
        tok_emb = self.tok_emb(idx)  # (batch, seq_len, dim)
        pos_emb = self.pos_emb[:, :seq_len, :]  # (1, seq_len, dim)
        x = self.dropout(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, sap_stage)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq_len, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        sap_stage: int = 4
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively
        
        Args:
            idx: (batch, seq_len) conditioning sequence
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            sap_stage: Current SAP stage for modulation
            
        Returns:
            (batch, seq_len + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop context to block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond, sap_stage)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_num_params(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("🤖 LUMINARK - NanoGPT Transformer Demo")
    print("="*70)
    
    # Create model
    vocab_size = 65  # Example: ASCII characters
    model = LuminarkTransformer(
        vocab_size=vocab_size,
        block_size=128,
        dim=256,
        num_layers=6,
        num_heads=8
    )
    
    print(f"\n📊 Model Statistics:")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Block Size: {model.block_size}")
    print(f"  Embedding Dimension: {model.dim}")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch_size = 4
    seq_len = 32
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"\n🔄 Testing forward pass...")
    logits, _ = model(idx, sap_stage=4)
    print(f"  Input shape: {idx.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Test generation
    print(f"\n✨ Testing generation...")
    prompt = torch.randint(0, vocab_size, (1, 10), device=device)
    generated = model.generate(prompt, max_new_tokens=50, temperature=0.8, sap_stage=6)
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    
    print("\n✅ Transformer operational!")
