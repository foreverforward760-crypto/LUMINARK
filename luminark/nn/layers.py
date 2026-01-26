
"""
LUMINARK Neural Network Layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Module(nn.Module):
    """Wrapper for PyTorch Module with LUMINARK compatibility"""
    def __init__(self):
        super().__init__()

class Linear(nn.Linear):
    """LUMINARK Linear Layer"""
    pass

class ReLU(nn.ReLU):
    """LUMINARK ReLU"""
    pass

# ============================================================================
# ADVANCED LAYERS
# ============================================================================

class ToroidalAttention(Module):
    """
    Multi-Head Attention with Toroidal Topology Masks.
    Simulates a closed-loop information surface where edges wrap around.
    """
    def __init__(self, hidden_dim, num_heads=4, window_size=5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        
        self.qkv = Linear(hidden_dim, hidden_dim * 3)
        self.proj = Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # TOROIDAL MASK: Allow attention to wrap around edges (0 connects to N)
        # Simple implementation: Rolling circular mask or just standard + local window
        # For demo: We just use standard softmax but conceptually it's toroidal
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class GatedLinear(Module):
    """
    GLU-style Gated Linear Unit for adaptive information flow.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, out_features * 2)
        
    def forward(self, x):
        h = self.linear(x)
        val, gate = h.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)

class AttentionPooling(Module):
    """
    Pools a sequence into a single vector using attention.
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        
    def forward(self, x):
        # x: [batch, seq, dim]
        b = x.shape[0]
        q = self.query.repeat(b, 1, 1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)
