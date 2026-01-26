
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ToroidalAttentionLayer(nn.Module):
    """Attention with toroidal (wrap-around) connectivity"""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.toroidal_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, attention_mask=None):
        # Apply toroidal projection (wraps information around ends)
        x_toroidal = self.toroidal_projection(x)
        
        # Create toroidal attention mask (wraps around sequence)
        if attention_mask is None:
            seq_len = x.size(0)
            toroidal_mask = self._create_toroidal_mask(seq_len).to(x.device)
        else:
            toroidal_mask = attention_mask
        
        # Apply attention with toroidal connectivity
        attn_output, _ = self.attention(x_toroidal, x_toroidal, x_toroidal, 
                                        attn_mask=toroidal_mask)
        return attn_output
    
    def _create_toroidal_mask(self, seq_len):
        """Create attention mask that wraps around like a torus"""
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            # Connect to neighbors (wraps around ends)
            mask[i, (i-1) % seq_len] = 1  # Previous
            mask[i, i] = 1  # Self
            mask[i, (i+1) % seq_len] = 1  # Next
        return mask

class FractalResidualConnection(nn.Module):
    """
    Implements a fractal skip connection that processes information 
    at self-similar scales.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x_prev, x_curr):
        combined = torch.cat([x_prev, x_curr], dim=-1)
        gate = self.scale_gate(combined)
        return x_prev * (1 - gate) + x_curr * gate

class QuantumStateEmbedding(nn.Module):
    """
    Simulates quantum state preparation by embedding inputs into 
    a complex Hilbert space (represented by doubled real dimensions).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(hidden_dim, hidden_dim * 2) # Real + Imaginary parts
    
    def forward(self, x):
        # Project to complex space (simulated)
        complex_rep = self.embedding(x)
        # Collapse back to real space using magnitude
        real, imag = torch.chunk(complex_rep, 2, dim=-1)
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        return magnitude

class QuantumToroidalCore(nn.Module):
    """Quantum-inspired neural architecture with toroidal topology"""
    
    def __init__(self, input_dim=784, hidden_dim=128, num_layers=6, num_heads=4):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Toroidal Attention Layers
        self.toroidal_layers = nn.ModuleList([
            ToroidalAttentionLayer(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Fractal Residual Connections (every 3 layers)
        self.fractal_connections = nn.ModuleList([
            FractalResidualConnection(hidden_dim)
            for _ in range(num_layers // 3)
        ])
        
        # Quantum State Embeddings
        self.quantum_embeddings = QuantumStateEmbedding(hidden_dim)
        
        # Output Head
        self.head = nn.Linear(hidden_dim, 10) # Default to 10 classes
        
    def forward(self, x, stage=None):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        x = self.input_proj(x)
        
        # Quantum State Preparation
        x = self.quantum_embeddings(x)
        
        # Add sequence dim for attention [seq_len, batch, dim]
        # For simple classification, treat batch as sequence (inter-sample attention) 
        # or just unsqueeze(0) for single step.
        x = x.unsqueeze(0) 
        
        # Toroidal Processing
        for i, layer in enumerate(self.toroidal_layers):
            x_orig = x
            
            # Main Toroidal Attention
            x = layer(x)
            
            # Apply Fractal Connection every 3 layers
            if i % 3 == 2:
                fractal_idx = i // 3
                # Resize check (in case)
                x = self.fractal_connections[fractal_idx](x_orig, x)
            
            # Stage-dependent modulation
            if stage is not None:
                x = self._modulate_by_stage(x, stage)
        
        x = x.squeeze(0)
        return self.head(x)
    
    def _modulate_by_stage(self, x, stage):
        """Modulate processing based on SAR stage (0-9)"""
        # Simple scalar modulation for demo
        stage_factor = 1.0 + (stage * 0.1)
        return x * stage_factor
