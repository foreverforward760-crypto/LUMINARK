"""
Advanced neural network layers with enhanced capabilities
"""
import numpy as np
from luminark.nn.module import Module, Parameter
from luminark.core.tensor import Tensor


class ToroidalAttention(Module):
    """
    Toroidal (wrap-around) attention mechanism
    Treats sequence as circular, allowing distant tokens to attend to each other
    Useful for periodic patterns and long-range dependencies
    """
    
    def __init__(self, hidden_dim, window_size=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        
        # Attention weights
        self.query = Parameter(np.random.randn(hidden_dim, hidden_dim) * 0.01)
        self.key = Parameter(np.random.randn(hidden_dim, hidden_dim) * 0.01)
        self.value = Parameter(np.random.randn(hidden_dim, hidden_dim) * 0.01)
        self.output_proj = Parameter(np.random.randn(hidden_dim, hidden_dim) * 0.01)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute Q, K, V
        Q = x @ self.query  # (batch, seq, hidden)
        K = x @ self.key
        V = x @ self.value
        
        # Toroidal attention: each position attends to window_size neighbors
        # wrapping around the sequence
        output_data = np.zeros_like(x.data)
        
        for i in range(seq_len):
            # Get indices of neighbors (with wraparound)
            neighbor_indices = []
            for offset in range(-self.window_size//2, self.window_size//2 + 1):
                idx = (i + offset) % seq_len  # Wraparound
                neighbor_indices.append(idx)
            
            # Attention scores
            q_i = Q.data[:, i:i+1, :]  # (batch, 1, hidden)
            k_neighbors = K.data[:, neighbor_indices, :]  # (batch, window, hidden)
            
            # Scaled dot-product attention
            scores = np.matmul(q_i, k_neighbors.transpose(0, 2, 1))  # (batch, 1, window)
            scores = scores / np.sqrt(hidden_dim)
            
            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)
            
            # Apply attention to values
            v_neighbors = V.data[:, neighbor_indices, :]  # (batch, window, hidden)
            attended = np.matmul(attention_weights, v_neighbors)  # (batch, 1, hidden)
            
            output_data[:, i, :] = attended[:, 0, :]
        
        # Output projection
        output = Tensor(output_data, requires_grad=x.requires_grad)
        result = output @ self.output_proj
        
        return result
    
    def __repr__(self):
        return f"ToroidalAttention(hidden_dim={self.hidden_dim}, window_size={self.window_size})"


class ResidualBlock(Module):
    """Residual connection with layer normalization"""
    
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self._modules['layer'] = layer
    
    def forward(self, x):
        """Apply layer and add residual connection"""
        layer_output = self.layer(x)
        
        # Simple residual connection
        result = Tensor(
            x.data + layer_output.data,
            requires_grad=x.requires_grad or layer_output.requires_grad,
            _children=(x, layer_output),
            _op='residual'
        )
        
        def _backward():
            if x.requires_grad:
                x.grad = x.grad + result.grad
            if layer_output.requires_grad:
                layer_output.grad = layer_output.grad + result.grad
        
        result._backward = _backward
        return result


class AttentionPooling(Module):
    """
    Attention-based pooling instead of simple averaging
    Learns which parts of the sequence are most important
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = Parameter(np.random.randn(hidden_dim, 1) * 0.01)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        returns: (batch_size, hidden_dim)
        """
        # Compute attention scores
        scores = x @ self.attention_weights  # (batch, seq, 1)
        
        # Softmax over sequence dimension
        exp_scores = np.exp(scores.data - np.max(scores.data, axis=1, keepdims=True))
        attention_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-10)
        
        # Weighted sum
        weighted_sum = np.sum(x.data * attention_weights, axis=1)  # (batch, hidden)
        
        result = Tensor(weighted_sum, requires_grad=x.requires_grad)
        return result


class GatedLinear(Module):
    """
    Linear layer with gating mechanism
    Learns which features to pass through
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Transform weights
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(np.random.uniform(-limit, limit, (in_features, out_features)))
        
        # Gate weights
        self.gate_weight = Parameter(np.random.uniform(-limit, limit, (in_features, out_features)))
        
        self.bias = Parameter(np.zeros(out_features))
        self.gate_bias = Parameter(np.zeros(out_features))
    
    def forward(self, x):
        # Linear transformation
        out = x @ self.weight
        out_biased = Tensor(
            out.data + self.bias.data,
            requires_grad=out.requires_grad or self.bias.requires_grad
        )
        
        # Gate
        gate = x @ self.gate_weight
        gate_biased = Tensor(
            gate.data + self.gate_bias.data,
            requires_grad=gate.requires_grad
        )
        
        # Sigmoid gate
        gate_sigmoid = 1 / (1 + np.exp(-gate_biased.data))
        
        # Gated output
        result = Tensor(
            out_biased.data * gate_sigmoid,
            requires_grad=x.requires_grad
        )
        
        return result
    
    def __repr__(self):
        return f"GatedLinear(in_features={self.in_features}, out_features={self.out_features})"
