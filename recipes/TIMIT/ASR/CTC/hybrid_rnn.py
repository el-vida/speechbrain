"""
Hybrid RNN module that combines different RNN and SNN architectures.

This module allows stacking different types of layers (LSTM, LiGRU, RNN, GRU, SNN)
in a single model for comparison studies.

Authors
 * Elija Vida 2026
"""

import torch
import torch.nn as nn

import speechbrain as sb

try:
    from sparse.snn import SpikingNeurons
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False


class SNNWrapper(nn.Module):
    """Wrapper for SpikingNeurons to match RNN interface.
    
    SpikingNeurons outputs only spikes without hidden state.
    This wrapper converts it to match the (output, hidden_state) interface
    expected by the HybridRNN.
    
    Arguments
    ---------
    input_size : int
        Size of input features.
    hidden_size : int
        Number of neurons (output size).
    **kwargs : dict
        Additional arguments passed to SpikingNeurons.
    """
    
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.snn = SpikingNeurons(
            input_size=input_size,
            **kwargs
        )
    
    def forward(self, x):
        """Forward pass through SNN layer.
        
        Arguments
        ---------
        x : torch.Tensor
            Input of shape (batch, time, features).
        
        Returns
        -------
        output : torch.Tensor
            Output of shape (batch, time, hidden_size).
        hidden : torch.Tensor
            Dummy hidden state for compatibility (zeros).
        """
        # SpikingNeurons expects (batch, time, features)
        output = self.snn(x)
        
        # Return output and dummy hidden state for RNN compatibility
        # Hidden state shape: (batch, hidden_size)
        batch_size = x.size(0)
        hidden = torch.zeros(batch_size, self.hidden_size, 
                            device=x.device, dtype=x.dtype)
        
        return output, hidden


class HybridRNN(nn.Module):
    """A hybrid RNN module that chains different RNN and SNN architectures.
    
    This module allows experimenting with combinations of different layer types
    (LSTM, LiGRU, RNN, GRU, SNN) across multiple layers.
    
    Arguments
    ---------
    input_size : int
        The size of the input features.
    hidden_size : int
        The number of hidden units in each RNN layer.
    num_layers : int
        The number of RNN/SNN layers (must match len(layer_types)).
    layer_types : list of str
        List of layer types for each layer. Options: 'lstm', 'ligru', 'rnn', 'gru', 'snn'.
        Example: ['lstm', 'ligru', 'snn', 'lstm']
    bidirectional : bool
        Whether each RNN layer should be bidirectional. (Not applicable to SNN)
    dropout : float
        Dropout rate between layers (0.0 to 1.0).
    snn_kwargs : dict
        Additional keyword arguments for SNN layers (tauu_lim, tauw_lim, etc).
    
    Example
    -------
    >>> import torch
    >>> model = HybridRNN(
    ...     input_size=256,
    ...     hidden_size=512,
    ...     num_layers=4,
    ...     layer_types=['lstm', 'ligru', 'snn', 'lstm'],
    ...     bidirectional=True,
    ...     dropout=0.15
    ... )
    >>> x = torch.randn(8, 100, 256)  # (batch, time, features)
    >>> output, _ = model(x)
    >>> output.shape
    torch.Size([8, 100, 1024])  # (batch, time, hidden_size*2 for bidirectional)
    """
    
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        layer_types=['lstm', 'ligru', 'snn', 'lstm'],
        bidirectional=True,
        dropout=0.15,
        snn_kwargs=None,
    ):
        super().__init__()
        
        if num_layers != len(layer_types):
            raise ValueError(
                f"num_layers ({num_layers}) must match len(layer_types) ({len(layer_types)})"
            )
        
        if 'snn' in [lt.lower() for lt in layer_types] and not SPARSE_AVAILABLE:
            raise ImportError(
                "SNN layer type requested but sparse module not available. "
                "Please install the sparse module."
            )
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_types = layer_types
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.snn_kwargs = snn_kwargs or {}
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i, layer_type in enumerate(layer_types):
            # Input size is hidden_size*2 (if bidirectional) for all layers except first
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size * (2 if bidirectional else 1)
            
            rnn_layer = self._build_layer(
                layer_type,
                layer_input_size,
                hidden_size,
                bidirectional,
            )
            self.layers.append(rnn_layer)
            
            # Add dropout between layers (except after last layer)
            if i < num_layers - 1:
                self.dropouts.append(nn.Dropout(p=dropout))
    
    def _build_layer(self, layer_type, input_size, hidden_size, bidirectional):
        """Build a single layer of the specified type."""
        layer_type = layer_type.lower().strip()
        
        if layer_type == 'lstm':
            return sb.nnet.RNN.LSTM(
                hidden_size=hidden_size,
                bidirectional=bidirectional,
            )
        elif layer_type == 'ligru':
            return sb.nnet.RNN.LiGRU(
                hidden_size=hidden_size,
                bidirectional=bidirectional,
            )
        elif layer_type == 'rnn':
            return sb.nnet.RNN.RNN(
                hidden_size=hidden_size,
                bidirectional=bidirectional,
            )
        elif layer_type == 'gru':
            return sb.nnet.RNN.GRU(
                hidden_size=hidden_size,
                bidirectional=bidirectional,
            )
        elif layer_type == 'snn':
            if not SPARSE_AVAILABLE:
                raise ImportError("sparse module not available for SNN layer")
            return SNNWrapper(
                input_size=input_size,
                hidden_size=hidden_size,
                **self.snn_kwargs
            )
        else:
            raise ValueError(
                f"Unknown layer_type: {layer_type}. "
                f"Supported types: 'lstm', 'ligru', 'rnn', 'gru', 'snn'"
            )
    
    def forward(self, x, hx=None):
        """Forward pass through all hybrid layers.
        
        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape (batch, time, features).
        hx : torch.Tensor or None
            Initial hidden state (optional).
        
        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (batch, time, hidden_size*num_directions).
        hx : torch.Tensor
            Final hidden states from all layers.
        """
        output = x
        hidden_states = []
        
        for i, layer in enumerate(self.layers):
            output, h = layer(output)
            hidden_states.append(h)
            
            # Apply dropout between layers (except after last)
            if i < len(self.layers) - 1:
                output = self.dropouts[i](output)
        
        return output, hidden_states


if __name__ == "__main__":
    # Test the HybridRNN module
    batch_size = 8
    time_steps = 100
    input_features = 256
    hidden_size = 512
    
    # Create model with mixed RNN types
    model = HybridRNN(
        input_size=input_features,
        hidden_size=hidden_size,
        num_layers=4,
        layer_types=['lstm', 'ligru', 'rnn', 'lstm'],
        bidirectional=True,
        dropout=0.15,
    )
    
    # Create random input
    x = torch.randn(batch_size, time_steps, input_features)
    
    # Forward pass
    output, hidden_states = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of hidden states: {len(hidden_states)}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
