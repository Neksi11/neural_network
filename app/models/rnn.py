from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from .base_model import BaseNeuralNetwork

class RecurrentNetwork(BaseNeuralNetwork):
    """Implementation of a Recurrent Neural Network with support for RNN and LSTM cells."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RNN.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - input_size (int): Size of input features
                - hidden_size (int): Size of hidden state
                - num_layers (int): Number of recurrent layers
                - output_size (int): Size of output
                - cell_type (str): Type of RNN cell ('rnn', 'lstm', 'gru')
                - bidirectional (bool): Whether to use bidirectional RNN
                - dropout_rate (float): Dropout rate between layers
        """
        self.rnn_cells = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }
        super().__init__(config)
    
    def _build_network(self) -> None:
        """Build the RNN architecture."""
        input_size = self.config['input_size']
        hidden_size = self.config['hidden_size']
        num_layers = self.config.get('num_layers', 1)
        dropout_rate = self.config.get('dropout_rate', 0.0)
        cell_type = self.config.get('cell_type', 'lstm').lower()
        bidirectional = self.config.get('bidirectional', False)
        output_size = self.config['output_size']
        
        # Validate cell type
        if cell_type not in self.rnn_cells:
            raise ValueError(f"Unsupported RNN cell type: {cell_type}")
        
        # Create RNN layer
        rnn_class = self.rnn_cells[cell_type]
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output features for the linear layer
        output_features = hidden_size * 2 if bidirectional else hidden_size
        
        # Add layers to ModuleList for compatibility with base class
        self.layers.append(self.rnn)
        if dropout_rate > 0:
            self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(output_features, output_size))
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
               ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            hidden (Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]): 
                Initial hidden state
            
        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
                - Output tensor of shape (batch_size, seq_length, output_size)
                - Final hidden state
        """
        # RNN forward pass
        rnn_out, hidden_state = self.rnn(x, hidden)
        
        # Apply dropout if specified
        if len(self.layers) > 2:  # Has dropout
            rnn_out = self.layers[1](rnn_out)
        
        # Apply final linear layer
        output = self.layers[-1](rnn_out)
        
        return output, hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device
                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Initialize hidden state.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensor on
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Initial hidden state
        """
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        num_directions = 2 if self.rnn.bidirectional else 1
        
        if isinstance(self.rnn, nn.LSTM):
            return (torch.zeros(num_layers * num_directions, batch_size, hidden_size).to(device),
                    torch.zeros(num_layers * num_directions, batch_size, hidden_size).to(device))
        else:
            return torch.zeros(num_layers * num_directions, batch_size, hidden_size).to(device)
    
    @classmethod
    def create(cls,
               input_size: int,
               hidden_size: int,
               output_size: int,
               cell_type: str = 'lstm',
               num_layers: int = 1,
               bidirectional: bool = False,
               dropout_rate: float = 0.0) -> 'RecurrentNetwork':
        """
        Factory method to create an RNN with specified architecture.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            output_size (int): Size of output
            cell_type (str): Type of RNN cell ('rnn', 'lstm', 'gru')
            num_layers (int): Number of recurrent layers
            bidirectional (bool): Whether to use bidirectional RNN
            dropout_rate (float): Dropout rate between layers
            
        Returns:
            RecurrentNetwork: Configured network instance
        """
        config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'cell_type': cell_type,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'dropout_rate': dropout_rate
        }
        return cls(config)
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """
        Get information about network layers.
        
        Returns:
            List[Dict[str, Any]]: List of layer information dictionaries
        """
        layer_info = []
        
        # RNN layer info
        rnn_info = {
            'type': self.config.get('cell_type', 'lstm'),
            'input_size': self.rnn.input_size,
            'hidden_size': self.rnn.hidden_size,
            'num_layers': self.rnn.num_layers,
            'bidirectional': self.rnn.bidirectional,
            'dropout': self.rnn.dropout
        }
        layer_info.append(rnn_info)
        
        # Add info for remaining layers
        for layer in self.layers[1:]:
            if isinstance(layer, nn.Dropout):
                layer_info.append({
                    'type': 'dropout',
                    'rate': layer.p
                })
            elif isinstance(layer, nn.Linear):
                layer_info.append({
                    'type': 'linear',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features
                })
        
        return layer_info 