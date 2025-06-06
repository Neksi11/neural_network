from typing import Dict, Any, List, Union
import torch
import torch.nn as nn
from .base_model import BaseNeuralNetwork

class FeedForwardNetwork(BaseNeuralNetwork):
    """Implementation of a feedforward neural network."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feedforward neural network.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - layer_sizes (List[int]): List of layer sizes including input and output
                - activation (str): Activation function name
                - dropout_rate (float): Dropout rate between layers
        """
        self.activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        super().__init__(config)
    
    def _build_network(self) -> None:
        """Build the feedforward network architecture."""
        layer_sizes = self.config['layer_sizes']
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.0)
        
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least input and output layers")
        
        activation_fn = self.activation_map.get(activation.lower())
        if activation_fn is None:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build layers
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add activation and dropout for all but the last layer
            if i < len(layer_sizes) - 2:
                self.layers.append(activation_fn)
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    @classmethod
    def create(cls,
               input_size: int,
               output_size: int,
               hidden_layers: List[int],
               activation: str = 'relu',
               dropout_rate: float = 0.0) -> 'FeedForwardNetwork':
        """
        Factory method to create a feedforward network with specified architecture.
        
        Args:
            input_size (int): Size of input layer
            output_size (int): Size of output layer
            hidden_layers (List[int]): List of hidden layer sizes
            activation (str): Activation function name
            dropout_rate (float): Dropout rate between layers
            
        Returns:
            FeedForwardNetwork: Configured network instance
        """
        layer_sizes = [input_size] + hidden_layers + [output_size]
        config = {
            'layer_sizes': layer_sizes,
            'activation': activation,
            'dropout_rate': dropout_rate
        }
        return cls(config)
    
    def get_layer_info(self) -> List[Dict[str, Union[str, int]]]:
        """
        Get information about network layers.
        
        Returns:
            List[Dict[str, Union[str, int]]]: List of layer information dictionaries
        """
        layer_info = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer_info.append({
                    'type': 'linear',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features
                })
            elif isinstance(layer, nn.Dropout):
                layer_info.append({
                    'type': 'dropout',
                    'rate': layer.p
                })
            else:
                layer_info.append({
                    'type': layer.__class__.__name__.lower()
                })
        return layer_info 