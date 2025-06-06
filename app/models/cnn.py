from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
from .base_model import BaseNeuralNetwork

class ConvolutionalNetwork(BaseNeuralNetwork):
    """Implementation of a Convolutional Neural Network."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CNN.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - input_channels (int): Number of input channels
                - input_height (int): Input image height
                - input_width (int): Input image width
                - conv_layers (List[Dict]): List of conv layer configs
                - fc_layers (List[int]): List of fully connected layer sizes
                - activation (str): Activation function name
                - dropout_rate (float): Dropout rate between layers
        """
        self.activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU()
        }
        super().__init__(config)
    
    def _build_network(self) -> None:
        """Build the CNN architecture."""
        input_channels = self.config['input_channels']
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.0)
        
        # Get activation function
        activation_fn = self.activation_map.get(activation.lower())
        if activation_fn is None:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build convolutional layers
        current_channels = input_channels
        conv_layers = self.config['conv_layers']
        current_size = (self.config['input_height'], self.config['input_width'])
        
        for i, conv_config in enumerate(conv_layers):
            # Add convolutional layer
            out_channels = conv_config['filters']
            kernel_size = conv_config['kernel_size']
            stride = conv_config.get('stride', 1)
            padding = conv_config.get('padding', 0)
            
            self.layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size, 
                         stride=stride, padding=padding)
            )
            self.layers.append(activation_fn)
            
            # Add batch normalization if specified
            if conv_config.get('batch_norm', False):
                self.layers.append(nn.BatchNorm2d(out_channels))
            
            # Add pooling if specified
            pool_size = conv_config.get('pool_size')
            if pool_size:
                self.layers.append(nn.MaxPool2d(pool_size))
                current_size = (
                    current_size[0] // pool_size,
                    current_size[1] // pool_size
                )
            
            # Update current channels
            current_channels = out_channels
            
            # Add dropout if specified
            if dropout_rate > 0:
                self.layers.append(nn.Dropout2d(dropout_rate))
        
        # Calculate flattened size for fully connected layers
        self.flatten_size = current_channels * current_size[0] * current_size[1]
        
        # Add flatten layer
        self.layers.append(nn.Flatten())
        
        # Build fully connected layers
        current_size = self.flatten_size
        fc_layers = self.config['fc_layers']
        
        for i, fc_size in enumerate(fc_layers):
            self.layers.append(nn.Linear(current_size, fc_size))
            
            # Add activation and dropout for all but the last layer
            if i < len(fc_layers) - 1:
                self.layers.append(activation_fn)
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
            
            current_size = fc_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    @classmethod
    def create(cls,
               input_channels: int,
               input_height: int,
               input_width: int,
               conv_configs: List[Dict[str, Any]],
               fc_sizes: List[int],
               activation: str = 'relu',
               dropout_rate: float = 0.0) -> 'ConvolutionalNetwork':
        """
        Factory method to create a CNN with specified architecture.
        
        Args:
            input_channels (int): Number of input channels
            input_height (int): Input image height
            input_width (int): Input image width
            conv_configs (List[Dict[str, Any]]): Configurations for conv layers
            fc_sizes (List[int]): Sizes of fully connected layers
            activation (str): Activation function name
            dropout_rate (float): Dropout rate between layers
            
        Returns:
            ConvolutionalNetwork: Configured network instance
        """
        config = {
            'input_channels': input_channels,
            'input_height': input_height,
            'input_width': input_width,
            'conv_layers': conv_configs,
            'fc_layers': fc_sizes,
            'activation': activation,
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
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                layer_info.append({
                    'type': 'conv2d',
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding
                })
            elif isinstance(layer, nn.Linear):
                layer_info.append({
                    'type': 'linear',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features
                })
            elif isinstance(layer, (nn.Dropout, nn.Dropout2d)):
                layer_info.append({
                    'type': 'dropout',
                    'rate': layer.p
                })
            elif isinstance(layer, nn.BatchNorm2d):
                layer_info.append({
                    'type': 'batch_norm',
                    'num_features': layer.num_features
                })
            elif isinstance(layer, nn.MaxPool2d):
                layer_info.append({
                    'type': 'max_pool',
                    'kernel_size': layer.kernel_size
                })
            else:
                layer_info.append({
                    'type': layer.__class__.__name__.lower()
                })
        return layer_info 