from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import json
import os

class BaseNeuralNetwork(nn.Module):
    """Base class for all neural network models in the system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base neural network.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing network parameters
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self._build_network()
    
    def _build_network(self) -> None:
        """Build the network architecture based on configuration."""
        raise NotImplementedError("Subclasses must implement _build_network method")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def save_model(self, path: str, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the model state and configuration.
        
        Args:
            path (str): Path to save the model
            additional_info (Optional[Dict[str, Any]]): Additional information to save
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'model_state': self.state_dict(),
            'config': self.config
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        torch.save(save_dict, path)
        
        # Save human-readable config
        config_path = path + '.config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    @classmethod
    def load_model(cls, path: str) -> 'BaseNeuralNetwork':
        """
        Load a model from a saved state.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            BaseNeuralNetwork: Loaded model instance
        """
        save_dict = torch.load(path)
        model = cls(save_dict['config'])
        model.load_state_dict(save_dict['model_state'])
        return model
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get the number of parameters in the model.
        
        Returns:
            Dict[str, int]: Dictionary containing trainable and total parameter counts
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable': trainable_params,
            'total': total_params
        }
    
    def to_device(self, device: torch.device) -> 'BaseNeuralNetwork':
        """
        Move the model to specified device.
        
        Args:
            device (torch.device): Target device (CPU/GPU)
            
        Returns:
            BaseNeuralNetwork: Self reference for chaining
        """
        return self.to(device) 