from typing import Dict, Any, Optional, Callable, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
import time
import json
import os

class ModelTrainer:
    """A trainer class to handle the training process of neural networks."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 config: Dict[str, Any] = None,
                 forward_fn: Optional[Callable] = None):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model to train
            optimizer: The optimizer to use
            criterion: The loss function
            device: The device to train on (CPU/GPU)
            config: Configuration dictionary
            forward_fn: Optional custom forward function
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config or {}
        self.forward_fn = forward_fn
        
        # Initialize best validation loss for early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader, metrics: Dict[str, Callable] = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        metric_values = {name: 0.0 for name in (metrics or {})}
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc='Training', leave=False) as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = tuple(b.to(self.device) for b in batch)
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if self.forward_fn is not None:
                    output, target = self.forward_fn(self.model, batch)
                else:
                    if isinstance(batch, (tuple, list)):
                        inputs, target = batch
                        output = self.model(inputs)
                    else:
                        output = self.model(batch)
                        target = batch
                
                # Calculate loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Calculate metrics
                if metrics:
                    with torch.no_grad():
                        for name, metric_fn in metrics.items():
                            metric_values[name] += metric_fn(output, target).item()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping if configured
                if self.config.get('clip_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['clip_grad_norm']
                    )
                
                self.optimizer.step()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average metrics
        metrics_dict = {'loss': total_loss / num_batches}
        if metrics:
            metrics_dict.update({
                name: value / num_batches
                for name, value in metric_values.items()
            })
        
        return metrics_dict
    
    def validate(self, val_loader: DataLoader, metrics: Dict[str, Callable] = None) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        metric_values = {name: 0.0 for name in (metrics or {})}
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = tuple(b.to(self.device) for b in batch)
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if self.forward_fn is not None:
                    output, target = self.forward_fn(self.model, batch)
                else:
                    if isinstance(batch, (tuple, list)):
                        inputs, target = batch
                        output = self.model(inputs)
                    else:
                        output = self.model(batch)
                        target = batch
                
                # Calculate loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Calculate metrics
                if metrics:
                    for name, metric_fn in metrics.items():
                        metric_values[name] += metric_fn(output, target).item()
        
        # Calculate average metrics
        metrics_dict = {'loss': total_loss / num_batches}
        if metrics:
            metrics_dict.update({
                name: value / num_batches
                for name, value in metric_values.items()
            })
        
        return metrics_dict
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              metrics: Dict[str, Callable] = None,
              num_epochs: int = 10,
              save_path: str = None) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            metrics: Dictionary of metric functions
            num_epochs: Number of epochs to train
            save_path: Path to save the best model
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Create directory for save_path if it doesn't exist
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, metrics)
            history['train_loss'].append(train_metrics['loss'])
            history['train_metrics'].append({
                k: v for k, v in train_metrics.items() if k != 'loss'
            })
            
            # Validate
            val_metrics = self.validate(val_loader, metrics)
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append({
                k: v for k, v in val_metrics.items() if k != 'loss'
            })
            
            # Print metrics
            metrics_str = f"Train Loss: {train_metrics['loss']:.4f}"
            metrics_str += f" - Val Loss: {val_metrics['loss']:.4f}"
            for name in (metrics or {}):
                metrics_str += f" - Train {name}: {train_metrics[name]:.4f}"
                metrics_str += f" - Val {name}: {val_metrics[name]:.4f}"
            print(metrics_str)
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saved best model to {save_path}")
            else:
                self.patience_counter += 1
                if (self.config.get('early_stopping_patience') and 
                    self.patience_counter >= self.config['early_stopping_patience']):
                    print("\nEarly stopping triggered")
                    break
        
        return history 