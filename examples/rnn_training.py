import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam
import numpy as np
import sys
import os
from typing import Tuple, List

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.rnn import RecurrentNetwork
from app.training.trainer import ModelTrainer

def generate_sequence_data(num_sequences: int = 1000,
                         seq_length: int = 20,
                         feature_size: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic sequence data where each element depends on previous elements.
    
    The target for each time step is a weighted sum of previous inputs plus noise.
    """
    # Generate random sequences
    sequences = np.random.randn(num_sequences, seq_length, feature_size)
    
    # Generate targets (weighted sum of previous 3 time steps)
    targets = np.zeros((num_sequences, seq_length, 1))
    weights = np.array([0.5, 0.3, 0.2])  # Weights for previous steps
    
    for i in range(num_sequences):
        for t in range(seq_length):
            # Sum over feature dimension
            seq_sum = np.sum(sequences[i, t], axis=-1)
            
            # For each time step, look back up to 3 steps
            for j in range(min(3, t + 1)):
                targets[i, t] += weights[j] * np.sum(sequences[i, t - j], axis=-1)
            
            # Add some noise
            targets[i, t] += np.random.randn() * 0.1
    
    return (torch.FloatTensor(sequences),
            torch.FloatTensor(targets))

def create_data_loaders(sequences: torch.Tensor,
                       targets: torch.Tensor,
                       batch_size: int = 32,
                       train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    # Split data
    split_idx = int(len(sequences) * train_split)
    train_seq, val_seq = sequences[:split_idx], sequences[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    # Create datasets
    train_dataset = TensorDataset(train_seq, train_targets)
    val_dataset = TensorDataset(val_seq, val_targets)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    feature_size = 5
    hidden_size = 64
    seq_length = 20
    batch_size = 32
    num_epochs = 20
    
    print("Generating sequence data...")
    sequences, targets = generate_sequence_data(
        num_sequences=1000,
        seq_length=seq_length,
        feature_size=feature_size
    )
    
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        sequences, targets, batch_size=batch_size
    )
    
    # Create model
    print("\nCreating RNN model...")
    model = RecurrentNetwork.create(
        input_size=feature_size,
        hidden_size=hidden_size,
        output_size=1,
        cell_type='lstm',  # Try 'rnn', 'lstm', or 'gru'
        num_layers=2,
        bidirectional=True,
        dropout_rate=0.2
    )
    
    # Print model architecture
    print("\nModel architecture:")
    for layer_info in model.get_layer_info():
        print(layer_info)
    
    # Training configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Custom forward function to handle RNN output
    def forward_fn(model, batch):
        inputs, targets = batch
        hidden = model.init_hidden(inputs.size(0), inputs.device)
        outputs, _ = model(inputs, hidden)
        return outputs, targets
    
    config = {
        'early_stopping_patience': 5,
        'clip_grad_norm': 1.0
    }
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        forward_fn=forward_fn  # Add custom forward function
    )
    
    # Define metrics
    def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        total_variance = torch.var(y_true) * len(y_true)
        residual_variance = torch.sum((y_true - y_pred) ** 2)
        return 1 - residual_variance / total_variance
    
    metrics = {
        'r2': r2_score
    }
    
    # Train the model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        metrics=metrics,
        num_epochs=num_epochs,
        save_path='checkpoints/rnn_model.pt'
    )
    
    # Make predictions
    print("\nMaking predictions...")
    model.eval()
    with torch.no_grad():
        # Initialize hidden state
        batch_seq, batch_targets = next(iter(val_loader))
        batch_size = batch_seq.shape[0]
        hidden = model.init_hidden(batch_size, device)
        
        # Move data to device
        batch_seq = batch_seq.to(device)
        predictions, _ = model(batch_seq, hidden)
        
        # Print some results
        print("\nPrediction examples:")
        for i in range(3):  # Show first 3 sequences
            print(f"\nSequence {i + 1}:")
            print(f"Target values: {batch_targets[i, -5:, 0].numpy()}")
            print(f"Predictions:   {predictions[i, -5:, 0].cpu().numpy()}")

if __name__ == '__main__':
    main() 