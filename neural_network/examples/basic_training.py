import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.feedforward import FeedForwardNetwork
from app.training.trainer import ModelTrainer

def generate_sample_data(num_samples: int = 1000):
    """Generate a simple dataset for demonstration."""
    # Generate random input data
    X = np.random.randn(num_samples, 5)
    
    # Generate target values (sum of inputs plus some noise)
    y = np.sum(X, axis=1) + np.random.randn(num_samples) * 0.1
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)
    
    return X, y

def create_data_loaders(X, y, batch_size=32, train_split=0.8):
    """Create training and validation data loaders."""
    # Split data into train and validation sets
    split_idx = int(len(X) * train_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(X, y)
    
    # Create model
    print("Creating neural network...")
    model = FeedForwardNetwork.create(
        input_size=5,
        output_size=1,
        hidden_layers=[32, 16],
        activation='relu',
        dropout_rate=0.1
    )
    
    # Print model architecture
    print("\nModel architecture:")
    for layer_info in model.get_layer_info():
        print(layer_info)
    
    # Training configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
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
        config=config
    )
    
    # Define metrics
    def r2_score(y_pred, y_true):
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
        num_epochs=20,
        save_path='checkpoints/model.pt'
    )
    
    # Make predictions
    print("\nMaking predictions...")
    model.eval()
    with torch.no_grad():
        X_test = torch.randn(5, 5)  # 5 random samples
        y_pred = model(X_test)
        print("\nTest predictions:")
        print("Input:")
        print(X_test)
        print("\nPredicted output:")
        print(y_pred)

if __name__ == '__main__':
    main() 