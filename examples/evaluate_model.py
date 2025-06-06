import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from typing import Tuple, Dict
import sys
import os
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.rnn import RecurrentNetwork
from examples.rnn_training import generate_sequence_data, create_data_loaders

def load_model(model_path: str, config: Dict) -> RecurrentNetwork:
    """Load the trained model."""
    model = RecurrentNetwork.create(**config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model: RecurrentNetwork,
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the model and return predictions and actual values.
    """
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            hidden = model.init_hidden(sequences.size(0), device)
            predictions, _ = model(sequences, hidden)
            
            # Move predictions and targets to CPU and convert to numpy
            predictions = predictions.cpu().numpy()
            targets = targets.numpy()
            
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    return (np.concatenate(all_predictions),
            np.concatenate(all_targets))

def calculate_metrics(predictions: np.ndarray,
                     targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate various evaluation metrics.
    """
    # Reshape arrays to 2D
    pred_2d = predictions.reshape(-1, predictions.shape[-1])
    target_2d = targets.reshape(-1, targets.shape[-1])
    
    metrics = {
        'MSE': float(mean_squared_error(target_2d, pred_2d)),
        'RMSE': float(np.sqrt(mean_squared_error(target_2d, pred_2d))),
        'MAE': float(mean_absolute_error(target_2d, pred_2d)),
        'R2': float(r2_score(target_2d, pred_2d))
    }
    
    # Calculate additional metrics
    metrics['MAPE'] = float(np.mean(np.abs((target_2d - pred_2d) / (target_2d + 1e-8))) * 100)
    metrics['Correlation'] = float(np.corrcoef(target_2d.ravel(), pred_2d.ravel())[0, 1])
    
    return metrics

def plot_results(predictions: np.ndarray,
                targets: np.ndarray,
                metrics: Dict[str, float],
                save_dir: str):
    """
    Create and save various plots to visualize model performance.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(targets.ravel(), predictions.ravel(), alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    
    # Add metrics text
    metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_plot.png'))
    plt.close()
    
    # 2. Time series plot for a few sequences
    plt.figure(figsize=(15, 6))
    num_sequences = 3
    sequence_length = predictions.shape[1]
    
    for i in range(num_sequences):
        plt.subplot(num_sequences, 1, i + 1)
        plt.plot(range(sequence_length), targets[i, :, 0], 'b-', label='Actual', alpha=0.7)
        plt.plot(range(sequence_length), predictions[i, :, 0], 'r--', label='Predicted', alpha=0.7)
        plt.title(f'Sequence {i + 1}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series.png'))
    plt.close()
    
    # 3. Error distribution
    errors = (predictions - targets).ravel()
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
    plt.close()
    
    # Save metrics to a JSON file
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    # Configuration
    model_config = {
        'input_size': 5,
        'hidden_size': 64,
        'output_size': 1,
        'cell_type': 'lstm',
        'num_layers': 2,
        'bidirectional': True,
        'dropout_rate': 0.2
    }
    
    # Generate test data
    print("Generating test data...")
    sequences, targets = generate_sequence_data(
        num_sequences=200,  # Smaller test set
        seq_length=20,
        feature_size=model_config['input_size']
    )
    
    # Create test data loader directly
    test_dataset = torch.utils.data.TensorDataset(sequences, targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'checkpoints/rnn_model.pt'
    print(f"Loading model from {model_path}")
    model = load_model(model_path, model_config)
    model = model.to(device)
    
    # Evaluate model
    print("Evaluating model...")
    predictions, targets = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(predictions, targets)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_results(predictions, targets, metrics, save_dir='evaluation_results')
    print("\nResults saved in 'evaluation_results' directory")

if __name__ == '__main__':
    main() 