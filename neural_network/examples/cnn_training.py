import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torchvision.datasets as datasets
import sys
import os
from typing import Tuple

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.cnn import ConvolutionalNetwork
from app.training.trainer import ModelTrainer
from app.data.augmentation import ImageAugmentation

def create_cnn_model(input_channels: int = 3,
                    input_size: Tuple[int, int] = (224, 224)) -> ConvolutionalNetwork:
    """Create a CNN model for image classification."""
    # Define CNN architecture
    conv_configs = [
        {
            'filters': 32,
            'kernel_size': 3,
            'padding': 1,
            'pool_size': 2,
            'batch_norm': True
        },
        {
            'filters': 64,
            'kernel_size': 3,
            'padding': 1,
            'pool_size': 2,
            'batch_norm': True
        },
        {
            'filters': 128,
            'kernel_size': 3,
            'padding': 1,
            'pool_size': 2,
            'batch_norm': True
        }
    ]
    
    # Calculate output size after convolutions and pooling
    output_size = (input_size[0] // 8, input_size[1] // 8)  # After 3 pooling layers
    flattened_size = 128 * output_size[0] * output_size[1]
    
    # Define fully connected layers
    fc_sizes = [512, 10]  # 10 classes for example
    
    return ConvolutionalNetwork.create(
        input_channels=input_channels,
        input_height=input_size[0],
        input_width=input_size[1],
        conv_configs=conv_configs,
        fc_sizes=fc_sizes,
        activation='relu',
        dropout_rate=0.3
    )

class AugmentedDataset(Dataset):
    """Dataset wrapper that applies augmentation."""
    
    def __init__(self, dataset: Dataset, augmentation: ImageAugmentation,
                 is_training: bool = True):
        self.dataset = dataset
        self.augmentation = augmentation
        self.is_training = is_training
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        augmented_image = self.augmentation.augment(image, self.is_training)
        return augmented_image, label

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    input_size = (224, 224)
    batch_size = 32
    num_epochs = 10
    
    # Create data augmentation pipeline
    augmentation = ImageAugmentation.create(
        image_size=input_size,
        augmentation_types=['rotate', 'flip', 'color', 'noise'],
        probability=0.5,
        intensity=0.5
    )
    
    print("Loading CIFAR-10 dataset...")
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True
    )
    
    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True
    )
    
    # Wrap datasets with augmentation
    train_dataset = AugmentedDataset(train_dataset, augmentation, is_training=True)
    val_dataset = AugmentedDataset(val_dataset, augmentation, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print("\nCreating CNN model...")
    model = create_cnn_model(input_channels=3, input_size=input_size)
    
    # Print model architecture
    print("\nModel architecture:")
    for layer_info in model.get_layer_info():
        print(layer_info)
    
    # Training configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
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
    def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        _, predicted = torch.max(y_pred, 1)
        return (predicted == y_true).float().mean()
    
    metrics = {
        'accuracy': accuracy
    }
    
    # Train the model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        metrics=metrics,
        num_epochs=num_epochs,
        save_path='checkpoints/cnn_model.pt'
    )
    
    # Make predictions
    print("\nMaking predictions...")
    model.eval()
    with torch.no_grad():
        # Get a batch of validation data
        images, labels = next(iter(val_loader))
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        print("\nPredictions for first few images:")
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
        for i in range(5):
            print(f"True: {classes[labels[i]]} | Predicted: {classes[predicted[i]]}")

if __name__ == '__main__':
    main() 