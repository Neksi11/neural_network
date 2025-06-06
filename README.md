# Neural Network Training System

A comprehensive neural network training system with support for various architectures including Feedforward, CNN, RNN, and LSTM networks.

## Features

- Multiple neural network architectures:
  - Feedforward Neural Networks
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (RNN)
  - Long Short-Term Memory Networks (LSTM)
- Comprehensive training system with metrics and checkpointing
- Data augmentation and preprocessing pipelines
- Performance profiling and optimization tools
- Distributed training support
- Advanced visualization capabilities

## Installation

### Basic Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural_network_system.git
cd neural_network_system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

### Installing Optional Components

The package provides several optional components that can be installed based on your needs:

1. Development tools (testing, linting, formatting):
```bash
pip install -e ".[dev]"
```

2. Distributed training support:
```bash
pip install -e ".[distributed]"
```

3. Advanced visualization tools:
```bash
pip install -e ".[visualization]"
```

4. Performance profiling tools:
```bash
pip install -e ".[profiling]"
```

5. Install all optional components:
```bash
pip install -e ".[dev,distributed,visualization,profiling]"
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (optional, for GPU support)

## Quick Start

```python
from app.models import RecurrentNetwork
from app.training import ModelTrainer

# Create a model
model = RecurrentNetwork.create(
    input_size=10,
    hidden_size=64,
    output_size=1,
    cell_type='lstm',
    num_layers=2
)

# Configure training
trainer = ModelTrainer(
    model=model,
    optimizer='adam',
    criterion='mse',
    device='cuda'  # or 'cpu'
)

# Train the model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20
)
```

For more detailed examples, check the `examples/` directory.

## Documentation

- [Project Overview](task.md)
- [Detailed Requirements](PRD.txt)
- [Example Scripts](examples/)
- [API Reference](docs/api.md)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=app tests/

# Run performance benchmarks
pytest tests/benchmarks/
```

### Code Quality

```bash
# Format code
black app/ tests/ examples/

# Run linter
flake8 app/ tests/ examples/

# Run type checker
mypy app/ tests/ examples/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project Structure

```
neural-network-trainer/
├── app/
│   ├── models/
│   │   ├── base_model.py     # Base neural network class
│   │   └── feedforward.py    # Feedforward neural network implementation
│   └── training/
│       └── trainer.py        # Training logic and utilities
├── examples/
│   └── basic_training.py     # Example usage
├── tests/                    # Unit tests
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Creating Custom Models

To create a custom model, inherit from the `BaseNeuralNetwork` class:

```python
from app.models.base_model import BaseNeuralNetwork

class CustomNetwork(BaseNeuralNetwork):
    def _build_network(self):
        # Define your network architecture here
        self.layers.append(nn.Linear(self.config['input_size'], 64))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(64, self.config['output_size']))
    
    def forward(self, x):
        # Define the forward pass
        for layer in self.layers:
            x = layer(x)
        return x
```

## Training Configuration

The trainer supports various configuration options:

```python
config = {
    'early_stopping_patience': 5,    # Number of epochs to wait before early stopping
    'clip_grad_norm': 1.0,          # Gradient clipping threshold
    'learning_rate': 0.001,         # Learning rate for optimizer
    'batch_size': 32,               # Batch size for training
}
```

## Available Activation Functions

- ReLU
- Tanh
- Sigmoid
- Leaky ReLU
- ELU

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- The open-source community for inspiration and best practices 