import pytest
import torch
import numpy as np
from app.models.rnn import RecurrentNetwork

@pytest.fixture
def sample_rnn_config():
    return {
        'input_size': 10,
        'hidden_size': 32,
        'output_size': 1,
        'num_layers': 2,
        'cell_type': 'lstm',
        'bidirectional': True,
        'dropout_rate': 0.1
    }

class TestRecurrentNetwork:
    @pytest.mark.parametrize("cell_type", ['rnn', 'lstm', 'gru'])
    def test_initialization(self, sample_rnn_config, cell_type):
        sample_rnn_config['cell_type'] = cell_type
        model = RecurrentNetwork(sample_rnn_config)
        assert isinstance(model, RecurrentNetwork)
        assert len(model.layers) > 0
        
        layer_info = model.get_layer_info()
        assert layer_info[0]['type'] == cell_type
    
    @pytest.mark.parametrize("batch_size,seq_length", [(5, 10), (10, 20)])
    def test_forward_pass(self, sample_rnn_config, batch_size, seq_length):
        model = RecurrentNetwork(sample_rnn_config)
        x = torch.randn(batch_size, seq_length, sample_rnn_config['input_size'])
        hidden = model.init_hidden(batch_size, torch.device('cpu'))
        
        output, final_hidden = model(x, hidden)
        
        # Check output shape
        assert output.shape == (batch_size, seq_length, sample_rnn_config['output_size'])
        
        # Check hidden state shape
        if sample_rnn_config['cell_type'] == 'lstm':
            assert isinstance(final_hidden, tuple)
            assert len(final_hidden) == 2
            h, c = final_hidden
            num_directions = 2 if sample_rnn_config['bidirectional'] else 1
            expected_hidden_shape = (
                sample_rnn_config['num_layers'] * num_directions,
                batch_size,
                sample_rnn_config['hidden_size']
            )
            assert h.shape == expected_hidden_shape
            assert c.shape == expected_hidden_shape
        else:
            assert isinstance(final_hidden, torch.Tensor)
            num_directions = 2 if sample_rnn_config['bidirectional'] else 1
            expected_hidden_shape = (
                sample_rnn_config['num_layers'] * num_directions,
                batch_size,
                sample_rnn_config['hidden_size']
            )
            assert final_hidden.shape == expected_hidden_shape
    
    def test_save_load(self, sample_rnn_config, tmp_path):
        model = RecurrentNetwork(sample_rnn_config)
        save_path = tmp_path / "rnn_model.pt"
        
        # Save model
        model.save_model(str(save_path))
        assert save_path.exists()
        
        # Load model
        loaded_model = RecurrentNetwork.load_model(str(save_path))
        assert isinstance(loaded_model, RecurrentNetwork)
        
        # Compare parameters
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)
    
    def test_bidirectional(self, sample_rnn_config):
        # Test with and without bidirectional
        for bidirectional in [True, False]:
            sample_rnn_config['bidirectional'] = bidirectional
            model = RecurrentNetwork(sample_rnn_config)
            
            # Check layer info
            layer_info = model.get_layer_info()
            assert layer_info[0]['bidirectional'] == bidirectional
            
            # Check output features of final linear layer
            final_layer = None
            for layer in reversed(model.layers):
                if isinstance(layer, torch.nn.Linear):
                    final_layer = layer
                    break
            
            assert final_layer is not None
            expected_in_features = sample_rnn_config['hidden_size'] * (2 if bidirectional else 1)
            assert final_layer.in_features == expected_in_features
    
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_multiple_layers(self, sample_rnn_config, num_layers):
        sample_rnn_config['num_layers'] = num_layers
        model = RecurrentNetwork(sample_rnn_config)
        
        # Check layer info
        layer_info = model.get_layer_info()
        assert layer_info[0]['num_layers'] == num_layers
        
        # Test forward pass
        batch_size, seq_length = 5, 10
        x = torch.randn(batch_size, seq_length, sample_rnn_config['input_size'])
        hidden = model.init_hidden(batch_size, torch.device('cpu'))
        
        output, final_hidden = model(x, hidden)
        
        # Check hidden state shape
        if sample_rnn_config['cell_type'] == 'lstm':
            h, c = final_hidden
            assert h.size(0) == num_layers * (2 if sample_rnn_config['bidirectional'] else 1)
        else:
            assert final_hidden.size(0) == num_layers * (2 if sample_rnn_config['bidirectional'] else 1)
    
    def test_dropout(self, sample_rnn_config):
        # Test with and without dropout
        for dropout_rate in [0.0, 0.5]:
            sample_rnn_config['dropout_rate'] = dropout_rate
            model = RecurrentNetwork(sample_rnn_config)
            
            # Count dropout layers
            dropout_layers = sum(1 for layer in model.layers if isinstance(layer, torch.nn.Dropout))
            
            if dropout_rate > 0:
                assert dropout_layers > 0
            else:
                assert dropout_layers == 0
    
    def test_invalid_cell_type(self, sample_rnn_config):
        sample_rnn_config['cell_type'] = 'invalid_type'
        with pytest.raises(ValueError, match="Unsupported RNN cell type"):
            RecurrentNetwork(sample_rnn_config) 