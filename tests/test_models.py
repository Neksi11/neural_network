import pytest
import torch
import numpy as np
from app.models.feedforward import FeedForwardNetwork
from app.models.cnn import ConvolutionalNetwork

@pytest.fixture
def sample_feedforward_config():
    return {
        'layer_sizes': [10, 32, 16, 1],
        'activation': 'relu',
        'dropout_rate': 0.1
    }

@pytest.fixture
def sample_cnn_config():
    return {
        'input_channels': 3,
        'input_height': 32,
        'input_width': 32,
        'conv_layers': [
            {
                'filters': 16,
                'kernel_size': 3,
                'padding': 1,
                'pool_size': 2
            },
            {
                'filters': 32,
                'kernel_size': 3,
                'padding': 1,
                'pool_size': 2
            }
        ],
        'fc_layers': [128, 10],
        'activation': 'relu',
        'dropout_rate': 0.1
    }

class TestFeedForwardNetwork:
    def test_initialization(self, sample_feedforward_config):
        model = FeedForwardNetwork(sample_feedforward_config)
        assert isinstance(model, FeedForwardNetwork)
        assert len(model.layers) > 0
    
    def test_forward_pass(self, sample_feedforward_config):
        model = FeedForwardNetwork(sample_feedforward_config)
        batch_size = 5
        input_size = sample_feedforward_config['layer_sizes'][0]
        x = torch.randn(batch_size, input_size)
        
        output = model(x)
        assert output.shape == (batch_size, sample_feedforward_config['layer_sizes'][-1])
    
    def test_save_load(self, sample_feedforward_config, tmp_path):
        model = FeedForwardNetwork(sample_feedforward_config)
        save_path = tmp_path / "model.pt"
        
        # Save model
        model.save_model(str(save_path))
        assert save_path.exists()
        
        # Load model
        loaded_model = FeedForwardNetwork.load_model(str(save_path))
        assert isinstance(loaded_model, FeedForwardNetwork)
        
        # Compare parameters
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)
    
    def test_parameter_count(self, sample_feedforward_config):
        model = FeedForwardNetwork(sample_feedforward_config)
        param_count = model.get_parameter_count()
        
        assert 'trainable' in param_count
        assert 'total' in param_count
        assert param_count['total'] >= param_count['trainable']
        assert param_count['total'] > 0

class TestConvolutionalNetwork:
    def test_initialization(self, sample_cnn_config):
        model = ConvolutionalNetwork(sample_cnn_config)
        assert isinstance(model, ConvolutionalNetwork)
        assert len(model.layers) > 0
    
    def test_forward_pass(self, sample_cnn_config):
        model = ConvolutionalNetwork(sample_cnn_config)
        batch_size = 5
        x = torch.randn(batch_size,
                       sample_cnn_config['input_channels'],
                       sample_cnn_config['input_height'],
                       sample_cnn_config['input_width'])
        
        output = model(x)
        assert output.shape == (batch_size, sample_cnn_config['fc_layers'][-1])
    
    def test_save_load(self, sample_cnn_config, tmp_path):
        model = ConvolutionalNetwork(sample_cnn_config)
        save_path = tmp_path / "cnn_model.pt"
        
        # Save model
        model.save_model(str(save_path))
        assert save_path.exists()
        
        # Load model
        loaded_model = ConvolutionalNetwork.load_model(str(save_path))
        assert isinstance(loaded_model, ConvolutionalNetwork)
        
        # Compare parameters
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)
    
    def test_layer_info(self, sample_cnn_config):
        model = ConvolutionalNetwork(sample_cnn_config)
        layer_info = model.get_layer_info()
        
        assert len(layer_info) > 0
        assert any(info['type'] == 'conv2d' for info in layer_info)
        assert any(info['type'] == 'linear' for info in layer_info)
    
    @pytest.mark.parametrize("batch_norm", [True, False])
    def test_batch_norm_option(self, sample_cnn_config, batch_norm):
        sample_cnn_config['conv_layers'][0]['batch_norm'] = batch_norm
        model = ConvolutionalNetwork(sample_cnn_config)
        layer_info = model.get_layer_info()
        
        has_batch_norm = any(info['type'] == 'batch_norm' for info in layer_info)
        assert has_batch_norm == batch_norm

@pytest.mark.parametrize("model_class,config_fixture", [
    (FeedForwardNetwork, "sample_feedforward_config"),
    (ConvolutionalNetwork, "sample_cnn_config")
])
class TestCommonFunctionality:
    def test_device_movement(self, model_class, config_fixture, request):
        config = request.getfixturevalue(config_fixture)
        model = model_class(config)
        
        if torch.cuda.is_available():
            model = model.to_device(torch.device('cuda'))
            assert next(model.parameters()).is_cuda
        
        model = model.to_device(torch.device('cpu'))
        assert not next(model.parameters()).is_cuda
    
    def test_training_mode(self, model_class, config_fixture, request):
        config = request.getfixturevalue(config_fixture)
        model = model_class(config)
        
        model.train()
        assert model.training
        
        model.eval()
        assert not model.training 