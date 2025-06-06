import pytest
import torch
import numpy as np
from PIL import Image
from app.data.augmentation import ImageAugmentation

@pytest.fixture
def sample_image():
    # Create a sample RGB image
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(image)

@pytest.fixture
def sample_config():
    return {
        'image_size': (32, 32),
        'augmentation_types': ['rotate', 'flip', 'color', 'noise'],
        'probability': 0.5,
        'intensity': 0.5
    }

class TestImageAugmentation:
    def test_initialization(self, sample_config):
        augmentation = ImageAugmentation(sample_config)
        assert isinstance(augmentation, ImageAugmentation)
        assert augmentation.image_size == sample_config['image_size']
        assert augmentation.probability == sample_config['probability']
    
    def test_basic_transform(self, sample_config, sample_image):
        augmentation = ImageAugmentation(sample_config)
        transformed = augmentation.augment(sample_image, is_training=False)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape[0] == 3  # RGB channels
        assert transformed.shape[1:] == sample_config['image_size']
    
    def test_training_transform(self, sample_config, sample_image):
        augmentation = ImageAugmentation(sample_config)
        transformed = augmentation.augment(sample_image, is_training=True)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape[0] == 3  # RGB channels
        assert transformed.shape[1:] == sample_config['image_size']
    
    @pytest.mark.parametrize("input_type", ["pil", "numpy", "tensor"])
    def test_different_input_types(self, sample_config, sample_image, input_type):
        augmentation = ImageAugmentation(sample_config)
        
        if input_type == "pil":
            image = sample_image
        elif input_type == "numpy":
            image = np.array(sample_image)
        else:  # tensor
            image = torch.from_numpy(np.array(sample_image)).permute(2, 0, 1).float()
        
        transformed = augmentation.augment(image, is_training=True)
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape[0] == 3
        assert transformed.shape[1:] == sample_config['image_size']
    
    def test_augmentation_consistency(self, sample_config, sample_image):
        """Test that non-training transforms are consistent."""
        augmentation = ImageAugmentation(sample_config)
        
        transform1 = augmentation.augment(sample_image, is_training=False)
        transform2 = augmentation.augment(sample_image, is_training=False)
        
        assert torch.allclose(transform1, transform2)
    
    @pytest.mark.parametrize("aug_type", ["rotate", "flip", "color", "noise", "blur", "distortion"])
    def test_individual_augmentations(self, sample_image, aug_type):
        config = {
            'image_size': (32, 32),
            'augmentation_types': [aug_type],
            'probability': 1.0,  # Ensure augmentation is applied
            'intensity': 1.0
        }
        
        augmentation = ImageAugmentation(config)
        transformed = augmentation.augment(sample_image, is_training=True)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 32, 32)
    
    def test_empty_augmentation_list(self, sample_image):
        config = {
            'image_size': (32, 32),
            'augmentation_types': [],
            'probability': 0.5,
            'intensity': 0.5
        }
        
        augmentation = ImageAugmentation(config)
        transformed = augmentation.augment(sample_image, is_training=True)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 32, 32)
    
    def test_get_augmentation_info(self, sample_config):
        augmentation = ImageAugmentation(sample_config)
        info = augmentation.get_augmentation_info()
        
        assert info['image_size'] == sample_config['image_size']
        assert info['augmentation_types'] == sample_config['augmentation_types']
        assert info['probability'] == sample_config['probability']
        assert info['intensity'] == sample_config['intensity']
    
    @pytest.mark.parametrize("size", [(64, 64), (128, 128), (224, 224)])
    def test_different_image_sizes(self, sample_image, size):
        config = {
            'image_size': size,
            'augmentation_types': ['rotate', 'flip'],
            'probability': 0.5,
            'intensity': 0.5
        }
        
        augmentation = ImageAugmentation(config)
        transformed = augmentation.augment(sample_image, is_training=True)
        
        assert transformed.shape[1:] == size 