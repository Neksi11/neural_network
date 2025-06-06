from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

class ImageAugmentation:
    """Handles image augmentation using both torchvision and albumentations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the augmentation pipeline.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - image_size (Tuple[int, int]): Target image size
                - augmentation_types (List[str]): List of augmentations to apply
                - probability (float): Probability of applying each augmentation
                - intensity (float): Intensity of augmentations (0-1)
        """
        self.config = config
        self.image_size = config['image_size']
        self.probability = config.get('probability', 0.5)
        self.intensity = config.get('intensity', 0.5)
        
        self._setup_transforms()
    
    def _setup_transforms(self) -> None:
        """Set up the transformation pipeline."""
        aug_types = self.config.get('augmentation_types', [])
        
        # Basic transforms that are always applied
        self.basic_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Albumentations pipeline for training
        aug_list = [
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
        ]
        
        # Add requested augmentations
        for aug_type in aug_types:
            if aug_type == 'rotate':
                aug_list.append(A.Rotate(limit=45 * self.intensity, p=self.probability))
            elif aug_type == 'flip':
                aug_list.extend([
                    A.HorizontalFlip(p=self.probability),
                    A.VerticalFlip(p=self.probability)
                ])
            elif aug_type == 'color':
                aug_list.extend([
                    A.ColorJitter(
                        brightness=0.2 * self.intensity,
                        contrast=0.2 * self.intensity,
                        saturation=0.2 * self.intensity,
                        hue=0.1 * self.intensity,
                        p=self.probability
                    ),
                    A.RandomBrightnessContrast(p=self.probability)
                ])
            elif aug_type == 'noise':
                aug_list.extend([
                    A.GaussNoise(var_limit=(10.0 * self.intensity, 50.0 * self.intensity),
                               p=self.probability),
                    A.ISONoise(p=self.probability)
                ])
            elif aug_type == 'blur':
                aug_list.extend([
                    A.GaussianBlur(blur_limit=(3, 7), p=self.probability),
                    A.MotionBlur(blur_limit=(3, 7), p=self.probability)
                ])
            elif aug_type == 'distortion':
                aug_list.extend([
                    A.OpticalDistortion(distort_limit=0.1 * self.intensity,
                                      shift_limit=0.1 * self.intensity,
                                      p=self.probability),
                    A.GridDistortion(distort_limit=0.1 * self.intensity, p=self.probability)
                ])
        
        # Add normalization and conversion to tensor
        aug_list.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.train_transform = A.Compose(aug_list)
    
    def augment(self, image: Union[torch.Tensor, np.ndarray, Image.Image],
                is_training: bool = True) -> torch.Tensor:
        """
        Apply augmentations to an image.
        
        Args:
            image: Input image (torch.Tensor, numpy array, or PIL Image)
            is_training (bool): Whether to apply training augmentations
            
        Returns:
            torch.Tensor: Augmented image
        """
        # Convert input to the right format
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply transforms
        if is_training:
            augmented = self.train_transform(image=image)
            return augmented['image']
        else:
            return self.basic_transform(Image.fromarray(image))
    
    @classmethod
    def create(cls,
               image_size: Tuple[int, int],
               augmentation_types: Optional[List[str]] = None,
               probability: float = 0.5,
               intensity: float = 0.5) -> 'ImageAugmentation':
        """
        Factory method to create an augmentation pipeline.
        
        Args:
            image_size (Tuple[int, int]): Target image size (height, width)
            augmentation_types (Optional[List[str]]): Types of augmentations to apply
            probability (float): Probability of applying each augmentation
            intensity (float): Intensity of augmentations (0-1)
            
        Returns:
            ImageAugmentation: Configured augmentation pipeline
        """
        config = {
            'image_size': image_size,
            'augmentation_types': augmentation_types or [],
            'probability': probability,
            'intensity': intensity
        }
        return cls(config)
    
    def get_augmentation_info(self) -> Dict[str, Any]:
        """
        Get information about the augmentation pipeline.
        
        Returns:
            Dict[str, Any]: Dictionary containing augmentation settings
        """
        return {
            'image_size': self.image_size,
            'augmentation_types': self.config.get('augmentation_types', []),
            'probability': self.probability,
            'intensity': self.intensity
        } 