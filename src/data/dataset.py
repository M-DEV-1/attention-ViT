import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_caltech101_splits(data_dir, transform=None):
    """
    Downloads and randomly splits Caltech101 into train/val sets.
    Uses a fixed random seed to guarantee exactly the same split every time.
    """
    # 101 object categories + 1 background = 102 total
    full_dataset = datasets.Caltech101(
        root=data_dir, 
        download=True, 
        target_type='category',
        transform=transform
    )
    
    # 80/20 train/test split
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    classes = full_dataset.categories
    return train_dataset, val_dataset, classes

def get_dataloaders(data_dir, batch_size):
    """
    Prepares dataloaders for Caltech101.
    """
    # Standard ImageNet normalization and sizing
    # Explicitly cast to RGB as Caltech101 contains some grayscale images
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset, val_dataset, classes = get_caltech101_splits(data_dir, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, classes
