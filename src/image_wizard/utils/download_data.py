import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np

__all__ = [
    "get_cifar10_transforms",
    "get_cifar10_dataloaders",
    "get_cifar10_classes",
]


def get_cifar10_transforms(
    augment: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get CIFAR-10 transforms for training and validation."""

    # Basic transforms
    base_transform = [
        # https://docs.pytorch.org/vision/0.8/transforms.html#torchvision.transforms.ToTensor
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of
        # shape (C x H x W) in the range [0.0, 1.0]
        transforms.ToTensor(),
        # https://github.com/kuangliu/pytorch-cifar/issues/19#issue-268972488
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    # Training transforms with augmentation
    if augment:
        train_transform = transforms.Compose(
            [
                # https://docs.pytorch.org/vision/0.8/transforms.html#torchvision.transforms.RandomCrop
                transforms.RandomCrop(32, padding=4),
                # https://docs.pytorch.org/vision/0.8/transforms.html#torchvision.transforms.RandomHorizontalFlip
                transforms.RandomHorizontalFlip(),
                *base_transform,
            ]
        )
    else:
        train_transform = transforms.Compose(base_transform)

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose(base_transform)

    return train_transform, val_transform


def get_cifar10_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    subset_fraction: Optional[float] = None,
    subset_size: Optional[int] = None,
    augment: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and validation dataloaders."""

    train_transform, val_transform = get_cifar10_transforms(augment=augment)

    # Download/load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)

    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=val_transform)

    # Create subsets if specified
    if subset_size is not None:
        # Random subset for faster experimentation
        train_indices = np.random.choice(len(train_dataset), size=subset_size, replace=False)
        val_indices = np.random.choice(len(val_dataset), size=subset_size // 5, replace=False)

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    elif subset_fraction is not None:
        # Fractional subset
        train_size = int(len(train_dataset) * subset_fraction)
        val_size = int(len(val_dataset) * subset_fraction)

        train_indices = np.random.choice(len(train_dataset), size=train_size, replace=False)
        val_indices = np.random.choice(len(val_dataset), size=val_size, replace=False)

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def get_cifar10_classes():
    """Get CIFAR-10 class names."""
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
