import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet normalization
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_imagenet_loaders(
    root: str = "data/imagenet",
    batch_size: int = 256,
    num_workers: int = 8,
    image_size: int = 224,
    val_resize_size: int = 256,
    pin_memory: bool = True,
    train_dir_name: str = "train",
    val_dir_name: str = "val",
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build ImageNet loaders using torchvision.datasets.ImageFolder.
    Directory structure expected:
      root/
        train/
          classA/ img1.jpg ...
          classB/ ...
        val/
          classA/ ...
          classB/ ...
    """
    train_dir = os.path.join(root, train_dir_name)
    val_dir = os.path.join(root, val_dir_name)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(val_resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set   = datasets.ImageFolder(val_dir, transform=val_tf)

    num_classes = len(train_set.classes)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, num_classes
