import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-100 normalization (standard)
_CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR100_STD  = (0.2673, 0.2564, 0.2761)

def get_cifar100_loaders(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
    pin_memory: bool = True,
    download: bool = True,
):
    """
    Returns train and test loaders for CIFAR-100.

    Transforms:
      - Train: RandomCrop(32, 4) + RandomHorizontalFlip + ToTensor + Normalize
      - Test:  ToTensor + Normalize
    """
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])

    train_set = datasets.CIFAR100(root=root, train=True, transform=train_tf, download=download)
    test_set  = datasets.CIFAR100(root=root, train=False, transform=test_tf, download=download)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, test_loader
