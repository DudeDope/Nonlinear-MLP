import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_loaders(batch_size=128, num_workers=4):
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])
    train = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tfm)
    test = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )