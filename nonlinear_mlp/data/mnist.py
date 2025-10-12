import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size=128, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )