```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

# Expect directory structure:
# imagenet/
#   train/
#     class1/xxx.jpeg
#   val/
#     class1/yyy.jpeg
# Provide path via root argument.

def get_imagenet_loaders(
    root: str = "data/imagenet",
    batch_size: int = 256,
    num_workers: int = 8,
    resolution: int = 224,
    val_resize: int = 256,
    distributed: bool = False
) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        normalize
    ])
    train_ds = datasets.ImageFolder(root=f"{root}/train", transform=train_tfm)
    val_ds = datasets.ImageFolder(root=f"{root}/val", transform=val_tfm)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, 1000  # 1000 classes
```