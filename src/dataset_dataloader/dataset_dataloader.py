import torchvision
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from torch.utils.data import random_split

def dataset_dataloader(data_root,val_split=0.1):
    transform = transforms.Compose(
        [transforms.ToTensor(),])

    train_dataset = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    train_size = len(train_dataset)
    val_size = int(val_split * train_size)
    train_size = train_size - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)
    return(train_loader, test_loader,val_loader)