import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import random_split

def dataset_dataloader(data_root,val_split=0.1):
    transform = transforms.Compose(
        [transforms.ToTensor(),])

    train_dataset = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    total_test_size = len(test_dataset)
    val_size = total_test_size // 2
    test_size = total_test_size - val_size

    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])


    train_loader = data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)
    return(train_loader, test_loader,val_loader)