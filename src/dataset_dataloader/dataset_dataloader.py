import torchvision
import torchvision.transforms as transforms
import torch
import torch.utils.data as data

def dataset_dataloader(data_root):
    transform = transforms.Compose(
        [transforms.ToTensor(),])

    train_dataset = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)
    return(train_loader, test_loader)