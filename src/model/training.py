import torch
import torch.nn as nn
import torch.optim as optim
from src.model import network
from torchvision import models
from src.model.early_stopping import EarlyStopping
import torch.nn.functional as F
from typing import Tuple


def train(train_loader, val_loader, model_path: str, pretrained: bool) -> Tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = model.to(device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10).to(device)
    else:
        model = (network.ResNet(num_classes=10)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)
    early_stopping = EarlyStopping(patience=7, verbose=True, path=model_path)
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probabilities = F.softmax(outputs, dim=1)

                _,predicted = torch.max(probabilities, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = val_loss / len(val_loader.dataset)
        print(f'Accuracy on validation set: {100 * correct / total:.2f}%')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step(val_loss)

    return model, device