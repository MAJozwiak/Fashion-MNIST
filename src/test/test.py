import torch
def test(test_loader,model,model_path,device):
    correct = 0
    total = 0
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Final accuracy: {100 * correct / total:.2f}%')
