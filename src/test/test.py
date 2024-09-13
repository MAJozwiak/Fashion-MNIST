import torch
def test(test_loader,model,model_path,device,scores_path) -> None:
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
    accuracy=100 * correct / total
    with open(scores_path, "w") as file:  # save scores to txt
        file.write(f'Final accuracy: {100 * correct / total:.2f}%')
    print(f'Final accuracy: {100 * correct / total:.2f}%')
