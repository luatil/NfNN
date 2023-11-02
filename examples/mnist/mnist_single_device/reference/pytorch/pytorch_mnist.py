import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F

device = 'cpu'

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
full_train_data = datasets.MNIST(
    root='../../../dataset', train=True, download=False, transform=transform)

# Split the original training data into a training set and a validation set
num_train = int(len(full_train_data) * 0.8)
num_val = len(full_train_data) - num_train
train_data, val_data = random_split(full_train_data, [num_train, num_val])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize the model, loss function and optimizer
criterion = nn.NLLLoss()

w1 = torch.rand((784, 512), requires_grad=True)
b1 = torch.rand((512,), requires_grad=True)
w2 = torch.rand((512, 10), requires_grad=True)
b2 = torch.rand((10,), requires_grad=True)

optimizer = optim.SGD([w1, b1, w2, b2], lr=0.01)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        flatten = images.reshape(-1, 28*28)
        l1 = flatten @ w1
        l1b = l1 + b1
        r1 = F.relu(l1b)
        l2 = r1 @ w2
        l2b = l2 + b2
        outputs = F.log_softmax(l2b, dim=1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation loop
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            flatten = images.reshape(-1, 28*28)
            l1 = flatten @ w1
            l1b = l1 + b1
            r1 = torch.relu(l1b)
            l2 = r1 @ w2
            l2b = l2 + b2
            outputs = F.log_softmax(l2b, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Validation Accuracy: {val_accuracy}%")

print("Training completed!")
