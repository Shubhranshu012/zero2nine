# Just Neural Networks

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def load_idx_images(path):
    with open(path, 'rb') as f:
        _ = f.read(16)  # Skip the header
        data = np.frombuffer(f.read(), np.uint8)
    num = data.size // (28*28)
    return data.reshape(num, 1, 28, 28).astype(np.float32) / 255.0

def load_idx_labels(path):
    with open(path, 'rb') as f:
        _ = f.read(8)  # Skip the header
        labels = np.frombuffer(f.read(), np.uint8)
    return labels.astype(np.int64)

# Usage
train_imgs = load_idx_images('train-images.idx3-ubyte')
train_lbls = load_idx_labels('train-labels.idx1-ubyte')
test_imgs  = load_idx_images('t10k-images.idx3-ubyte')
test_lbls  = load_idx_labels('t10k-labels.idx1-ubyte')



class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create datasets and loaders
train_dataset = MNISTDataset(train_imgs, train_lbls)
test_dataset  = MNISTDataset(test_imgs, test_lbls)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 30)   # input layer
        self.fc2 = nn.Linear(30, 30)      # hidden layer
        self.fc3 = nn.Linear(30, 10)       # output layer

    def forward(self, x):
        x = x.view(-1, 28*28)              # flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")





plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='green')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()