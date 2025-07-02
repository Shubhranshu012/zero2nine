import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def load_idx_images(path):
    with open(path, 'rb') as f:
        header = f.read(16)  # Skip the header
        data = np.frombuffer(f.read(), np.uint8)
    num = data.size // (28*28)
    return data.reshape(num, 1, 28, 28).astype(np.float32) / 255.0

def load_idx_labels(path):
    with open(path, 'rb') as f:
        header = f.read(8)  # Skip the header
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
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input channel (grayscale), 16 output channels, 3x3 conv
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # -> 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> 14x14

        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        self.fc2 = nn.Linear(512 , 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))               
        x = F.relu(self.fc3(x))             
        x = self.fc4(x)                       
        return x
    
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")    

torch.save(model.state_dict(), "mnist_cnn.pth")    