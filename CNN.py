
import torch
import torch.optim as optim
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
test_imgs  = load_idx_images('t10k-images.idx3-ubyte')
test_lbls  = load_idx_labels('t10k-labels.idx1-ubyte')

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
model.load_state_dict(torch.load("mnist_cnn.pth"))



def temp():
    test_data = torch.tensor(test_imgs)
    test_labels = torch.tensor(test_lbls)
    
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model.eval()

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            max_val, predicted = torch.max(outputs.data, 1)
            for pred in predicted:
                all_preds.append(pred.item())
            for label in labels:    
                all_labels.append(label.item())
    
    count =0
    for i in range(len(all_labels)):
        if all_labels[i] == all_preds[i]:
            count+=1
    accuracy = 100 * count / len(all_labels)
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Number of classes
    num_classes = 10
    
    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Populate confusion matrix
    for true_label, pred_label in zip(all_labels, all_preds):
        cm[true_label, pred_label] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(8,6))
    plt.imshow(cm)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)

    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    
    # Label axes
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Annotate each cell with number of samples
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i][j],ha="center", va="center")
    
    plt.show()

temp()


