# from flask import Flask, jsonify, request
# from flask_cors import CORS
import torch.nn as nn
import torch.nn.functional as F
import torch

# app = Flask(__name__)
# CORS(app)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input channel (grayscale), 16 output channels, 3x3 conv
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # -> 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> 14x14

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNN()    
model.load_state_dict(torch.load("mnist_cnn.pth"))


def model_runner(body_tensor):
    label=7
    with torch.no_grad():
        output = model(body_tensor) 
        predicted = torch.argmax(output, 1).item()    
    print(f"Predicted: {predicted}, Actual: {label}")
    return predicted

# @app.route('/')
# def home():
#     return 'Flask is working!'

# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     body1=request.get_json()
#     body_tensor=torch.tensor(body1, dtype=torch.float32).view(1,1, 28, 28)
#     x=model_runner(body_tensor)

#     return jsonify({'message':x })

# if __name__ == '__main__':
#     app.run(debug=True)
