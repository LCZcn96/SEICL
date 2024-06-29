import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.norm0 = nn.BatchNorm1d(input_size)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_size // 4), 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, hidden_size)
        self.norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.norm0(x)
        x = self.relu0(x)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CLModel(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_classes):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.projection_head = ProjectionHead(hidden_size, hidden_size, proj_size)
        self.classifier = Classifier(hidden_size, num_classes)

    def forward(self, x, mode='supervised'):
        features = self.encoder(x)
        if mode == 'supervised':
            logits = self.classifier(features)
            return logits
        elif mode == 'self_supervised':
            projections = self.projection_head(features)
            return projections
        else:
            raise ValueError("Invalid mode: {}".format(mode))