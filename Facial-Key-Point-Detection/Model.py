from torch import nn
import torch.nn.functional as F

class DFKModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 2), stride=1, padding='valid')
        self.conv1_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=1, padding='same')
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=1, padding='valid')
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=1, padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_features=16928, out_features=500)
        self.linear2 = nn.Linear(in_features=500, out_features=250)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(in_features=250, out_features=30)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        return x
