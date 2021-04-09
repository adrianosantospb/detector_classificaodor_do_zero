import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AdrianoNet(nn.Module):
    
    def __init__(self, num_classes):
        super(AdrianoNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.Dropout = nn.Dropout(0.3)

        self.header = nn.Linear(in_features=1038336, out_features=4+num_classes)


    def forward(self, x):
        
        output = self.conv1(x)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool(output)

        output = self.conv3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.relu4(output)

        output = output.view(output.size(0), -1)
        
        output = self.Dropout(output)

        return self.header(output) 
