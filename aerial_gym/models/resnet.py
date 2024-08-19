
""" 
    Modified from BcFull in https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation/blob/master/racing_models/bc_full.py
"""
import torch
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.create_model(0.5)

    def forward(self, img):
        # Input
        x1 = self.conv0(img)
        x1 = self.max0(x1)

        # First residual block
        x2 = self.bn0(x1)
        x2 = nn.ReLU()(x2)
        x2 = self.conv1(x2)

        x2 = self.bn1(x2)
        x2 = nn.ReLU()(x2)
        x2 = self.conv2(x2)

        x1 = self.conv3(x1)
        x3 = torch.add(x1, x2)

        # Second residual block
        x4 = self.bn2(x3)
        x4 = nn.ReLU()(x4)
        x4 = self.conv4(x4)

        x4 = self.bn3(x4)
        x4 = nn.ReLU()(x4)
        x4 = self.conv5(x4)

        x3 = self.conv6(x3)
        x5 = torch.add(x3, x4)

        # Third residual block
        x6 = self.bn4(x5)
        x6 = nn.ReLU()(x6)
        x6 = self.conv7(x6)

        x6 = self.bn5(x6)
        x6 = nn.ReLU()(x6)
        x6 = self.conv8(x6)

        x5 = self.conv9(x5)
        x7 = torch.add(x5, x6)

        x = torch.flatten(x7, 1)

        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        vel_cmd = self.dense3(x)

        return vel_cmd

    def create_model(self, dropout_prob):
        print('[ResNet] Starting model')

        self.max0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn0 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)

        self.dense0 = nn.Linear(6272, 128)
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, 16)
        self.dense3 = nn.Linear(16, 10)
        
        self.dropout = nn.Dropout(dropout_prob)

        print('[ResNet] Done with model')