import torch
import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from .resnet import Resnet

class TrackAgileModuleVer0(nn.Module):
    """
    First Version of Tracking Agile Target
    """
    def __init__(self, input_size=6, hidden_size1=32, hidden_size2=32, output_size=3, device='cpu'):
        print("TrackSpaceModel Initializing...")

        super(TrackAgileModuleVer0, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ELU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ELU().to(device)
        self.output_layer = nn.Linear(hidden_size2, output_size).to(device)

        torch.nn.init.kaiming_normal_(self.hidden_layer1.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer2.weight)
        torch.nn.init.kaiming_normal_(self.output_layer.weight)


    def forward(self, now_state, rel_dis):
        
        x = torch.cat((now_state, rel_dis), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x) * 2 - 1
        return x


class TrackAgileModuleVer0(nn.Module):
    def __init__(self, input_size=9+3, hidden_size=64, output_size=4, num_layers=2, device='cpu'):
        super(TrackAgileModuleVer0, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size).to(x[0].device)
        # print(x.shape, h0.shape)
        out, _ = self.gru(x, h0)
        # print(out.shape)
        out = self.fc(out[-1, :, :])
        # print(out.shape)
        out = torch.sigmoid(out) * 2 - 1
        return out

