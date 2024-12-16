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


class TrackAgileModuleVer1(nn.Module):
    def __init__(self, input_size=9+3, hidden_size=256, output_size=4, num_layers=2, device='cpu'):
        super(TrackAgileModuleVer1, self).__init__()
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

class TrackAgileModuleVer2Dicision(nn.Module):
    def __init__(self, input_size=9+9, hidden_size=256, output_size=4, num_layers=2, device='cpu'):
        super(TrackAgileModuleVer2Dicision, self).__init__()
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
    
class TrackAgileModuleVer2ExtractorVer2(nn.Module):
    def __init__(self, device):
        super(TrackAgileModuleVer2ExtractorVer2, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=11, stride=11, padding=2)

        self.fc = nn.Sequential(
            nn.Linear(20 * 20, 80),
            nn.ReLU(),
            nn.Linear(80, 9)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # For ReLU activations
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero (optional)
        self.device = device

    def forward(self, x):
        
        x = -self.maxpooling(-x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # print(out.shape)
        # out = torch.sigmoid(out) * 2 - 1
        return out
    
class TrackAgileModuleVer2Extractor(nn.Module):
    def __init__(self, device):
        super(TrackAgileModuleVer2Extractor, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=5, stride=5, padding=0)

        self.fc = nn.Sequential(
            nn.Linear(44 * 44, 44),
            nn.ReLU(),
            nn.Linear(44, 9)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # For ReLU activations
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero (optional)
        self.device = device

    def forward(self, x):
        
        x = -self.maxpooling(-x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # print(out.shape)
        # out = torch.sigmoid(out) * 2 - 1
        return out
    
class DirectionPrediction(nn.Module):
    def __init__(self, device):
        super(DirectionPrediction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9+9, 9),
            nn.ReLU(),
            nn.Linear(9, 3)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # For ReLU activations
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero (optional)
        self.device = device
    def forward(self, x):
        x = self.fc(x)
        return x

class TrackAgileModuleVer3(nn.Module):
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer3, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer2Dicision(device=device).to(device)

        # Initialize Extractor module
        self.extractor_module = TrackAgileModuleVer2ExtractorVer2(device=device).to(device)

        self.directpred = DirectionPrediction(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()