import torch
import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from .resnet import Resnet

class MLP(nn.Module):

    def __init__(self, input_size=20, hidden_size1=256, hidden_size2=256, hidden_size3=256, hidden_size4=256, output_size=4, device='cpu'):
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.hidden_layer4 = nn.Linear(hidden_size3, hidden_size4).to(device)
        self.activation4 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size4, output_size).to(device)

    def forward(self, imu_quadrotor, rel_dis, latent_intent):
        
        x = torch.cat((imu_quadrotor, rel_dis, latent_intent), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.hidden_layer4(x)
        x = self.activation4(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, output_size=5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, imu_quadrotor, rel_dis):
        x = torch.cat((imu_quadrotor, rel_dis), dim=1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)

        return out

class LSTMDecoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)

        return out

class VTTVer0(nn.Module):
    def __init__(self):
        self.mlp = MLP()
        self.lstm = LSTM()
        self.lstm_decoder = LSTMDecoder()

    def forward(self, imu_quadrotor, rel_dis):
        latent_intent = self.lstm(imu_quadrotor, rel_dis)
        action = self.mlp(imu_quadrotor, rel_dis, latent_intent)
        predict_rel_dis = self.lstm_decoder
        return action, predict_rel_dis
    
    