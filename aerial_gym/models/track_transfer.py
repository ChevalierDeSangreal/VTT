import torch
import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class TrackAgileModuleVer2Dicision(nn.Module):
    def __init__(self, input_size=9+9, hidden_size=256, output_size=4, num_layers=2, seq_len=10, device='cpu'):
        super(TrackAgileModuleVer2Dicision, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, seq_len * output_size).to(device)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size).to(x[0].device)
        # print(x.shape, h0.shape)
        embedding, _ = self.gru(x, h0)
        # print(out.shape)
        out = self.fc(embedding[-1, :, :])
        # print(out.shape)
        out = torch.sigmoid(out) * 2 - 1
        out = out.view(x.shape[1], self.seq_len, -1)
        return out, embedding[-1, :, :]
    
class TrackTransferModuleVer0Predict(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, seq_len=5, output_size=9, device='cpu'):
        super(TrackTransferModuleVer0Predict, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.seq_len * output_size)
        ).to(device)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.mlp(x)
        out = out.view(batch_size, self.seq_len, -1)
        return out
    

class TrackTransferModuleVer0(nn.Module):
    """
    Based on Ver3
    Use distance to output action
    """
    def __init__(self, device='cpu'):
        super(TrackTransferModuleVer0, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer2Dicision(input_size=9+3,device=device).to(device)

        self.predict_module = TrackTransferModuleVer0Predict(device=device).to(device)

    

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()