import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)

class TabShiftPredictor(nn.Module):
    def __init__(self, 
                 device,
                 input_dim,
                 cond_dim, 
                 hidden_dim=128):
        super().__init__()
        self.input_size = input_dim
        self.cond_size = cond_dim
        self.device = device
        self.predictor = nn.Sequential(
            nn.Linear(self.cond_size, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, 64),  # 8
            Swish(),
            nn.Linear(64, self.input_size),  # 32
        )


    def forward(self, x):
        return self.predictor(x)

        # linspace
        # return torch.matmul(x, self.mean_matrix).reshape(-1, self.image_channel, self.image_size, self.image_size)