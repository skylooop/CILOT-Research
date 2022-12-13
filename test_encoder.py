import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, state_shape) -> None:
        super().__init__()
        
        self.linear_net = nn.Sequential(
            nn.Linear(state_shape, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, state_shape),
        )
    
    def forward(self, x):
        #x = x / 255.0 - 0.5
        x = self.linear_net(x)
        x = x.view(x.shape[0], -1)
        return x