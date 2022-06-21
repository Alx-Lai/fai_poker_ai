import torch
import torch.nn as nn

class Model(nn.Module):
    def init(self):
        self.model = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, card):
        out = self.model(card)
        return out