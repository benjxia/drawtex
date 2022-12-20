import torch
from torch import nn


class DrawTexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, 5)  # 45x45 -> 41x41
        self.conv2 = nn.Conv2d(10, 20, 3)  # 41x41 -> 39x39
        self.lin1 = nn.Linear(30420, 500)
        self.lin2 = nn.Linear(500, 82)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 30420)
        x = self.relu(self.lin1(x))
        x = self.lin2(x)
        return x
