import torch
from torch import nn


class DrawTexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 100, 9, bias=False)  # 45x45 -> 37x37
        self.conv1_bn = nn.BatchNorm2d(200)
        self.conv2 = nn.Conv2d(200, 300, 9, bias=False)  # 37x37 -> 29x29
        self.conv2_bn = nn.BatchNorm2d(400)
        self.conv3 = nn.Conv2d(300, 500, 9, bias=False)  # 29x29 -> 21x21
        self.conv3_bn = nn.BatchNorm2d(800)
        self.conv4 = nn.Conv2d(500, 800, 9, bias=False)  # 21x21 -> 13x13
        self.conv4_bn = nn.BatchNorm2d(1200)
        self.conv5 = nn.Conv2d(800, 1000, 9, bias=False)  # 13x13 -> 5x5
        self.conv5_bn = nn.BatchNorm2d(1800)
        self.lin1 = nn.Linear(25000, 82, bias=False)
        self.lin1_bn = nn.BatchNorm1d(82)

    def forward(self, x: torch.Tensor):
        x: torch.Tensor = self.relu(self.conv1_bn(self.conv1(x)))
        x: torch.Tensor = self.relu(self.conv2_bn(self.conv2(x)))
        x: torch.Tensor = self.relu(self.conv3_bn(self.conv3(x)))
        x: torch.Tensor = self.relu(self.conv4_bn(self.conv4(x)))
        x: torch.Tensor = self.relu(self.conv5_bn(self.conv5(x)))
        x = torch.flatten(x.permute(0, 2, 3, 1), 1)
        x = self.lin1_bn(self.lin1(x))
        return x
