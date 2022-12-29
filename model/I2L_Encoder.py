import torch
from torch import nn


class I2L_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Activation type stuff
        self.unfold = nn.Unfold(1)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=13, bias=False)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=9, bias=False)
        self.norm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, bias=False)
        self.norm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, bias=False)
        self.norm4 = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(512, 338)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.max_pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.norm3(x)
        x = self.max_pool1(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.norm4(x)
        x = self.max_pool1(x)

        x = self.unfold(x).permute(0, 2, 1)
        x = self.lin1(x)

        return x
