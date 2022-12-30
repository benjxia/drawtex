import torch
from torch import nn


class DT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout2d(p=0.2)

        self.conv1 = nn.Conv2d(1, 64, 9, bias=False)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 9, bias=False)
        self.norm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 5, bias=False)
        self.norm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, bias=False)
        self.norm4 = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(270848, 374)
        self.sftm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)  # 45x45 -> 37x37
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)  # 37x37 -> 29x29
        x = self.norm2(x)
        x = self.relu(x)

        x = self.drop(x)

        x = self.conv3(x)  # 29x29 -> 25x25
        x = self.norm3(x)
        x = self.relu(x)

        x = self.conv4(x)  # 25x25 -> 23x23
        x = self.norm4(x)
        x = self.relu(x)

        x = self.drop(x)

        x = torch.flatten(x.permute(0, 2, 3, 1), 1)

        x = self.lin1(x)
        x = self.sftm(x)

        return x


if __name__ == "__main__":
    model = DT_Model()
    r = torch.rand(1, 1, 45, 45)
    out = model(r)
    print(out)
    print(out.shape)