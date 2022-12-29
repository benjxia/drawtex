import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from torchvision import transforms

from I2L_140K import I2L_140K


class I2L_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Activation type stuff
        self.unfold = nn.Unfold(1)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=13)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    train_set = I2L_140K("../data/dataset5")

    train_load = DataLoader(dataset=train_set,
                            batch_size=64,
                            num_workers=4,
                            shuffle=True)

    model2 = I2L_Encoder()
    idx, (img, label) = next(enumerate(train_load))
    tmp_tgt = torch.cat((torch.ones(len(label), 1), label), dim=1)
    print(img.shape)
    x = torch.flatten(img, start_dim=2)
    print(x.shape)
    out = model2(img)
    print(out.shape)
