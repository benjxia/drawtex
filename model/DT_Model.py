import torch
from torch import nn

from DT_Dataset import DT_Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class DT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout2d(p=0.2)

        self.conv1 = nn.Conv2d(1, 200, 9, bias=False)
        self.norm1 = nn.BatchNorm2d(200)

        self.conv2 = nn.Conv2d(200, 300, 9, bias=False)
        self.norm2 = nn.BatchNorm2d(300)

        self.conv3 = nn.Conv2d(300, 400, 5, bias=False)
        self.norm3 = nn.BatchNorm2d(400)

        self.conv4 = nn.Conv2d(400, 512, 3, bias=False)
        self.norm4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 512, 3, bias=False)
        self.norm5 = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(225792, 374)

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

        x = self.conv5(x)  # 23x23 -> 21x21
        x = self.norm5(x)
        x = self.relu(x)

        x = self.drop(x)

        x = torch.flatten(x.permute(0, 2, 3, 1), 1)

        x = self.lin1(x)

        return x


def train(epoch_cnt):
    """
    Train for 1 epoch
    """
    total_steps = len(train_load)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epoch_cnt):
        correct = 0
        total = 0
        for i, (img, label) in enumerate(train_load):
            output = model(img)
            loss = loss_fn(output, label).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (output == label).float().sum()
            total += len(label)
            if i + 1 % 100 == 0:
                print(
                    f"Epoch: [{epoch}/{epoch_cnt}], Step: [{i + 1}/{total_steps}], Loss: [{loss.item():.4f}], Accuracy: [{100 * correct / total}%]")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = DT_Dataset(path="../data",
                           train=True,
                           transform=transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.75, 1),
                                                             fill=255))
    train_load = DataLoader(dataset=train_set,
                            batch_size=128,
                            num_workers=4,
                            shuffle=True)
    test_set = DT_Dataset(path="../data",
                          train=False,
                          transform=transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.75, 1), fill=255))
    test_load = DataLoader(dataset=test_set,
                           batch_size=1000,
                           num_workers=4,
                           shuffle=False)
    model = DT_Model().to(device)

    train(1)
