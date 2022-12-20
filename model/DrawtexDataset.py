import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DrawtexDataset(Dataset):
    classes: list[str] = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A',
                          'alpha', 'ascii_124', 'b', 'beta', 'C', 'cos', 'd', 'Delta', 'div', 'e', 'exists', 'f',
                          'forall', 'forward_slash', 'G', 'gamma', 'geq', 'gt', 'H', 'i', 'in', 'infty', 'int', 'j',
                          'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'M', 'mu', 'N', 'neq', 'o', 'p',
                          'phi', 'pi', 'pm', 'prime', 'q', 'R', 'rightarrow', 'S', 'sigma', 'sin', 'sqrt', 'sum', 'T',
                          'tan', 'theta', 'times', 'u', 'v', 'w', 'X', 'y', 'z', '[', ']', '{', '}']
    mapping: dict[str, int] = {'!': 0, '(': 1, ')': 2, '+': 3, ',': 4, '-': 5, '0': 6, '1': 7, '2': 8, '3': 9, '4': 10,
                               '5': 11, '6': 12, '7': 13, '8': 14, '9': 15, '=': 16, 'A': 17, 'alpha': 18,
                               'ascii_124': 19, 'b': 20, 'beta': 21, 'C': 22, 'cos': 23, 'd': 24, 'Delta': 25,
                               'div': 26, 'e': 27, 'exists': 28, 'f': 29, 'forall': 30, 'forward_slash': 31, 'G': 32,
                               'gamma': 33, 'geq': 34, 'gt': 35, 'H': 36, 'i': 37, 'in': 38, 'infty': 39, 'int': 40,
                               'j': 41, 'k': 42, 'l': 43, 'lambda': 44, 'ldots': 45, 'leq': 46, 'lim': 47, 'log': 48,
                               'lt': 49, 'M': 50, 'mu': 51, 'N': 52, 'neq': 53, 'o': 54, 'p': 55, 'phi': 56, 'pi': 57,
                               'pm': 58, 'prime': 59, 'q': 60, 'R': 61, 'rightarrow': 62, 'S': 63, 'sigma': 64,
                               'sin': 65, 'sqrt': 66, 'sum': 67, 'T': 68, 'tan': 69, 'theta': 70, 'times': 71, 'u': 72,
                               'v': 73, 'w': 74, 'X': 75, 'y': 76, 'z': 77, '[': 78, ']': 79, '{': 80, '}': 81}
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, transform=None):
        self.data: np.ndarray = np.load("../data/data_matrix.npy")
        self.labels: np.ndarray = np.load("../data/label_matrix.npy")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: any) -> tuple[torch.Tensor, torch.Tensor]:
        img: np.ndarray = self.data[item]
        label = self.labels[item]

        if self.transform is not None:
            img_tensor: torch.Tensor = self.transform(img).to(self.device)
        else:
            img_tensor: torch.Tensor = torch.from_numpy(img).to(self.device)

        label_tensor: torch.Tensor = torch.tensor(label).to(self.device)
        return img_tensor, label_tensor


# Debugging stuff below
if __name__ == "__main__":
    training_set = DrawtexDataset(transforms.ToTensor())

    train_load: DataLoader = DataLoader(
        dataset=training_set,
        batch_size=100,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    iter = enumerate(train_load)
    idx, (img, lab) = next(iter)
    plt.title(training_set.classes[lab[0]])
    plt.imshow(img[0][0], cmap="gray")
    plt.show()
