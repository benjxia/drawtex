import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DrawtexDataset(Dataset):
    classes: list[str] = ['!', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'alpha', 'b', 'beta', 'C', 'cos', 'd', 'Delta', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'G', 'gamma', 'geq', 'gt', 'H', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'leq', 'lim', 'log', 'lt', 'M', 'mu', 'N', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'q', 'R', 'rightarrow', 'S', 'sigma', 'sin', 'sqrt', 'sum', 'T', 'tan', 'theta', 'times', 'u', 'v', 'w', 'X', 'y', 'z', '[', ']', '{', '}']

    mapping: dict[str, int] = {'!': 0, '(': 1, ')': 2, '+': 3, '-': 4, '0': 5, '1': 6, '2': 7, '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14, '=': 15, 'A': 16, 'alpha': 17, 'b': 18, 'beta': 19, 'C': 20, 'cos': 21, 'd': 22, 'Delta': 23, 'div': 24, 'e': 25, 'exists': 26, 'f': 27, 'forall': 28, 'forward_slash': 29, 'G': 30, 'gamma': 31, 'geq': 32, 'gt': 33, 'H': 34, 'i': 35, 'in': 36, 'infty': 37, 'int': 38, 'j': 39, 'k': 40, 'l': 41, 'lambda': 42, 'leq': 43, 'lim': 44, 'log': 45, 'lt': 46, 'M': 47, 'mu': 48, 'N': 49, 'neq': 50, 'o': 51, 'p': 52, 'phi': 53, 'pi': 54, 'pm': 55, 'q': 56, 'R': 57, 'rightarrow': 58, 'S': 59, 'sigma': 60, 'sin': 61, 'sqrt': 62, 'sum': 63, 'T': 64, 'tan': 65, 'theta': 66, 'times': 67, 'u': 68, 'v': 69, 'w': 70, 'X': 71, 'y': 72, 'z': 73, '[': 74, ']': 75, '{': 76, '}': 77}

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, transform=None):
        self.data: np.ndarray = np.load("./data/data_matrix.npy")
        self.labels: np.ndarray = np.load("./data/label_matrix.npy")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: any) -> tuple[torch.Tensor, torch.Tensor]:
        img: np.ndarray = self.data[item]
        label = self.labels[item]

        if self.transform is not None:
            img_tensor: torch.Tensor = self.transform(img)
        else:
            img_tensor: torch.Tensor = torch.from_numpy(img)

        label_tensor: torch.Tensor = torch.tensor(label)
        return img_tensor, label_tensor


# Debugging stuff below
if __name__ == "__main__":
    training_set = DrawtexDataset(transforms.ToTensor())

    train_load: DataLoader = DataLoader(
        dataset=training_set,
        batch_size=100,
        shuffle=True,
        num_workers=4
    )

    iter = enumerate(train_load)
    idx, (img, lab) = next(iter)
    plt.title(training_set.classes[lab[0].cpu()])
    plt.imshow(img[0][0].cpu(), cmap="gray")
    plt.show()
