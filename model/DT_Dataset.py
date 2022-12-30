import torch
from torch.utils.data import Dataset

import pandas as pd
import cv2

import sys

from preprocess.keys import WORD2ID


class DT_Dataset(Dataset):
    def __init__(self, path: str, transform=None, train=True):
        """
        :param path: Path to data folder
        :param transform: Transformations to perform on data
        :param train: True for training data, false for test data
        """
        self.path = path
        self.transform = transform
        self.train = train
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            if self.train:
                self.df = pd.read_csv(f"{path}/train.csv")
            else:
                self.df = pd.read_csv(f"{path}/test.csv")
        except IOError:
            print("Invalid path")
            sys.exit(1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        """
        Retrieves item'th element of dataset
        :param item:
        :return:
        """
        cur_row = self.df.iloc[item]
        img_path = cur_row["path"]
        img = self.prepare_img(f"{self.path}/hasy-data/{img_path}")
        label = WORD2ID[cur_row["latex"]]
        label_tensor = torch.tensor([label]).to(self.device)

        if self.transform is not None:
            img = self.transform(img)

        return img, label_tensor

    def prepare_img(self, path: str) -> torch.Tensor:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((1, 45, 45))
        return torch.from_numpy(img).to(self.device)
