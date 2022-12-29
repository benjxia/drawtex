import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import cv2


class I2L_140K(Dataset):
    """
    PyTorch Dataset class for im2latex 140k dataset
    """

    def __init__(self, path: str, transform=None, mode: str = "train"):
        """
        Create instance of I2L_140K
        :param path: Path to dataset5 folder
        :param transform: Transformations to perform on outputs
        :param mode: "train", "validate", "test"
        """
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.path = path
        self.dfpath = path
        self.transform = transform
        self.mode = mode

        if self.mode == "train":
            self.dfpath = f"{path}/training_56/df_train.pkl"
        elif self.mode == "validate":
            self.dfpath = f"{path}/training_56/df_valid.pkl"
        elif self.mode == "test":
            self.dfpath = f"{path}/training_56/df_test.pkl"
        else:
            raise ValueError("Invalid mode")

        with open(self.dfpath, "rb") as f:
            self.df: pd.DataFrame = pickle.load(f, encoding="latin1")

    def __len__(self):
        """
        :return: Size of dataset (# of elements)
        """
        return len(self.df)

    def __getitem__(self, item):
        """
        Retrives element of given index from dataset

        :param item:
        :return: tuple (element, ground_truth)
        """
        # Read image
        curr_row = self.df.iloc[item]
        img_path = curr_row["image"]
        img = cv2.imread(f"{self.path}/formula_images/{img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert labels and image to tensor
        transform = transforms.ToTensor()
        img_tensor = transform(img).to(self.device)
        label = curr_row["padded_seq"]
        label_tensor = torch.tensor(label).to(self.device)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # Add padding to images and labels
        img_reshape = torch.ones(1, 128, 1088).to(self.device)
        label_reshape = torch.zeros(151, dtype=torch.int32).to(self.device)

        channels = len(img_tensor)
        height = len(img_tensor[0])
        width = len(img_tensor[0][0])

        img_reshape[:, :height, :width] = img_tensor
        label_reshape[:len(label)] = label_tensor

        return img_reshape, label_reshape


# For debugging purposes
if __name__ == "__main__":
    train_set = I2L_140K("../data/dataset5")

    train_load = DataLoader(dataset=train_set,
                            batch_size=64,
                            num_workers=4,
                            shuffle=True)

    idx, (img, label) = next(enumerate(train_load))
    print(img.shape)
    x = torch.flatten(img, start_dim=2)
    print(x.shape)
