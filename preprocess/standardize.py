import cv2
import pandas as pd
import torchvision

from skimage import morphology
import matplotlib.pyplot as plt
import os


# Process new images to meet dataset standards
def standardize(img):
    """
    Converts an image to meet the standards of the dataset: 45x45 black/white skeletonized
    :param img: Image to standardize
    :return: Standardized image
    """
    img = cv2.resize(img, (45, 45))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)[1]
    # Convert image shape to have 1 channel (standard image format)
    img = img.reshape((45, 45, 1))
    img ^= 1
    img = morphology.skeletonize(img)
    img ^= 1
    img *= 255
    img = cv2.merge((img, img, img))
    return img


def standardize_dir(path: str):
    """
    Standardizes all images in the given directory
    :param path:
    :return:
    """
    files = os.listdir(path)
    cnt = 0
    for file in files:
        img = cv2.imread(f"{path}/{file}")
        img = standardize(img)
        cv2.imwrite(f"{path}/{file}", img)
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)


if __name__ == "__main__":
    standardize_dir("../data/hasy-data")
