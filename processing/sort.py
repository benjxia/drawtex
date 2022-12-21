import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from skimage import morphology


def standardize(img):
    img = cv2.resize(img, (45, 45))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)[1]
    # Convert image shape to have 1 channel (standard image format)
    img = img.reshape((45, 45, 1))
    img ^= 1
    img = morphology.skeletonize(img)
    img ^= 1
    return img


def sort_csv(path: str):
    key: pd.DataFrame = pd.read_csv(path)
    img = cv2.imread("../data/HASYv2/hasy-data/v2-00000.png")
    img = standardize(img)
    plt.imshow(img)
    plt.show()

path1 = "../data/HASYv2/classification-task/fold-10/test.csv"