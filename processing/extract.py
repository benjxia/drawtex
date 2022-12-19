import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

NUM_DATA = 375974  # Number of images in training data
DATA_RES_X = 45    # Resolution of each image horizontally
DATA_RES_Y = 45    # Resolution of each image vertically

if __name__ == "__main__":
    mappings: dict[str, int] = dict()  # String to index mapping
    mappings_r: list[str] = []  # Index to
    data_matrix = np.ndarray(shape=(NUM_DATA, 1, DATA_RES_Y, DATA_RES_X))
    i = 0
    for root, dirs, files in os.walk("../data/extracted_images", ):
        for f in files:
            # print(root[root.rindex("\\") + 1:])
            # print(f"{root}/{f}")
            img: np.ndarray = cv2.imread(f"{root}/{f}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape((1, DATA_RES_Y, DATA_RES_X))
            data_matrix[i] = img
            i += 1
            if i % 1000 == 0:
                print(f"{i}/{375974}")

    print(data_matrix.shape)

# Reading the image using imread() function
# image: np.ndarray = cv2.imread('../data/extracted_images/!/!_7731.jpg', 1)
#
# image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # plt.imshow(image, cmap= "gray")
# # plt.show()
# # Extracting the height and width of an image
# h, w = image.shape[:2]
# # Displaying the height and width
# # print("Height = {},  Width = {}".format(h, w))
#
# arr = [[1, 2], [3, 4]]
