import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

idx = 300000

data: np.ndarray = np.load("./data/data_matrix.npy")
label: np.ndarray = np.load("./data/label_matrix.npy")
print(label[idx])
classes = os.listdir("./data/extracted_images")
mappings = dict(zip(classes, range(len(classes))))
print(classes[label[idx]])
plt.imshow(data[idx], cmap="gray", interpolation="none")
plt.show()
