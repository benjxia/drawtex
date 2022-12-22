import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from skimage import morphology


# Brings in data from CROHME dataset and ports them over to updated HASYv2 dataset
if __name__ == "__main__":
    df = pd.read_csv("../data/HASYv2/combined2.csv")
    df = df.drop(columns=["ooga"])
    PATH_SRC = "../data/extracted_images/z"
    letter= "Z"
    PATH_DEST = "../data/HASYv2/hasy-data"
    cnt = 0
    for file in os.listdir(PATH_SRC):
        img = cv2.imread(f"{PATH_SRC}/{file}")
        cv2.imwrite(f"{PATH_DEST}/{letter}_{cnt}.png", img)
        new_row = {"path" : f"../../hasy-data/{letter}_{cnt}.png", "latex" : f"{letter}"}
        df = df.append(new_row, ignore_index=True)
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
        if cnt >= 999:  # Limit number of training data per label
            break
