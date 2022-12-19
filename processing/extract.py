import cv2
import numpy as np
import os

NUM_DATA = 375974  # Number of images in training data
DATA_RES_X = 45  # Resolution of each image horizontally
DATA_RES_Y = 45  # Resolution of each image vertically
DATA_PATH = "../data/extracted_images"

# Matrix of all images, data_matrix[i] is i'th image
data_matrix = np.ndarray(shape=(NUM_DATA, DATA_RES_Y, DATA_RES_X, 1), dtype=np.uint8)

# Matrix of all image labels, label_matrix[i] is label of i'th image
label_matrix = np.ndarray(shape=(NUM_DATA), dtype=np.uint32)


THRESH = 128 # Threshold for image grayscale -> B/W conversion

if __name__ == "__main__":  # String to index mapping
    classes: list[str] = os.listdir(DATA_PATH)  # Index to string mapping
    mappings: dict[str, int] = dict(zip(classes, range(len(classes))))  # String to index mapping

    i = 0
    for directory in classes:
        for file in os.listdir(f"{DATA_PATH}/{directory}"):
            # Read image, convert to black and white
            img: np.ndarray = cv2.imread(f"{DATA_PATH}/{directory}/{file}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.threshold(img, THRESH, 255, cv2.THRESH_BINARY)[1]
            # Convert image shape to have 1 channel (standard image format)
            img = img.reshape((DATA_RES_Y, DATA_RES_X, 1))
            # Normalize image to binary
            img //= 255
            # Save to matrices
            data_matrix[i] = img
            label_matrix[i] = np.uint8(mappings[directory])
            i += 1
            if i % 1000 == 0:
                print(f"Processed {i}/{NUM_DATA}")

    np.save("../data/data_matrix.npy", data_matrix)
    np.save("../data/label_matrix.npy", label_matrix)
