import pandas as pd
import os
import cv2


# Brings in data from CROHME dataset and ports them over to updated HASYv2 dataset
def port(letter: str, latex: str, in_df: pd.DataFrame):
    PATH = f"../data/extracted_images/{letter}"
    PATH_DEST = f"../data/hasy-data"

    cnt = 0
    for file in os.listdir(PATH):
        img = cv2.imread(f"{PATH}/{file}")
        cv2.imwrite(f"{PATH_DEST}/{letter}_{cnt}.png", img)
        in_df = in_df.append({"path": f"{letter}_{cnt}.png",
                              "latex": latex}, ignore_index=True)
        cnt += 1
        if cnt % 200 == 0:
            print(cnt)

    return in_df


if __name__ == "__main__":
    df = pd.read_csv("../data/hasy-data-labels.csv")
    df = df.drop(columns=["symbol_id", "user_id"])

    df = port("cos", "\\cos", df)
    df = port("lim", "\\lim", df)
    df = port("log", "\\log", df)
    df = port("sin", "\\sin", df)
    df = port("tan", "\\tan", df)

    df.to_csv("../data/combined.csv", index=False)
