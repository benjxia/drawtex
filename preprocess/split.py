import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../data/combined.csv")

    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    train.to_csv("../data/train.csv", index=False)
    test.to_csv("../data/test.csv", index=False)
