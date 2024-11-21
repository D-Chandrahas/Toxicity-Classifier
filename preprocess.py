from os import makedirs
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "./data1"
makedirs(TRAIN_PATH := "./train", exist_ok=True)
makedirs(VALID_PATH := "./valid", exist_ok=True)
makedirs(TEST_PATH := "./test", exist_ok=True)

data = {}

data["dhate"] = pd.read_csv(f"{DATA_PATH}/dhate.csv")

data["ucbhate"] = pd.read_csv(f"{DATA_PATH}/ucbhate.csv")

jigsaw0 = pd.read_csv(f"{DATA_PATH}/jigsaw_0.csv").sample(n=20000)
jigsaw1 = pd.read_csv(f"{DATA_PATH}/jigsaw_1.csv")
data["jigsaw"] = pd.concat([jigsaw0, jigsaw1], ignore_index=True)

cad0 = pd.read_csv(f"{DATA_PATH}/cad_0.csv").sample(n=5000)
cad1 = pd.read_csv(f"{DATA_PATH}/cad_1.csv")
data["cad"] = pd.concat([cad0, cad1], ignore_index=True)

for name, dataset in data.items():

    train_data, test_data = train_test_split(dataset, test_size=0.1, shuffle=True)
    valid_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=True)

    train_data.sort_values("text", key = lambda s: s.str.len(), inplace=True)
    valid_data.sort_values("text", key = lambda s: s.str.len(), inplace=True)
    test_data.sort_values("text", key = lambda s: s.str.len(), inplace=True)

    # train_data.to_csv(f"{TRAIN_PATH}/{name}.csv", index=False)
    # valid_data.to_csv(f"{VALID_PATH}/{name}.csv", index=False)
    # test_data.to_csv(f"{TEST_PATH}/{name}.csv", index=False)
