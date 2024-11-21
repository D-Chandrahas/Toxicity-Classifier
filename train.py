from os import walk
from os.path import join
from ToxicityModels import ToxicityClassifier, ToxicityDataset, ToxicityDataloader

TRAIN_DIR = "./train"
VALID_DIR = "./valid"

# note: max batch size for rtx 3050 mobile(4gb vram) is 4
# note: max batch size for tesla T4(16gb vram) is 16
BATCH_SIZE = 4

def load_from_dir(dir, batch_size=1):
    data = []

    for path in (join(dir, file) for file in next(walk(dir), (None, None, []))[2]):
        data.append(
            ToxicityDataloader(
                ToxicityDataset(path),
                batch_size
            )
        )
    return data



train_data = load_from_dir(TRAIN_DIR, BATCH_SIZE)
valid_data = load_from_dir(VALID_DIR, BATCH_SIZE)


if __name__ == "__main__":
    model = ToxicityClassifier()
    # model.load("D:/Misc/model_2_130059.ckpt")
    model.to("cuda")

    model.custom_train(train_data, 2, valid_data, "R:/")

    # print(model.evaluate(train_data))
    # print(model.evaluate(valid_data))
