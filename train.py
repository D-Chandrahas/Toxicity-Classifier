from torch.utils.data import DataLoader
from ToxicityModels import ToxicityClassifier, ToxicityDataset

# train_data = ToxicityDataset("./data/train_data.csv")
valid_data = ToxicityDataset("./data/valid_data.csv")


# note: max batch size for rtx 3050 mobile(4gb vram) is 4
# note: max batch size for tesla T4(16gb vram) is 16
# train_dataloader = DataLoader(train_data, batch_size=4)
valid_dataloader = DataLoader(valid_data, batch_size=4)

model = ToxicityClassifier()
model.load("D:/Misc/model_85.pth")
model.to("cuda")

# model.custom_train(train_dataloader, 2, valid_dataloader, "R:/")

# print(model.evaluate(train_dataloader))
print(model.evaluate(valid_dataloader))
