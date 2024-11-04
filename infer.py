print("\nLoading Modules...")
# from torch.utils.data import DataLoader
from ToxicityModels import ToxicityClassifier, ToxicityDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# test_data = ToxicityDataset("./data/test_data.csv")
# hate_check = ToxicityDataset("./data/hate_check.csv")

# note: max batch size for rtx 3050 mobile(4gb vram) is 4
# note: max batch size for tesla T4(16gb vram) is 16
# test_dataloader = DataLoader(test_data, batch_size=4)
# hate_check_dataloader = DataLoader(hate_check, batch_size=4)

PATH = "D:/Misc/model_85.pth"
print("\nLoading model from", PATH)
model = ToxicityClassifier()
model.load(PATH)
model.to("cuda")

# print(model.evaluate(test_dataloader))
# print(model.evaluate(hate_check_dataloader))

print("\x1b[3J\x1b[2J\x1b[1;1H", end="")
while(text := input("\x1b[4mEnter text\x1b[0m: ")):
    print("\x1b[30;101mHate speech detected" if model(text) else "\x1b[30;102mNo hate speech detected", end="\x1b[0m\n\n")