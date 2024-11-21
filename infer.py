print("\nLoading Modules...")
from ToxicityModels import ToxicityClassifier, ToxicityDataset
from train import load_from_dir, BATCH_SIZE
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

LABELS = ToxicityDataset.labels

CLSCR = lambda : print("\x1b[2J\x1b[3J\x1b[1;1H", end="")
RESET = "\x1b[0m"
RESET_TERM = lambda : print(RESET, end="")
U_LINE = "\x1b[4m"
HIGHLIGHT = "\x1b[7m"
RED = "\x1b[30;101m"
GREEN = "\x1b[30;102m"
U_LINE_TEXT = lambda s : U_LINE + s + RESET
HIGHLIGHT_TEXT = lambda s : HIGHLIGHT + s + RESET
RED_TEXT = lambda s : RED + s + RESET
GREEN_TEXT = lambda s : GREEN + s + RESET


# note: max batch size for rtx 3050 mobile(4gb vram) is 4
# note: max batch size for tesla T4(16gb vram) is 16

# test_data = load_from_dir("./test", BATCH_SIZE)
# foreign_data = load_from_dir("./foreign_data", BATCH_SIZE)

PATH = "D:/Misc/model_2_130059.ckpt"

if __name__ == "__main__":
    print("\nLoading model from", PATH)
    model = ToxicityClassifier()
    model.load(PATH)
    model.to("cuda")

    # print(model.evaluate(test_data))
    # print(model.evaluate(foreign_data))

    CLSCR()
    while(text := input(U_LINE_TEXT("Enter text") + ": ")):

        pred_labels = model(text)

        if pred_labels[0]:
            if pred_labels[1]:
                print(RED_TEXT("Hate speech and toxicity detected"))
            else:
                print(RED_TEXT("Hate speech detected"))
        else:
            if pred_labels[1]:
                print(RED_TEXT("Toxicity detected"))
            else:
                print(GREEN_TEXT("No hate speech or toxicity detected"))

        if (pred_labels[0] or pred_labels[1]) and sum(pred_labels[2:]):
            print(HIGHLIGHT_TEXT("Possible subcategories:"), end=" ")
            for pred_label, label in zip(pred_labels[2:], LABELS[2:]):
                if pred_label:
                    print(label, end=", ")
            print("\b\b  ")

        print()

    RESET_TERM()
    CLSCR()