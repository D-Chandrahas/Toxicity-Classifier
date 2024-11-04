import torch
from torch.optim import Adam
from torch.nn import BCELoss
from BaseModels import Roberta, CustomClassificationHead, EncoderClassifier
from torch.utils.data import Dataset
import pandas as pd
from collections import namedtuple


class ToxicityDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path, header=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]["text"], torch.tensor([self.data.iloc[index]["label"]], dtype=torch.float32)


class ToxicityClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        encoder = Roberta("roberta-base")
        classification_head = CustomClassificationHead(encoder.config.hidden_size, 256, 1)
        self.model = EncoderClassifier(encoder, classification_head)

    def to(self, device):
        self.model.to(device)
        return self

    def get_confusion_mat(self, data):
        tp, tn, fp, fn = 0, 0, 0, 0
        self.eval()
        with torch.no_grad():
            for sequences, labels in data:
                pred_labels = self.model(sequences)
                for pred_label, label in zip(pred_labels, labels):
                    if pred_label.item() >= 0.5:
                        if label.item() == 1:
                            tp += 1
                        else:
                            fp += 1
                    elif pred_label.item() < 0.5:
                        if label.item() == 0:
                            tn += 1
                        else:
                            fn += 1

        return namedtuple("ConfusionMatrix", ["tp", "tn", "fp", "fn"])(tp, tn, fp, fn)
    
    def evaluate(self, data):
        tp, tn, fp, fn = self.get_confusion_mat(data)
        return f"TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn} - Accuracy:{(tp + tn) / (tp + tn + fp + fn)}"
    
    def custom_train(self, train_data, epochs, valid_data=None, save_path="", resume_from=None, learning_rate=1e-5):
        return self.model.train_model(Adam, {"lr":learning_rate}, BCELoss, {}, train_data, epochs, self.evaluate, valid_data, save_path, resume_from)
    
    def classify(self, text):
        self.eval()
        with torch.no_grad():
            for pred_label in self.model(text, batch_mode=False):
                if pred_label.item() >= 0.5:
                    return True
            return False
        
    def __call__(self, text):
        return self.classify(text)
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        return self.model.load(path)