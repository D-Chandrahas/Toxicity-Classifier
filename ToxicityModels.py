import torch
from torch.optim import Adam
from torch.nn import BCELoss
from BaseModels import Roberta, CustomClassificationHead, EncoderClassifier
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from numpy import float32
from collections import defaultdict, namedtuple


class ToxicityDataset(Dataset):
    cols = ["text", "hate", "toxic", "derogation", "dehumanization", "threatening", "animosity", "obscene"]
    labels = cols[1:]
    def __init__(self, path):
        (dtypes := defaultdict(lambda: float32))["text"] = str
        self.data = pd.read_csv(path, header=0, dtype=dtypes)
        self.data = self.data.reindex(columns=self.cols, fill_value=float32(-1))
        self.loss_mask = [(True if x>=0 else False) for x in self.data.iloc[0, 1:]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index, 0], torch.tensor(self.data.iloc[index, 1:].array, dtype=torch.float32)
    

class ToxicityDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1):
        self.loss_mask = dataset.loss_mask
        super().__init__(dataset, batch_size)


class ToxicityClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        encoder = Roberta("roberta-base")
        classification_head = CustomClassificationHead(encoder.config.hidden_size, 384, 7)
        self.model = EncoderClassifier(encoder, classification_head)

    def to(self, device):
        self.model.to(device)
        return self

    def get_confusion_mat(self, data):
        if type(data) != list and type(data) != tuple:
            data = (data,)
        h_tp, h_tn, h_fp, h_fn = 0, 0, 0, 0
        t_tp, t_tn, t_fp, t_fn = 0, 0, 0, 0
        self.eval()
        with torch.no_grad():
            for dataloader in data:
                for sequences, labels in dataloader:
                    pred_labels = self.model(sequences)

                    for pred_label, label in zip(pred_labels, labels):
                        pred_label = pred_label[0].item()
                        label = label[0].item()
                        if label == -1.0: break
                        if pred_label >= 0.5:
                            if label == 1.0:
                                h_tp += 1
                            else:
                                h_fp += 1
                        else:
                            if label == 0.0:
                                h_tn += 1
                            else:
                                h_fn += 1

                    for pred_label, label in zip(pred_labels, labels):
                        pred_label = pred_label[1].item()
                        label = label[1].item()
                        if label == -1.0: break
                        if pred_label >= 0.5:
                            if label == 1.0:
                                t_tp += 1
                            else:
                                t_fp += 1
                        else:
                            if label == 0.0:
                                t_tn += 1
                            else:
                                t_fn += 1
        
        CM = namedtuple("ConfusionMatrix", ["tp", "tn", "fp", "fn"])

        return CM(h_tp, h_tn, h_fp, h_fn), CM(t_tp, t_tn, t_fp, t_fn)
    
    def evaluate(self, data):
        h, t = self.get_confusion_mat(data)
        h_acc = (h.tp + h.tn) / h_sum if (h_sum := h.tp + h.tn + h.fp + h.fn) else 0.0
        t_acc = (t.tp + t.tn) / t_sum if (t_sum := t.tp + t.tn + t.fp + t.fn) else 0.0
            
        return f"TP:{h.tp}, TN:{h.tn}, FP:{h.fp}, FN:{h.fn} - Accuracy:{h_acc:.4f}; TP:{t.tp}, TN:{t.tn}, FP:{t.fp}, FN:{t.fn} - Accuracy:{t_acc:.4f}"
    
    def custom_train(self, train_data, epochs, valid_data=None, save_path="", resume_from=None, learning_rate=1e-5):
        return self.model.train_model(Adam, {"lr":learning_rate}, BCELoss, {}, train_data, epochs, self.evaluate, valid_data, save_path, resume_from)
    
    def classify(self, text):
        if type(text) != str: raise TypeError("text must be a string")
        self.eval()
        with torch.no_grad():
            out =  self.model(text, batch_mode=False)
            return out.round().sum(dim=0, dtype=torch.int32).bool().tolist()
        
    def __call__(self, text):
        return self.classify(text)
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        return self.model.load(path)