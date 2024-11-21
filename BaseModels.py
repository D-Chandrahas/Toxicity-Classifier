import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from datetime import datetime


class Roberta(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.device = "cpu"
        self.tokenizer = RobertaTokenizer.from_pretrained(variant)
        self.roberta = RobertaModel.from_pretrained(variant, add_pooling_layer=False)
        self.config = self.roberta.config

    def to(self, device):
        self.device = device
        self.roberta.to(device)
        return self

    def forward(self, text, batch_mode=True):
        with torch.no_grad():
            if batch_mode:
                tokenized_text = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
                input_ids, attention_mask = tokenized_text.input_ids, tokenized_text.attention_mask
            else:
                if type(text) != str: raise TypeError("text must be a string when batch_mode is False")
                max_len = self.tokenizer.model_max_length
                tokenized_text = self.tokenizer(text, return_tensors='pt', verbose=False)
                input_ids, attention_mask = tokenized_text.input_ids, tokenized_text.attention_mask
                of_len = input_ids.shape[1] - max_len
                if of_len > 0:
                    max_sen_len = max_len - 2; stride = 128; step = max_sen_len - stride
                    config = self.roberta.config
                    pad = 0, (step - (of_len % step)) % step
                    input_ids = F.pad(input_ids[0, 1:-1], pad, mode="constant", value=config.pad_token_id)
                    temp = torch.empty( ( ( input_ids.shape[0] - max_sen_len) // step ) + 1, max_sen_len, dtype=torch.int64)
                    for i in range(temp.shape[0]):
                        temp[i] = input_ids[i*step : i*step + max_sen_len]
                    temp = F.pad(temp, (1, 0), mode="constant", value=config.bos_token_id)
                    temp = F.pad(temp, (0, 1), mode="constant", value=config.eos_token_id)
                    attention_mask = torch.ones_like(temp)
                    if pad[1] > 0:
                        temp[-1, -pad[1]-1] = config.eos_token_id
                        temp[-1, -1] = config.pad_token_id
                        attention_mask[-1, -pad[1]:] = 0
                    input_ids = temp
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        encoded_text = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return encoded_text

    def train(self, mode=True):
        self.requires_grad_(mode)
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)


class CustomClassificationHead(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.l1 = torch.nn.Linear(input_size, input_size)
        self.l2 = torch.nn.Linear(input_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)

        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.sigmoid(self.l3(x))
        return x

    def train(self, mode=True):
        self.requires_grad_(mode)
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)
    

class EncoderClassifier(torch.nn.Module):
    def __init__(self, encoder, classification_head):
        super().__init__()
        self.device = "cpu"
        self.encoder = encoder.eval()
        self.classifier = classification_head.eval()

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.classifier.to(device)
        return self

    def forward(self, text, batch_mode=True):
        return self.classifier(self.encoder(text, batch_mode)[:,0])
    
    def __call__(self, text, batch_mode=True):
        return self.forward(text, batch_mode)

    def train_model(self, optimizer, optimizer_kwargs, loss_cls, loss_cls_kwargs, train_data, epochs, valid_fn=None, valid_data=None, save_path="", resume_from=None):
        if save_path and save_path[-1] != "/" and save_path[-1] != "\\": save_path += "/"
        if valid_fn is not None:
            self.valid_fn = valid_fn
        else:
            self.valid_fn = lambda _: ""

        if type(train_data) != list and type(train_data) != tuple:
            train_data = (train_data,)

        optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        criterion = loss_cls(**loss_cls_kwargs)
        start_epoch = 1

        if resume_from:
            epoch, optimizer_state_dict = self.load_ckpt(resume_from)
            start_epoch = epoch + 1
            optimizer.load_state_dict(optimizer_state_dict)

        num_batches_list = list(map(len, train_data))
        fmts = [len(str(num)) for num in num_batches_list]

        for epoch in range(start_epoch, epochs+1):

            self.train()
            sum_loss = 0.0

            curr_batch_list = [0 for _ in num_batches_list]

            for curr_dataloader_idx, dataloader in enumerate(train_data):
                try:
                    loss_mask = dataloader.loss_mask
                except AttributeError:
                    loss_mask = slice(None)

                for curr_batch_idx, (sequences, labels) in enumerate(dataloader):
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    pred_labels = self.forward(sequences)
                    loss = criterion(pred_labels[:, loss_mask], labels[:, loss_mask])
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()

                    if curr_batch_idx % 100 == 0:
                        curr_batch_list[curr_dataloader_idx] = curr_batch_idx + 1
                        loading_bar = ""
                        for curr_batch, fmt, num_batches in zip(curr_batch_list, fmts, num_batches_list):
                            loading_bar += f"{curr_batch:{fmt}}/{num_batches}, "
                        print(f"Epoch {epoch}/{epochs} - {loading_bar[:-2]}", end="\r", flush=True)


            self.save_ckpt(epoch, optimizer.state_dict(), f"{save_path}model_{epoch}_{datetime.now().strftime('%H%M%S')}.ckpt")
            print(f"Epoch {epoch}/{epochs} - Avg. Loss:{(sum_loss/sum(num_batches_list)):.4f}; {self.valid_fn(valid_data)}")


    def save_ckpt(self, epoch, optimizer_state_dict, path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer_state_dict
            }, path)

    def load_ckpt(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.load_state_dict(ckpt["model_state_dict"])
        self.train()
        return ckpt["epoch"], ckpt["optimizer_state_dict"]

    def save(self, path):
        torch.save({"model_state_dict": self.state_dict()}, path)

    def load(self, path):
        out = self.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
        self.eval()
        return out
