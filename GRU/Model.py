from torch import nn
from torch.autograd import Variable
import torch
import torch.optim
import numpy as np
import time

class CharModel(nn.Module):
    def __init__(self, num_embed, embed_dim, hidden_size, num_layers, dropout,cuda= True):
        super(CharModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.word_to_vec = nn.Embedding(num_embed, embed_dim)           # 定义了一个高位空间，每个词的位置
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, dropout=dropout, bidirectional=True)
        self.project = nn.Linear(hidden_size * 2, num_embed)
        self.device = torch.device("cuda" if cuda else 'cpu')

    def forward(self, x, hs=None):
        batch_size = x.shape[0]
        if hs is None:
            hs = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size,device=self.device) # 如果是双向的则numlayer*2
        x = x.to(torch.long).to(self.device)
        word_embed = self.word_to_vec(x).permute(1, 0, 2)  # (batch, len, embed)
        out, h0 = self.rnn(word_embed, hs)  # (len, batch, hidden)
        len, batch_size, hidden_size = out.shape
        out = self.project(out.view(len * batch_size, hidden_size)).view(len, batch_size, -1)
        out = out.permute(1, 0, 2)  # (batch, len, hidden)
        return out.reshape(-1, out.shape[2]), h0


def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


class Execution:
    def __init__(self, model,cuda=True):
        self.model = model
        self.device = torch.device("cuda" if cuda else 'cpu')

    def train(self, train_data):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0,1,2,3]).to(self.device)
        self.model = self.model.to(self.device)
        start = time.time()

        print("开始训练")
        epochs =200
        for e in range(epochs):
            train_loss = 0
            for i, data in enumerate(train_data):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)
                score, _ = self.model(x)
                loss = criterion(score, y.view(-1).to(torch.long))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), 5)
                optimizer.step()
                train_loss += loss.item()
            print('epoch: {}, perplexity is: {:.3f}'.format(e + 1, np.exp(train_loss / len(train_data))))
        print("结束训练")
        end = time.time()
        print(end-start)

    def eval(self, convert, begin):
        text_len = 30
        self.model.eval()
        self.model = self.model.to(self.device)
        samples = [convert.word_to_int(c) for c in begin]
        input_txt = torch.LongTensor(samples)[None]
        input_txt = input_txt.to(self.device)
        _, init_state = self.model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]
        for i in range(text_len):
            out, init_state = self.model(model_input, init_state)
            pred = pick_top_n(out.data)
            model_input = torch.LongTensor(pred)[None].to(self.device)
            result.append(pred[0])
        text = convert.arr_to_text(result)
        print('Generate text is: {}'.format(text))
