from each_model import *
from Model import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'


# 代码出处:http://nlp.seas.harvard.edu/2018/04/03/attention.html
class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]  # 解码器输入
            self.trg_y = trg[:, 1:]  # 解码器期望输出  右移一位
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum()

    def make_std_mask(self, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1))
        # type_as(tgt_mask.data)))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    'Standard Training and Logging Function'
    # start = time.time()
    # print(start,'this is strat')
    total_tokens = 0    # 总令牌数量
    total_loss = 0      # 总损失
    tokens = 0          # 当前批次的令牌数量
    for i, batch in enumerate(data_iter):
        print('迭代',i)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        print("loss",total_loss)
        if i % 50 == 1:
            # elapsed = time.time() - start + 1  # 防止时间少于1秒从而报错
            # print(elapsed,'this is elapsed')
            print('Epoch Step: %d Loss:%f'%
                  (i, loss / batch.ntokens))
            # start = time.time()
            # print(start,'this is second start')
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)  # 因为要右移
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# opts = [NoamOpt(512, 1, 4000, None),
#         NoamOpt(512, 1, 8000, None),
#         NoamOpt(256, 1, 4000, None)]
# plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
# plt.legend(["512:4000", "512:8000", "256:4000"])
# plt.show()


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):  # 没看懂
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1).to(torch.long),
                           self.confidence)  # https://blog.csdn.net/duan_zhihua/article/details/82556676
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if mask.sum() > 0 and len(mask) > 0:  # 这句根据评论有改动 mask.dim() > 0
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.requires_grad_(False))

#
# crit = LabelSmoothing(5, 0, 0.4)
# predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0]])
# v = crit(predict.log(),
#          torch.LongTensor([2, 1, 0]))
#
# # Show the target distributions expected by the system.
# plt.imshow(crit.true_dist)
# plt.show()
#
# crit = LabelSmoothing(5, 0, 0.1)
#
#
# def loss(x):
#     d = x + 3 * 1
#     predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],])
#     # print(predict)
#     return crit(predict.log(),
#                 torch.LongTensor([1])).item()
#
#
# plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
# plt.show()
#  传入数据！！！！！！！！！！！！！！！！！！！！！！！！！1
import spacy
from torchtext import datasets, data
print('aaaa')
spacy_de = spacy.load('de')  # nlp =spacy.load('de_core_news_sm')
spacy_en = spacy.load('en')  # nlp =spacy.load('en_core_web_sm')
print('bbbb')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)
print('cccc')
MAX_LEN = 100
train, val, test = datasets.IWSLT.splits(
    exts=('.de', '.en'), fields=(SRC, TGT),
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                          len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
print('ddddd')

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)



# def data_gen(V, batch, nbatches):
#     "Generate random data for a src-tgt copy task."
#     for i in range(nbatches):
#         data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
#         data[:, 0] = 1
#         src = torch.clone(data)
#         tgt = torch.clone(data)
#         yield Batch(src, tgt, 0)

#
# class SimpleLossCompute:
#     "A simple loss compute and train function."
#
#     def __init__(self, generator, criterion, opt=None):
#         self.generator = generator
#         self.criterion = criterion
#         self.opt = opt
#
#     def __call__(self, x, y, norm):
#         x = self.generator(x)
#         loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
#                               y.contiguous().view(-1)) / norm
#         loss.backward()
#         if self.opt is not None:
#             self.opt.step()
#             self.opt.optimizer.zero_grad()
#         return loss.item() * norm
#
#
# # Train the simple copy task.
# V = 11
# criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
# model = make_model(V, V, N=2)
# model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
#                     torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#
# for epoch in range(10):
#     model.train()
#     run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
#     model.eval()
#     print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))
#
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
#
# model.eval()
# src = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]])
# src_mask = torch.ones(1, 1, 10)
# print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion,devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,devices=self.devices)
        out_scatter = nn.parallel.scatter(out,target_gpus=self.devices) # 将out放入多块GPU中，按照一批中句子总数按比例分配
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size): # 每一块GPU中的每一行句子又分为多块
            # Predict distributions
            # 所有句子的第i块组成out_column
            # out_column = [torch.tensor(o[:, i:i + chunk_size],requires_grad=self.opt is not None)
            out_column = [o[:, i:i + chunk_size].clone().detach().requires_grad_(self.opt is not None)
                             for o in out_scatter]  # 修改这一句，然后跑代码
            gen = nn.parallel.parallel_apply(generator, out_column)
            # 预测结果
            # Compute loss.
            y = [(g.reshape(-1, g.size(-1)),   # 这句有修改
                  t[:, i:i + chunk_size].reshape(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)
            l = nn.parallel.gather(loss, target_device=self.devices[0])

            # Sum and normalize loss
            l = l.sum()
            l = l / normalize
            total += l.item()  # 因为每次的损失其实是所有句子里的第i块，所以需要全加起来
            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, _ in enumerate(loss):  # j应该是0,1
                    out_grad[j].append(out_column[j].grad)

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [torch.cat(og, dim=1) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


# GPUs to use
devices = [0,1,2]
pad_idx = TGT.vocab.stoi["<blank>"]
model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
model.cuda()
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()
BATCH_SIZE = 12000
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)
model_par = nn.DataParallel(model, device_ids=devices)
para = sum([np.prod(list(p.size())) for p in model.parameters()])
print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
# 是否训练模型 i = Ture
i = False
if i:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        print('trian',epoch)
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  MultiGPULossCompute(model.generator, criterion,
                                      devices=devices, opt=model_opt))
        print('开始验证')
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model_par,
                         MultiGPULossCompute(model.generator, criterion,
                                             devices=devices, opt=None))
        print(loss)
else:
    # !wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
    model = torch.load("iwslt.pt")


for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]   # 第一段，相当于
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask,
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):  # stoi表示所有字符转化为对应数字标签的集合
        sym = TGT.vocab.itos[out[0, i]] # itos 表示为source to index 和上面相反
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break
