#!/usr/bin/python
# -*- coding:utf8 -*-
from each_model import *
from Model import *
import numpy as np
import torch
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
from nltk.translate import bleu_score
import os
from data_loading import  *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# 代码出处:http://nlp.seas.harvard.edu/2018/04/03/attention.html
class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]  # 解码器输入 训练时解码器的输入实际是目标语句，这里体现了teacher focing
            self.trg_y = trg[:, 1:]  # 解码器期望输出  右移一位
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum()  # 令牌的数量

    def make_std_mask(self, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1))  # 最终通过广播使得tgt_mask对于每一个句子
        # 的尺寸为 （N,L,L） N 为句子个数，L为句子长度
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    'Standard Training and Logging Function'
    # start = time.time()
    # print(start,'this is strat')
    total_tokens = 0  # 总令牌数量
    total_loss = 0  # 总损失
    tokens = 0  # 当前批次的令牌数量
    writer = SummaryWriter('bleu_store')
    for i, batch in enumerate(data_iter):
        print('迭代', i)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        print("loss", total_loss)
        if i % 200 == 1:
            # elapsed = time.time() - start + 1  # 防止时间少于1秒从而报错
            bleu = valid_bleu_store()
            print('Epoch Step: %d Loss:%f' %
                  (i, loss / batch.ntokens))
            writer.add_scalars('bleu %d epoch' % epoch, {'bleu': bleu}, i)
            # start = time.time()
            tokens = 0
    writer.close()
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):  # sofar是src,tgt句子当前的最大长度
    # 根据句子的长度动态更新当前批次的句子数量
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:  # count 应该是句子的数量
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)  # 因为多一个开始符和结束符
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
        # 更新学习率
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')  # https://www.cnblogs.com/kk17/p/10246324.html
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):  # https://www.cnblogs.com/zyrb/p/9699168.html
        '''
        标签平滑实际是改变了真实Label的的概率函数，然后将概率函数和log预测函数一起输入到KLDivLoss 中
        这里的x实际是经过softmax和全连接层后的log(p(k|x))，这里最后的true_dist实际是真实概率分布函数
        '''
        # 这里的x是生成的预测语句，target为目标语句
        assert x.size(1) == self.size  # size(0)为一句话中单词的个数，size(1) 为字典总单词个数 36327
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 这一句就是所有位置加上 smoothing*(1/k)  K为类别
        # 减去二是因为要减去padding_idx和正确的label本身
        # 下面这一句就是将对应的one-hot改为（1-smoothing）*1 = 0.9 ，1 为原本one-hot对应正确位置的取值
        # ，这一句没有加上上面那一项估计是因为太小了可以忽略
        true_dist.scatter_(1, target.unsqueeze(1).to(torch.long),
                           self.confidence)  # https://blog.csdn.net/duan_zhihua/article/details/82556676
        true_dist[:, self.padding_idx] = 0  # 这里padding_idx为1  , 应该意思是在one-hot 中 pad的值为[0,1,0.....]
        mask = torch.nonzero(target == self.padding_idx)  # 查看目标语句中是否有填充的部分，有的话返回他的索引 （n,2）n个填充
        # mask 的size为（N,1） N为句子的数目
        if len(mask) > 0 and mask.sum() > 0:  # 这句根据评论有改动 mask.dim() > 0
            true_dist.index_fill_(0, mask.squeeze(), 0.0)  # 这里的squeeze()把mask变为一个一维张量
        # 这一句就把true_dist（N，vocab_size）中的值为pad的一维（1，vocab_size）全变为0，代表着把这一个pad单词的位置全部变为0
        self.true_dist = true_dist
        # 这一句拿预测的x与标签平滑过的结果进行计算损失函数
        return self.criterion(x, true_dist.requires_grad_(False))


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                i = 0
                for p in data.batch(d, self.batch_size * 5):  # 每次P总共返回batch*5个句子
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    i += 1
                    # p_batch 按顺序排好后每次返回一个批次，批次的数量由句子长短决定， count*size = batch
                    # 一批句子中最长的句子越短，则这批句子的数量越多
                    for b in random_shuffler(list(p_batch)):  # 打乱顺序，随机返回一个句子
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
            print('i =%d' % i)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.ones(src.shape[0], 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = model.decode(memory, src_mask,
                               ys,
                               subsequent_mask(ys.size(1)).type_as(src.data))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat((ys, next_word.unsqueeze(-1)), dim=1)
        return ys


class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)  # 将out放入多块GPU中，按照一批中句子总数按比例分配
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):  # 每一块GPU中的每一行句子又分为多块
            # Predict distributions
            # 所有句子的第i块组成out_column
            # out_column = [torch.tensor(o[:, i:i + chunk_size],requires_grad=self.opt is not None)
            out_column = [o[:, i:i + chunk_size].clone().detach().requires_grad_(self.opt is not None)
                          for o in out_scatter]
            # 将每块的输出送入softmax和全连接层进行预测处理
            gen = nn.parallel.parallel_apply(generator, out_column)
            # 预测结果
            # Compute loss.
            y = [(g.reshape(-1, g.size(-1)),  # 这句有修改
                  t[:, i:i + chunk_size].reshape(-1))  # 这句话把所有句子中的那一块单词都转换成了一个维度，
                 # 接起来变成了一个长句子
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)  # y包含两部分第一部分为预测结果，第二部分为目标语句
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
devices = [0]
pad_idx = TGT.vocab.stoi["<blank>"]
model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
model.cuda()
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()
BATCH_SIZE = 4000  # 每批的句子个数
# train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
#                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                         batch_size_fn=batch_size_fn, train=True)
# valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
#                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                         batch_size_fn=batch_size_fn, train=False)
train_WMT14_iter = MyIterator(dataset, batch_size=BATCH_SIZE, device=0,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=True)

model_par = nn.DataParallel(model, device_ids=devices)


def valid_bleu_store():
    total_bleu = []
    for i, batch in enumerate(train_WMT14_iter):  # 完整测试
        src = batch.src.transpose(0, 1).cuda()  # src ['i','am','a','man','.']
        trg = batch.trg.transpose(0, 1).cuda()  # trg ['<start>','ha','ha']
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2).cuda()
        out = greedy_decode(model, src, src_mask,  # out ['<start>','ha','ha']
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        # print("Translation:", end="\t")
        hypothesis = [[] for _ in range(out.shape[0])]
        for j in range(out.size(0)):
            for k in range(1, out.size(1)):  # stoi表示所有字符转化为对应数字标签的集合
                sym = TGT.vocab.itos[out[j, k]]  # itos 表示为source to index 和上面相反
                if sym == "</s>": break
                # print(sym, end=" ")
                hypothesis[j].append(sym)
        print()
        # print("Target:", end="\t")
        references = [[] for _ in range(out.shape[0])]
        for j in range(out.size(0)):
            for k in range(1, trg.size(1)):
                sym = TGT.vocab.itos[trg.data[j, k]]
                if sym == "</s>": break
                # print(sym, end=" ")
                references[j].append(sym)
        func = bleu_score.SmoothingFunction()
        arg_bleu = 0.0
        for reference, hypothesi in zip(references, hypothesis):
            try:
                arg_bleu += bleu_score.sentence_bleu(reference, hypothesi, smoothing_function=func.method7)
            except ZeroDivisionError:
                pass
        arg_bleu /= len(references)
        print(len(references), 'references len')
        print('are_bleu:%.4f' % arg_bleu)
        total_bleu.append(arg_bleu)
        print('i = ',i)
    print('total_bleu:%.4f:' % (sum(total_bleu) / len(total_bleu)))
    return sum(total_bleu) / len(total_bleu)


# 是否训练模型 i = Ture
i = False
if i:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        print('train', epoch)
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_WMT14_iter),
                  model_par,
                  MultiGPULossCompute(model.generator, criterion,
                                      devices=devices, opt=model_opt))
else:
    # !wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
    model = torch.load("iwslt.pt")
    model.eval()
    valid_bleu_store()
#
# for i, batch in enumerate(valid_iter):   # 小测试
#     src = batch.src.transpose(0, 1)[1:2].cuda()  # 第一段，相当于
#     src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2).cuda()
#     out = greedy_decode(model, src, src_mask,
#                         max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
#     print("Translation:", end="\t")
#     hypothesis = []
#     for i in range(1, out.size(1)):  # stoi表示所有字符转化为对应数字标签的集合
#         sym = TGT.vocab.itos[out[0, i]]  # itos 表示为source to index 和上面相反
#         if sym == "</s>": break
#         print(sym, end=" ")
#         hypothesis.append(sym)
#     print()
#     print("Target:", end="\t")
#     references = []
#     for i in range(1, batch.trg.size(0)):
#         sym = TGT.vocab.itos[batch.trg.data[i, 1]]
#         if sym == "</s>": break
#         print(sym, end=" ")
#         references.append(sym)
#     func = bleu_score.SmoothingFunction()
#     bleu = bleu_score.sentence_bleu(references,hypothesis,smoothing_function=func.method7)
#     print('bleu:{}'.format(bleu))
#     print()
