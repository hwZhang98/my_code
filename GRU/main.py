import read_data as rd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Model import *
import torch.nn as nn
import numpy as np
from torch.utils.data import sampler

convert = rd.TextConvert('./dataset/poetry.txt', max_vocab=10000)  # 字典
n_step = 128
num_seq = int(convert.vocab_size() / n_step)  # 去掉最后不足一个序列长度的部分
text = convert.vocab[:num_seq * n_step]
arr = convert.text_to_arr(text)
vocab_size = convert.vocab_size()
arr = arr.reshape((num_seq, -1))  # 分成句子
arr = torch.from_numpy(arr)

train_set = rd.TextDataset(arr)  # 获取数据(num_seq=19,n_step=512)  句子长度为19，总共512个句子

batch_size = 64
train_data = DataLoader(train_set, batch_size)

model = CharModel(vocab_size, 512,  512, 2, 0.5,cuda=True)

begin = '易烊千玺'
execution = Execution(model,cuda=True)
execution.train(train_data)
execution.eval(convert, begin)
