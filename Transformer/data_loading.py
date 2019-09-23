#!/usr/bin/python
# -*- coding:utf8 -*-
import spacy
from torchtext import data,datasets
# raw_data_de = open('europarl-v7/europarl-v7.de',mode='rb').read()
# raw_data_de = raw_data_de.decode().split('\n')
# raw_data_en = open('europarl-v7/europarl-v7.en',mode='rb').read()
# raw_data_en = raw_data_en.decode().split('\n')
#
# print(len(raw_data_de))
# print(len(raw_data_en))
# print(raw_data_en[1])
# print(raw_data_de[1])
# def data_iter(raw_data):
#     for i in range(10):
#         yield raw_data[i*5000:(i+1)*5000]

# raw_data_de = ''.join(raw_data_de[:5000])
# raw_data_en = ''.join(raw_data_en[:5000])

# a = data_iter(raw_data_de)


# data_de = spacy_de.tokenizer(''.join(data_iter(raw_data_de).gi_frame.f_locals['raw_data']))
# data_en = spacy_en.tokenizer(''.join(data_iter(raw_data_en).gi_frame.f_locals['raw_data']))
# print(len(data_de.text))
#
#
# data_de = ''.join(data_iter(raw_data_de).gi_frame.f_locals['raw_data'])
# data_en = ''.join(data_iter(raw_data_en).gi_frame.f_locals['raw_data'])
# print('asddddddddddddddddddddd')
# # MIN_FREQ = 2
# # SRC.build_vocab(train.src, min_freq=MIN_FREQ)
# # TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

# 导入数据

spacy_de = spacy.load('de')  # nlp =spacy.load('de_core_news_sm')
spacy_en = spacy.load('en')  # nlp =spacy.load('en_core_web_sm')


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
print('1234566')
MAX_LEN = 40
dataset = datasets.TranslationDataset(path='WMT14/europarl-v7', exts=('.de', '.en'), fields=(SRC, TGT)
        , filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
print('asdadadadasdad')
MIN_FREQ = 3 # 出现频率小于这个频率的丢掉
SRC.build_vocab(dataset.src, min_freq=MIN_FREQ)
TGT.build_vocab(dataset.trg, min_freq=MIN_FREQ)
print('build_vocab is successful')
'''
#  传入数据！！！！！！！！！！！！！！！！！！！！！！！！！传入
import spacy
from torchtext import datasets, data
spacy_de = spacy.load('de')  # nlp =spacy.load('de_core_news_sm')
spacy_en = spacy.load('en')  # nlp =spacy.load('en_core_web_sm')


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
print('build_vocab')
MAX_LEN = 100  # 大于这个长度的数据丢掉
train, val, test = datasets.IWSLT.splits(
    exts=('.de', '.en'), fields=(SRC, TGT),
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                          len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 2  # 出现频率小于这个频率的丢掉
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
print('build_vocab is successful')
'''