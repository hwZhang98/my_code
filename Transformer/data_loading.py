#!/usr/bin/python
# -*- coding:utf8 -*-
import spacy
from torchtext import data,datasets

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

MAX_LEN = 40
dataset = datasets.TranslationDataset(path='WMT14/europarl-v7', exts=('.de', '.en'), fields=(SRC, TGT)
        , filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)

MIN_SRC_FREQ = 9 # 出现频率小于这个频率的丢掉   ,这个值设置的太小会导致字典尺寸太大，从而导致embed失败
MIN_TGT_FREQ = 3
SRC.build_vocab(dataset.src, min_freq=MIN_SRC_FREQ)
TGT.build_vocab(dataset.trg, min_freq=MIN_TGT_FREQ)
len1 = SRC.vocab.__len__()
len2 = TGT.vocab.__len__()
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
len1 = SRC.vocab.__len__()
len2 = TGT.vocab.__len__()
print('build_vocab is successful')
'''