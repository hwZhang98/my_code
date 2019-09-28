#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

global sizes
seaborn.set_context(context="talk")


# 源码地址http://nlp.seas.harvard.edu/2018/04/03/attention.html

class EncoderDecoder(nn.Module):
    '''
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    '''

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = src_embed  # 源语句词嵌入
        self.tgt_embed = tgt_embed  # 目标语句词嵌入
        self.generator = generator  # 生成器，生成最终词向量

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        '''
        src:源语句
        tgt:目标语句
        src_mask:源语句掩码
        tgt_mask:目标语句掩码
        '''
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)   # mask 没变但是src经过了embed编码层

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory 是解码器传过来的隐层中间状态
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # d_model 词向量的维度
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x.float()), dim=-1)  # 这里改动了


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 加入Parameter意味着# 这些参数可以进行不断地学习优化
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    '''

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subseqent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # 这个k=1相当于增加了上三角矩阵0的数量
    a = (torch.from_numpy(subseqent_mask) == 0)
    return torch.from_numpy(subseqent_mask) == 0        # 最终返回的是下三角矩阵中0的位置，为True，其余为0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  # size : N (句子个数), 8 (多头注意力数目) , h_size (每个句子中单词的数目) , 64 (向量维度 'd_model' 512/8'h')
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)    # scores 当前的size : N,8,h_size,h_size
    if mask is not None:          # 而此时如果是tgtmask 则mask 的size: N ,1 （每一头注意力的mask相同）,h_size,h_size
        scores = scores.masked_fill(mask == 0, -1e9)  # 根据mask矩阵把mask位置都填入负值，经过softmax后就会变为0
        # 所以这上边一步在mask 为tgtmask时同时引入padmask 和sequentmask
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        '''
        h:多头的数目 default:8
        '''
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h   # 多头注意力中 ,几个变量的维度是由头数和模型隐层维度决定的
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 4个线性层用来产生q,k,v 和最后的组合输出层的权重Wq,Wk,Wv,W
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)        # 一批有多少个句子
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
                             # 注意这里上面的linears其实包含4个不同的全连接层
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    # 简单的前馈神经网络,又两个线性层和一个激活层组成
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)         # 如果字典尺寸超过了Emebed最大尺寸会报错，这里的尺寸为58000左右
        self.d_model = d_model

    def forward(self, x):
        # x = x.to(torch.long)
        y = self.lut(x)
        a = math.sqrt(self.d_model)
        c = y * a
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2).float() *  # 这一句要加float()，否则结果会为0
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)  # 维度偶数用sin ，奇数用cos
        pe[:, 1::2] = torch.cos(position.float() * div_term)  # 这样每个单词的相对位置的每个维度都有各自的位置编码
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)   # 把pe变为self.pe

    def forward(self, x):
        y = self.pe[:, :x.size(1)].clone().detach().requires_grad_(False)
        x = x + y

        return self.dropout(x)

