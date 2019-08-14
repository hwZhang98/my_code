import numpy as np
import torch


class TextConvert(object):
    def __init__(self, text_path, max_vocab=5000):
        with open(text_path, 'rb') as f:
            text = f.read()
        # print("text type:", type(text))
        text = text.decode()
        # print(text)
        text = text.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')
        vocab = set(text)  # 去掉重复的字符
        vocab_count = {}
        for word in vocab:
            vocab_count[word] = 0
        for word in text:
            vocab_count[word] += 1
        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)

        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab]
        vocab = [x[0] for x in vocab_count_list]
        self.vocab = vocab
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))
        # print(self.int_to_word_table)
        # print(self.word_to_int_table)

    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)


# convert=TextConvert('./dataset/poetry.txt', max_vocab=100)

class TextDataset(object):  # 生成标签
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, item):
        x = self.arr[item, :]

        # 构造 label
        y = torch.zeros(x.shape)
        # 将输入的第一个字符作为最后一个输入的 label
        y[:-1], y[-1] = x[1:], x[0]
        y = y.int()
        return x, y

    def __len__(self):
        return self.arr.shape[0]
