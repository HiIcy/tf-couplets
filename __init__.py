# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/8/29
# __file__ = __init__.py
# __desc__ =
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.ops.rnn import dynamic_rnn
from utils import WordCut
import numpy as np

file = r'F:\Resources\kdata\couplet\origin_data\vocabs'
data_file = r'F:\Resources\kdata\couplet\parse_data\couplets_train.txt'
wordcut = WordCut(file)
go_token = "<GO>"
pad_token = "<PAD>"
end_token = "<EOS>"
unk_token = "<UNK>"

print("当前字符数量 ：", len(wordcut))
for token in ['end', 'unk', 'go', 'pad']:
    wordcut.add_word(eval(token + "_token"))
print("当前字符数量 ：", len(wordcut), "\n第一个字符", wordcut.idx2word(0))

def get_data(data_file):
    inputs = []
    targets = []
    with open(data_file, 'r', encoding='utf-8') as fi:
        for line in fi:  # 不消化内存
            result = line.strip('\n\t')
            if not result:
                continue
            tinput, ttarget = line.strip('\n\t').split(" == ")
            res = []
            for word in tinput:
                if wordcut.reworddict.get(word, -1) == -1:
                    wordcut.add_word(word, 'append')
                res.append(wordcut.reworddict.get(word))
            input = res
            res = []
            for word in ttarget:
                if wordcut.reworddict.get(word, -1) == -1:
                    wordcut.add_word(word, 'append')
                res.append(wordcut.reworddict.get(word))
            # 结束字符
            res.append(wordcut.reworddict.get('<EOS>'))
            target = res
            inputs.append(input)
            targets.append(target)
    return np.array(inputs), np.array(targets)


def gene_data(inputs, targets, batch_size):
    inputs_copy = inputs.copy()
    targets_copy = targets.copy()
    uper = inputs.shape[0]
    indices = np.arange(uper)
    np.random.shuffle(indices)
    count = 0
    while True:
        if count + batch_size <= uper:
            # 这样至少保证了都可以访问到
            input_seq = inputs_copy[count:count + batch_size]
            target_seq = targets_copy[count:count + batch_size]
            max_length_input = max([len(o) for o in input_seq])
            max_length_target = max([len(o) for o in target_seq])
            # REW:pad序列的手写好方法
            inputs_batch = np.zeros((batch_size, max_length_input), dtype=np.int32)
            for i, input_ in enumerate(input_seq):
                for j, element in enumerate(input_):
                    inputs_batch[i, j] = element
            target_batch = np.zeros((batch_size, max_length_target), dtype=np.int32)
            for i, target_ in enumerate(target_seq):
                for j, element in enumerate(target_):
                    target_batch[i, j] = element
            yield inputs_batch, target_batch
            count += batch_size
        else:
            count = 0
            indices = np.arange(uper)
            np.random.shuffle(indices)
            inputs_copy = inputs_copy[indices]
            targets_copy = targets_copy[indices]
            continue


decoder_embedding_dim = 32
encoder_embedding_dim = 32
batch_size = 8

train_inputs, target_inputs = get_data(data_file)
val_inputs, target_val = get_data(data_file)
num_train = train_inputs.shape[0]
num_val = val_inputs.shape[0]
print('data count: ', num_train)

data_gen = gene_data(train_inputs, target_inputs, batch_size)
vocab_size = len(wordcut) + 2
for i in range(6000):
    inputs, targets = next(data_gen)
    if i > 3000:
        seq_lengths = []
        for seq in inputs:
            try:
                length = list(seq).index(0)
            except ValueError:
                length = len(seq)
            seq_lengths.append(length)
        inputs = inputs.tolist()
        targets = targets.tolist()
        for input_, target_ in zip(inputs, targets):
            print(input_, '--', target_)
        print('该batch实际长度:', seq_lengths)
        print()
        input('pause:')
