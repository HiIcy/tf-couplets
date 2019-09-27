# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/8/30
# __file__ = netlib
# __desc__ =
import os
import random

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.ops.rnn import dynamic_rnn
from utils import WordCut
import numpy as np
from tqdm import tqdm

file = r'F:\Resources\kdata\couplet\origin_data\vocabs'
data_file = r'F:\Resources\kdata\couplet\parse_data\couplets_train.txt'
test_file = r'F:\Resources\kdata\couplet\parse_data\couplets_test.txt'
log_dir = r'F:\Resources\kdata\couplet\logs'
batch_size = 32
epochs = 300
model_path = r"F:\Resources\kdata\couplet\models"
model_name = r'poetry.ckpt'
model_output = os.path.join(model_path, model_name)
wordcut = WordCut(file)
go_token = "<GO>"
pad_token = "<PAD>"
end_token = "<EOS>"
unk_token = "<UNK>"
max_len = 20
print("当前字符数量 ：", len(wordcut))
for token in ['end', 'unk', 'go', 'pad']:
    wordcut.add_word(eval(token + "_token"))
print("当前字符数量 ：", len(wordcut), "\n第一个字符", wordcut.idx2word(0))


def get_data(data_file, mode="eval"):
    inputs = []
    targets = []
    with open(data_file, 'r', encoding='utf-8') as fi:
        for line in fi:  # 不消化内存
            result = line.strip('\n\t')
            if not result:
                continue
            tinput, ttarget = line.strip('\n\t').split(" == ")
            if len(tinput) >= 20:
                continue
            res = []
            for word in tinput:
                if wordcut.reworddict.get(word, -1) == -1 and mode != "eval":
                    wordcut.add_word(word, 'append')
                res.append(wordcut.reworddict.get(word, 2))
            input = res
            res = []
            for word in ttarget:
                if wordcut.reworddict.get(word, -1) == -1 and mode != "eval":
                    wordcut.add_word(word, 'append')
                res.append(wordcut.reworddict.get(word, 2))
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


decoder_embedding_dim = 1024
encoder_embedding_dim = 1024

train_inputs, target_inputs = get_data(data_file)
val_inputs, target_val = get_data(data_file)
num_train = train_inputs.shape[0]
num_val = val_inputs.shape[0]
print('data count: ', num_train)

data_gen = gene_data(train_inputs, target_inputs, batch_size)
val_gen = gene_data(val_inputs, target_val, batch_size)
vocab_size = len(wordcut) + 2

print('data prepare!')


def get_lstm(rnn_size, rnn_dim):
    return tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.GRUCell(rnn_dim)
         for _ in range(rnn_size)])


def get_encoder(encoder_input, source_input_length, vocab_size,
                embedding_dim, rnn_size, rnn_dim):
    # 嵌入查询表shape:[词数量维度,嵌入向量维度]
    encode_embed = tf.get_variable('encoder_embedding', shape=[vocab_size, embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(encode_embed, encoder_input, name='encoder_embedding_input')

    with tf.name_scope("encoder"):
        encoder_cell = get_lstm(rnn_size, rnn_dim)
        encoder_output, encoder_state = dynamic_rnn(encoder_cell, embedding_inputs,
                                                    dtype=tf.float32,
                                                    sequence_length=source_input_length)
    print(f'encoder done!')
    return encoder_output, encoder_state


# 因为目前唐诗走绝句形式，所以长度一致，等到词了，再改成非固定长度
def get_decoder(decoder_target, encoder_output,
                encoder_state, source_input_length,
                target_input_length, embedding_dim,
                rnn_size, rnn_dim,
                batch_size, vocab_size,
                max_target_seq_length,
                use_beam=False, is_inference=False):
    with tf.variable_scope('decoder'):
        decoder_cell = get_lstm(rnn_size, rnn_dim)
        attentioner = seq2seq.BahdanauAttention(num_units=rnn_dim,  # 就是decoder的隐藏层维度
                                                memory=encoder_output,
                                                normalize=True,
                                                # 这又是针对encoder输入的长度
                                                memory_sequence_length=source_input_length)

        # attention在中间对decoder_cell的一层包裹
        attention_cell = seq2seq.AttentionWrapper(decoder_cell,
                                                  attentioner,
                                                  attention_layer_size=rnn_dim)
        decoder_init_state = attention_cell.zero_state(batch_size, dtype=tf.float32). \
            clone(cell_state=encoder_state)
        output_layer = tf.layers.Dense(vocab_size,  # FAQ:不用激活
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        if use_beam:
            pass
        else:
            # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
            ending = tf.strided_slice(decoder_target, [0, 0], [batch_size, -1], [1, 1])  # 感觉就是普通的切片法赋值啊
            # REW:通过go_token表示开始解码得到输出，相当于指示
            decoder_input = tf.concat([tf.fill([batch_size, 1], wordcut.word2idx("<GO>")), ending], 1)

            decode_embed = tf.get_variable('decoder_embedding', shape=[vocab_size, embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(decode_embed, decoder_input, name='decoder_embedding_input')

            # 定义数据解析helper
            train_helper = seq2seq.TrainingHelper(embedding_inputs,
                                                  # 针对decoder的长度
                                                  sequence_length=target_input_length)
            decoder = seq2seq.BasicDecoder(attention_cell,
                                           train_helper,
                                           initial_state=decoder_init_state,
                                           output_layer=output_layer)
            train_outputs, final_state, final_seq_length = \
                seq2seq.dynamic_decode(decoder,
                                       impute_finished=True,
                                       maximum_iterations=max_target_seq_length)
            # 重用decoder里的参数
            with tf.variable_scope('decoder', reuse=True):
                # 推断式数据解析helper,用前个输入 + 状态向量
                # start_tokens = tf.ones([batch_size, ], tf.int32) * wordcut.word2idx(go_token,1)
                start_tokens = tf.tile(tf.constant([wordcut.word2idx(go_token)], dtype=tf.int32), [batch_size],
                                       name='start_tokens')
                eval_helper = seq2seq.GreedyEmbeddingHelper(decode_embed,
                                                            start_tokens=start_tokens,
                                                            end_token=wordcut.word2idx(end_token))
                inference_decoder = seq2seq.BasicDecoder(attention_cell,
                                                         eval_helper,
                                                         decoder_init_state,
                                                         output_layer)
                predicting_decoder_output, _, _ = seq2seq.dynamic_decode(inference_decoder,
                                                                         impute_finished=True,
                                                                         maximum_iterations=max_target_seq_length)
            print('decoder done!')
            return train_outputs, predicting_decoder_output


def get_model(encoder_input, decoder_target,
              source_input_length, vocab_size,
              decoder_embedding_dim, encoder_embedding_dim,
              encoder_rnn_size, encoder_rnn_dim,
              decoder_rnn_size, decoder_rnn_dim,
              batch_size, global_steps,
              lr=1e-3):
    target_seq_length = tf.add(source_input_length, 1)  # label长度 - 1 = inputs长度
    max_target_seq_length = tf.reduce_max(target_seq_length, name='max_target_length')
    masks = tf.sequence_mask(target_seq_length, max_target_seq_length,
                             dtype=tf.float32,
                             name='masks'
                             )  # **
    encoder_output, encoder_state = get_encoder(encoder_input, source_input_length, vocab_size,
                                                encoder_embedding_dim,
                                                encoder_rnn_size,
                                                encoder_rnn_dim)
    train_outputs, predicting_decoder_output = get_decoder(decoder_target, encoder_output,
                                                           encoder_state,
                                                           source_input_length,
                                                           target_seq_length,
                                                           decoder_embedding_dim,
                                                           decoder_rnn_size, decoder_rnn_dim, batch_size,
                                                           vocab_size,
                                                           max_target_seq_length)
    decoder_logit = tf.identity(train_outputs.rnn_output, name='decoder_logit')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
    decoder_predict = tf.argmax(decoder_logit, axis=2, name='decoder_predict')
    loss = seq2seq.sequence_loss(decoder_logit,
                                 decoder_target,
                                 weights=masks)
    # 自然指数学习率衰减
    decayed_lr = tf.train.exponential_decay(lr,global_steps,30,0.9)
    optimizer = tf.train.AdamOptimizer(decayed_lr)
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
                        gradients, 0.5)
    train_op =optimizer.apply_gradients(zip(clipped_gradients,params),global_step=global_steps)
    print('model done')
    return predicting_logits, loss, train_op


def main():
    encoder_input = tf.placeholder(tf.int32, shape=[None, None], name='encoder_input')
    source_input_length = tf.placeholder(tf.int32, shape=[None], name='source_length')
    decoder_target = tf.placeholder(tf.int32, shape=[None, None], name='decoder_target')
    global_steps = tf.Variable(0, trainable=False)
    decoder_predict_op, loss_op, train_op = get_model(encoder_input, decoder_target,
                                                      source_input_length,
                                                      vocab_size,
                                                      decoder_embedding_dim=decoder_embedding_dim,
                                                      encoder_embedding_dim=encoder_embedding_dim,
                                                      encoder_rnn_size=6, encoder_rnn_dim=1024,
                                                      decoder_rnn_size=6, decoder_rnn_dim=1024,
                                                      batch_size=batch_size, global_steps=global_steps,
                                                      lr=1e-3)

    loss_su = tf.summary.scalar(tensor=loss_op, name='loss')
    saver = tf.train.Saver()
    print('prepare train!')
    count_step = 0
    graph = tf.get_default_graph()
    with tf.Session(graph=graph) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        summary_merge = tf.summary.merge_all()  # 合并所有summary
        f_summary = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
        for epoch in range(epochs):
            print(f'run epoch:{epoch}')
            loss_train = 0.
            acc_train = 0.
            loss_val = 0.
            acc_eval = 0.
            num_batch = num_train // batch_size
            for index in tqdm(range(num_batch)):
                inputs, targets = next(data_gen)
                # REW:找出每个序列实际长度
                seq_lengths = []
                for seq in inputs:
                    try:
                        length = list(seq).index(0)
                    except ValueError:
                        length = len(seq)
                    seq_lengths.append(length)

                predictions, _, loss, summary_tmp = sess.run([decoder_predict_op, train_op, loss_op, summary_merge],
                                                             feed_dict={
                                                                 encoder_input: inputs,
                                                                 source_input_length: np.array(seq_lengths),
                                                                 decoder_target: targets
                                                             })
                loss_train += loss.item()
                if index % 200 == 199:  # 每两百次 mini batch 打印损失
                    print(f'[epoch:{epoch} / {index} : loss:{loss_train // 200}')
                    loss_train = 0
                    f_summary.add_summary(summary_tmp, index + num_train * epoch)
                if index % 400 == 399:
                    print('predictions:',predictions)
                    print('save model')
                    saver.save(sess, model_output, global_step=index + num_train * epoch)
                    sid = random.randint(0, batch_size - 1)
                    input_text = wordcut.idxs2words(inputs[sid].tolist())
                    output_text = wordcut.idxs2words(predictions[sid].tolist())
                    target_text = wordcut.idxs2words(targets[sid].tolist()[:-1])
                    print("**************************")
                    print('src: ' + input_text)
                    print('output: ' + output_text)
                    print('target: ' + target_text)

            n_val_batch = num_val // batch_size
            for index in tqdm(range(n_val_batch)):
                inputs, targets = next(val_gen)
                seq_lengths = []
                for seq in inputs:
                    try:
                        length = list(seq).index(0)
                    except ValueError:
                        length = len(seq)
                    seq_lengths.append(length)

                _, loss, summary_tmp = sess.run([train_op, loss_op, summary_merge],
                                                feed_dict={
                                                    encoder_input: inputs,
                                                    source_input_length: np.array(seq_lengths),
                                                    decoder_target: targets
                                                })
                loss_val += loss.item()
                if index % 200 == 199:  # 每两百次 mini batch 打印损失
                    print(f'[epoch:{epoch} / {index} : eval loss:{loss_val // 200}')
                    loss_val = 0


if __name__ == "__main__":
    main()
    text = "我欲乘风来"
    text2 = "今朝有酒今朝醉"
    etext = wordcut.words2idxs(text)
    etext2 = wordcut.words2idxs(text2)
    text_list = [text, text2]
    etext_list = [etext, etext2]
    ckpt = model_path + "/" + model_name
    loaded_graph = tf.Graph().as_default()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(ckpt + ".meta")
        loader.restore(sess, ckpt)

        input_data = loaded_graph.get_tensor_by_name('encoder_input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_length:0')
        for txt, etxt in zip(text_list, etext_list):
            pre_logits = sess.run(logits, feed_dict={
                input_data: np.array([txt] * batch_size),
                source_sequence_length: [len(etxt)] * batch_size
            })
            print('前首输入:', txt)
            print('\nSource')
            print('  Input Words: {}', etxt)

            print('\n接')
            print('  Word 编号:       {}'.format([i for i in pre_logits if i != 0]))
            print('  Response Words: {}'.format(" ".join([wordcut.idx2word(i) for i in pre_logits if i != 0])))
            print('  Response Words: {}', wordcut.idxs2words(pre_logits))
