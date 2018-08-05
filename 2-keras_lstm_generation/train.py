# 此脚本是用于训练，可直接运行。
# 需要安装keras、jieba、hdf5等依赖
# 修改原网站:https://github.com/shiwusong/keras_lstm_generation
import jieba
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding, SimpleRNN
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# 使用jieba进行分词
f = open('new_wangfeng.txt', 'r', encoding='utf-8')
all_str = f.read().replace('\n', '').replace(' ', '')  # 去除空格
f.close()
cut_list = jieba.cut(all_str)  # <generator object Tokenizer.cut at 0x000000DCE8118938>,出来并不是list
seg_list = []  # 分词后的文本数据,22464个词
for c in cut_list:
    seg_list.append(c)

vocab = sorted(list(set(seg_list)))  # 2841
word_to_int = dict((w, i) for i, w in enumerate(vocab))
int_to_word = dict((i, w) for i, w in enumerate(vocab))

n_words = len(seg_list)  # 总词量,未去重，22464
n_vocab = len(vocab)  # 词表长度,去重,2841
print('总词汇量：', n_words)
print('词表长度：', n_vocab)

seq_length = 100
dataX = []  # 22464-100=22364个list,每个list是长度为100的数
dataY = []  # 22364,形如[[1922], [1016], [238]]
for i in range(0, n_words - seq_length, 1):  # 句子长度,在下面循环中其实长度为101
    seq_in = seg_list[i:i + seq_length + 1]
    dataX.append([word_to_int[word] for word in seq_in])

# 乱序
np.random.shuffle(dataX)
for i in range(len(dataX)):
    dataY.append([dataX[i][seq_length]])  # 每个seq最后一个词,也就是训练数据是前100个,标签是最后一个
    dataX[i] = dataX[i][:seq_length]  # 句子长度由101变成100

n_simples = len(dataX)
print('样本数量：', n_simples)
X = np.reshape(dataX, (n_simples, seq_length))  # <class 'tuple'>: (22364, 100)
y = np_utils.to_categorical(dataY)  # <class 'tuple'>: (22364, 2841), one-hot, 每行全是0,1向量, 2841为去重后词表长度

# 网络结构
print('开始构建网络')
model = Sequential()
model.add(Embedding(n_vocab, 512, input_length=seq_length))  # 映射为512维embedding
model.add(LSTM(512, input_shape=(seq_length, 512), return_sequences=True))
# model.add(Dropout(0.2))  # 丢失层，防止过拟合
model.add(LSTM(1024))
# model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
# print('加载上一次的网络')
# filename = 'weights-improvement=05-5.901622.hdf5'
# model.load_weights(filename)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam)

# 存储每一次迭代的网络权重
filepath = "weights-improvement={epoch:02d}-{loss:4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
print('开始训练')
model.fit(X, y, epochs=30, batch_size=100, callbacks=callbacks_list, verbose=1)


"""
笔记：程序训练流程
首先将原文本分词,得到词典;
将22464个词(未去重)划分(22464-100=)22364个长度为101的序列,前100个词(index,并不是one-hot)用作训练数据,最后一个词的one-hot作为输出,也就是标签
"""
