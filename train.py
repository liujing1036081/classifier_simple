# -*- coding: utf-8 -*-

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Dense, Dropout, Flatten, AveragePooling2D, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from data_load import excel_table_byname


# 单字切分,文本序列化预处理
def data_preprocess(list):
    res_list = []
    for text in list:
        split_list = []
        for char in text:
            split_list.append(char)  # 单个字符列表
        str = ' '.join(split_list)
        res_list.append(str)  # 单个字符以空格隔开的字符串（tokenizer要求  的输入格式）
    max_len = max([len(x) for x in list])
    print(u'最长句子长度max_len: ', max_len)

    tokenizer = Tokenizer(num_words=None)  # 出现次数少于num_words的单词去掉
    tokenizer.fit_on_texts(res_list)
    sequences = tokenizer.texts_to_sequences(res_list)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=max_len)
    print(word_index)
    return data, word_index, max_len


# 准备embedding层 使用预训练好的词向量
def pre_embedd(wordVecFile, word_index):
    embeddings_index = {}
    f = open(wordVecFile)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))  # 词向量文档中词向量的个数

    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():  # i是从1开始的，所以上面要加1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# 构建model
def build_model(word_index, embedding_matrix, max_len):
    embedding_layer = Embedding(len(word_index) + 1,
                                100,
                                weights=[embedding_matrix],
                                input_length=max_len,
                                trainable=False
                                )
    sequence_input = Input(shape=(max_len,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    # layer = GlobalAveragePooling1D()(embedded_sequences)
    layer = Dense(128, activation='relu')(embedded_sequences)
    layer = Dropout(0.2)(layer)
    layer = Flatten()(layer)
    layer = Dense(128, activation='relu')(layer)
    preds = Dense(3, activation='softmax')(layer)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model


def main():
    # filename = 'wh-20180324.xlsx'
    # by_name = u'page9'

    filename = 'test2.xlsx'
    by_name = u'Sheet1'
    textList, labels = excel_table_byname(filename, 0, by_name)

    data, word_index, max_len = data_preprocess(textList)
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    print('-----------')
    embedding_matrix = pre_embedd('CNLetterVec.100d.txt', word_index)
    model = build_model(word_index, embedding_matrix, max_len)
    batch_size = 5
    nb_epoch = 40
    VALIDATION_SPLIT = 0.2

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('training.....')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(x_val, y_val))


if __name__ == "__main__":
    main()
