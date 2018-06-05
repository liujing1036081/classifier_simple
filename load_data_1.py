# -*- coding: utf-8 -*-

import xdrlib, sys
import xlrd
import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Dense, Dropout, Flatten, AveragePooling2D, GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split


def open_excel(file):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception, e:
        print str(e)


def excel_table_byname(file, colnameindex, by_name):
    data = open_excel(file)
    table = data.sheet_by_name(by_name)
    nrows = table.nrows  # 行数
    colnames = table.row_values(colnameindex)  # 某一行数据
    # print(colnames)
    text_list = []
    label_list = []
    label_name=[]
    for rownum in range(1, nrows):
        row = table.row_values(rownum)
        if row:
            # for i in range(len(colnames)):
            if colnames[1] == u'问句' and row[5] != '':
                text_list.append(row[1])
                if row[5] == u'对主播的印象':
                    label_list.append(int(0))
                if row[5] == u'用户对自己的描述':
                    label_list.append(int(1))
                if row[5] == u'直播相关问题':
                    label_list.append(int(2))
    # print(label_list)

    return text_list, label_list


# 单字切分,文本序列化
def data_preprocess(list):
    res_list = []
    for text in list:
        split_list = []
        for char in text:
            split_list.append(char)
        str = ' '.join(split_list)
        res_list.append(str)
    max_len = max([len(x) for x in list])
    print('max_len: ',max_len)

    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(res_list)
    sequences = tokenizer.texts_to_sequences(res_list)
    # print(sequences)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_len)

    return data, word_index, max_len


# 准备embedding层
def pre_embed(file, word_index):
    embeddings_index = {}
    f = open(file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    # print(embeddings_index['是'])

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def main():
    # filename = 'wh-20180324.xlsx'
    filename = 'wh-20180324.xlsx'
    by_name = u'page9'
    textList, labels = excel_table_byname(filename, 0, by_name)

    data, word_index, max_len = data_preprocess(textList)
    labels = to_categorical(np.asarray(labels))
    # labels = labels.reshape(len(labels), -1)
    # label = (np.transpose(labels))
    print('Shape of data tensor:', data.shape)
    # print('Shape of label tensor:', labels)
    # print(len(labels))

    print('-----------')

    embedding_matrix = pre_embed('CNLetterVec.100d.txt', word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                                100,
                                weights=[embedding_matrix],
                                input_length=max_len,
                                trainable=False
                                )
    sequence_input = Input(shape=(max_len,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    # print(embedded_sequences)
    print('*********')
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
    # print('编译结束....')

    # x_train = data
    # y_train = labels
    batch_size = 5
    nb_epoch = 40
    VALIDATION_SPLIT = 0.2

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    try:
        labels = labels[indices]
    except IndexError as e:
        print(e)
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    # print(x_train)
    # print(y_train)
    #
    # print(x_val)
    #
    # print(y_val)

    print('training.....')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(x_val, y_val))


if __name__ == "__main__":
    main()
