import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.Session(config=config)
import numpy as np
import keras
from keras import layers
from keras import optimizers
from keras.layers import *
from keras.models import Model
from hier_attention_mask import Attention
from keras import backend as K
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio import SeqIO
import sys

def convertSampleToPhysicsVector_pca(seq):
    """
    Convertd the raw data to physico-chemical property
    PARAMETER
    seq: "MLHRPVVKEGEWVQAGDLLSDCASSIGGEFSIGQ" one fasta seq
        X denoted the unknow amino acid.
    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    """
    letterDict = {}
    letterDict["A"] = [0.008, 0.134, -0.475, -0.039, 0.181]
    letterDict["R"] = [0.171, -0.361, 0.107, -0.258, -0.364]
    letterDict["N"] = [0.255, 0.038, 0.117, 0.118, -0.055]
    letterDict["D"] = [0.303, -0.057, -0.014, 0.225, 0.156]
    letterDict["C"] = [-0.132, 0.174, 0.070, 0.565, -0.374]
    letterDict["Q"] = [0.149, -0.184, -0.030, 0.035, -0.112]
    letterDict["E"] = [0.221, -0.280, -0.315, 0.157, 0.303]
    letterDict["G"] = [0.218, 0.562, -0.024, 0.018, 0.106]
    letterDict["H"] = [0.023, -0.177, 0.041, 0.280, -0.021]
    letterDict["I"] = [-0.353, 0.071, -0.088, -0.195, -0.107]
    letterDict["L"] = [-0.267, 0.018, -0.265, -0.274, 0.206]
    letterDict["K"] = [0.243, -0.339, -0.044, -0.325, -0.027]
    letterDict["M"] = [-0.239, -0.141, -0.155, 0.321, 0.077]
    letterDict["F"] = [-0.329, -0.023, 0.072, -0.002, 0.208]
    letterDict["P"] = [0.173, 0.286, 0.407, -0.215, 0.384]
    letterDict["S"] = [0.199, 0.238, -0.015, -0.068, -0.196]
    letterDict["T"] = [0.068, 0.147, -0.015, -0.132, -0.274]
    letterDict["W"] = [-0.296, -0.186, 0.389, 0.083, 0.297]
    letterDict["Y"] = [-0.141, -0.057, 0.425, -0.096, -0.091]
    letterDict["V"] = [-0.274, 0.136, -0.187, -0.196, -0.299]
    letterDict["X"] = [0, -0.00005, 0.00005, 0.0001, -0.0001]
    letterDict["-"] = [0, 0, 0, 0, 0, 1]
    AACategoryLen = 5  # 6 for '-'
    l = len(seq)
    probMatr = np.zeros((l, AACategoryLen))
    AANo = 0
    for AA in seq:
        if not AA in letterDict:
            probMatr[AANo] = np.full(AACategoryLen, 0)
        else:
            probMatr[AANo] = letterDict[AA]

        AANo += 1
    return probMatr


def convertSampleToBlosum62(seq):
    letterDict = {}
    letterDict["A"] = [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0]
    letterDict["R"] = [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3]
    letterDict["N"] = [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3]
    letterDict["D"] = [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3]
    letterDict["C"] = [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1]
    letterDict["Q"] = [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2]
    letterDict["E"] = [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2]
    letterDict["G"] = [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3]
    letterDict["H"] = [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3]
    letterDict["I"] = [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3]
    letterDict["L"] = [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1]
    letterDict["K"] = [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2]
    letterDict["M"] = [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1]
    letterDict["F"] = [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1]
    letterDict["P"] = [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2]
    letterDict["S"] = [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2]
    letterDict["T"] = [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0]
    letterDict["W"] = [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3]
    letterDict["Y"] = [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1]
    letterDict["V"] = [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]
    AACategoryLen = 20  # 6 for '-'
    l = len(seq)
    probMatr = np.zeros((l, AACategoryLen))
    AANo = 0
    for AA in seq:
        if not AA in letterDict:
            probMatr[AANo] = np.full(AACategoryLen, 0)
        else:
            probMatr[AANo] = letterDict[AA]

        AANo += 1
    return probMatr


def convertlabels_to_categorical(seq):
    label = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.]])
    for index in seq.split(";"):
        i = int(index.split(".")[0])
        j = int(index.split(".")[1])
        label[i][j] = 1.0
    return label


def readPSSM(pssmfile):
    pssm = []
    with open(pssmfile, 'r') as f:
        count = 0
        for eachline in f:
            count += 1
            if count <= 3:
                continue
            if not len(eachline.strip()):
                break
            line = eachline.split()
            pssm.append(line[2: 22])  # 22:42
    return np.array(pssm)


def getTrue4out1(y):  # input [?, 10, 8]  output [?, 10]  elements are 0 or 1
    if len(y.shape) == 2:
        label = y.sum(axis=1)
        label[label >= 1] = 1
    if len(y.shape) == 3:
        label = y.sum(axis=2)
        label[label >= 1] = 1
    return label


def getTrue4out2(y):  # [?,10,8]
    label = []
    if len(y.shape) == 2:
        x = y
        a = []
        a.extend(x[0][0:8])
        a.extend(x[1][0:8])
        a.extend(x[2][0:2])
        a.extend(x[3][0:5])
        a.extend(x[4][0:6])
        a.extend(x[5][0:5])
        a.extend(x[6][0:5])
        a.extend(x[7][0:4])
        a.extend(x[8][0:1])
        a.extend(x[9][0:1])
        label.append(a)
    else:
        for x in y:
            a = []
            a.extend(x[0][0:8])
            a.extend(x[1][0:8])
            a.extend(x[2][0:2])
            a.extend(x[3][0:5])
            a.extend(x[4][0:6])
            a.extend(x[5][0:5])
            a.extend(x[6][0:5])
            a.extend(x[7][0:4])
            a.extend(x[8][0:1])
            a.extend(x[9][0:1])
            label.append(a)
    return np.array(label)


def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
            axis=-1)

    return weighted_loss


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.1 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def gen_mask_mat(num_want, num_mask):
    seq_want = np.ones(num_want)
    seq_mask = np.zeros(num_mask)
    seq = np.concatenate([seq_want, seq_mask])
    return seq

def mask_func(x):
    return x[0] * x[1]


def singlemodel(train_x):
    [dim_lstm, da, r, W_regularizer, Att_regularizer_weight, drop_per, drop_hid, lr] = [
        180, 369, 41, 0.00001,0.0007159, 0.1, 0.1, 0.0005]
    input = Input(shape=(train_x.shape[1:]), name="Input")  # input's shape=[?,seq_len,encoding_dim]
    input_mask = Input(shape=([train_x.shape[1], 1]), dtype='float32')  # (batch_size,max_len,1)
    l_indrop = layers.Dropout(drop_per)(input)
    mask_input = []
    mask_input.append(l_indrop)
    mask_input.append(input_mask)
    mask_layer1 = Lambda(mask_func)(mask_input)
    x1 = layers.Bidirectional(LSTM(dim_lstm, kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
                                        return_sequences=True), merge_mode='sum')(mask_layer1)  # [?,seq_len,dim_lstm]
    x1bn = layers.BatchNormalization()(x1)
    x1d = layers.Dropout(drop_hid)(x1bn)
    mask_input = []
    mask_input.append(x1d)
    mask_input.append(input_mask)
    mask_layer2 = Lambda(mask_func)(mask_input)
    x2 = layers.Bidirectional(LSTM(dim_lstm, kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
                                        return_sequences=True), merge_mode='sum')(mask_layer2)  # [?,seq_len,dim_lstm]
    x2bn = layers.BatchNormalization()(x2)
    x2d = layers.Dropout(drop_hid)(x2bn)
    mask_input = []
    mask_input.append(x2d)
    mask_input.append(input_mask)
    mask_layer3 = Lambda(mask_func)(mask_input)
    att = Attention(hidden=dim_lstm, da=da, r=r, init='glorot_uniform', activation='tanh',
                    W1_regularizer=keras.regularizers.l2(W_regularizer),
                    W2_regularizer=keras.regularizers.l2(W_regularizer),
                    W1_constraint=None, W2_constraint=None, return_attention=True,
                    attention_regularizer_weight=Att_regularizer_weight)(
        layers.concatenate([mask_layer3, input_mask]))  # att=[?,r,dim_lstm]
    attbn = layers.BatchNormalization()(att[0])
    att_drop = layers.Dropout(drop_hid)(attbn)
    flat = layers.Flatten()(att_drop)
    flat_drop = layers.Dropout(drop_hid)(flat)
    lev2_output = layers.Dense(units=10 * 8, kernel_initializer='orthogonal', activation=None)(flat_drop)
    lev2_output_reshape = layers.Reshape([10, 8, 1])(lev2_output)
    lev2_output_bn = layers.BatchNormalization()(lev2_output_reshape)
    lev2_output_pre = layers.Activation('sigmoid')(lev2_output_bn)
    lev2_output_act = layers.Reshape([10,8],name='lev2')(lev2_output_pre)
    final = layers.MaxPooling2D(pool_size=[1, 8], strides=None, padding='same', data_format='channels_last')(
        lev2_output_pre)
    final = layers.Reshape([-1,],name='1ev1')(final)
    model_small = Model(inputs=[input, input_mask], outputs=[lev2_output_act, final])
    model_big = Model(inputs=[input, input_mask], outputs=[final])
    adam = optimizers.Adam(lr=lr)
    model_big.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model_small.compile(optimizer=adam, loss=['binary_crossentropy', 'binary_crossentropy'], metrics = ['accuracy'])
    model_big.summary()
    model_small.summary()
    return model_big, model_small

def process_input_train(seq_file,dir):
    processed_num=0
    if not os.path.exists(dir):
        os.mkdir(dir)
    for seq_record in list(SeqIO.parse(seq_file, "fasta")):
        processed_num+=1
        print("in loop, processing"+str(processed_num)+"\n")
        pssmfile=dir+seq_record.id+"_pssm.txt"
        inputfile=dir+'tempseq.fasta'
        seql = len(seq_record)
        if not os.path.exists(pssmfile):
            if os.path.exists(inputfile):
                os.remove(inputfile)
            SeqIO.write(seq_record, inputfile, 'fasta')
            psiblast_cline = NcbipsiblastCommandline(query=inputfile, db='./db/swissprot/swissprot', num_iterations=3,
                                                     evalue=0.001, out_ascii_pssm=pssmfile, num_threads=4)
            stdout, stderr = psiblast_cline()

def process_input_user(seq_file,dir):
    processed_num=0
    if not os.path.exists(dir):
        os.mkdir(dir)
    index=0
    for seq_record in list(SeqIO.parse(seq_file, "fasta")):
        processed_num+=1
        print("in loop, processing"+str(processed_num)+"\n")
        pssmfile=dir+str(index)+"_pssm.txt"
        inputfile=dir+'tempseq.fasta'
        seql = len(seq_record)
        if not os.path.exists(pssmfile):
            if os.path.exists(inputfile):
                os.remove(inputfile)
            SeqIO.write(seq_record, inputfile, 'fasta')
            try:
              psiblast_cline = NcbipsiblastCommandline(query=inputfile, db='./db/swissprot/swissprot', num_iterations=3,
                                                     evalue=0.001, out_ascii_pssm=pssmfile, num_threads=4)
              stdout, stderr = psiblast_cline()
            except:
              print("invalid protein: "+seq_record)

        index=index+1

def var_model(train_x):
    [dim_lstm, da, r, W_regularizer, Att_regularizer_weight, drop_per, drop_hid, lr] = [
        180, 369, 41, 0.00001,0.0007159, 0.1, 0.1, 0.0005]
    input = Input(shape=(train_x.shape[1:]), name="Input")  # input's shape=[?,seq_len,encoding_dim]
    input_mask = Input(shape=([train_x.shape[1], 1]), dtype='float32')  # (batch_size,max_len,1)
    l_indrop = layers.Dropout(drop_per)(input)

    mask_input = []
    mask_input.append(l_indrop)
    mask_input.append(input_mask)
    mask_layer1 = Lambda(mask_func)(mask_input)

    x1 = layers.Bidirectional(CuDNNLSTM(dim_lstm, kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
                                        return_sequences=True), merge_mode='sum')(mask_layer1)  # [?,seq_len,dim_lstm]
    x1bn = layers.BatchNormalization()(x1)
    x1d = layers.Dropout(drop_hid)(x1bn)
    mask_input = []
    mask_input.append(x1d)
    mask_input.append(input_mask)
    mask_layer2 = Lambda(mask_func)(mask_input)

    x2 = layers.Bidirectional(CuDNNLSTM(dim_lstm, kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
                                        return_sequences=True), merge_mode='sum')(mask_layer2)  # [?,seq_len,dim_lstm]
    x2bn = layers.BatchNormalization()(x2)
    x2d = layers.Dropout(drop_hid)(x2bn)
    mask_input = []
    mask_input.append(x2d)
    mask_input.append(input_mask)
    mask_layer3 = Lambda(mask_func)(mask_input)

    att = Attention(hidden=dim_lstm, da=da, r=r, init='glorot_uniform', activation='tanh',
                    W1_regularizer=keras.regularizers.l2(W_regularizer),
                    W2_regularizer=keras.regularizers.l2(W_regularizer),
                    W1_constraint=None, W2_constraint=None, return_attention=True,
                    attention_regularizer_weight=Att_regularizer_weight)(
        layers.concatenate([mask_layer3, input_mask]))  # att=[?,r,dim_lstm]

    attbn = layers.BatchNormalization()(att[0])
    att_drop = layers.Dropout(drop_hid)(attbn)
    flat = layers.Flatten()(att_drop)
    flat_drop = layers.Dropout(drop_hid)(flat)
    lev1_output = layers.Dense(units=10, kernel_initializer='orthogonal', activation=None)(flat_drop)
    lev1_output_bn = layers.BatchNormalization()(lev1_output)
    lev1_output_act = layers.Activation('softmax')(lev1_output_bn)
    model = Model(inputs=[input, input_mask], outputs=lev1_output_act)
    adam = optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model