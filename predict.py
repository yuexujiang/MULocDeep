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
from keras.metrics import categorical_accuracy, binary_crossentropy
import matplotlib.pyplot as plt
import os
import calendar
import time
import argparse


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


def endpad(seqfile, pssmdir):
        new_pssms = []
        mask_seq = []
        ids=[]
        f = open(seqfile, "r")
        line = f.readline()
        while line != '':
            pssmfile = pssmdir + line[1:].strip() + "_pssm.txt"
            print("doing " + pssmfile + "\n")
            if os.path.exists(pssmfile):
                id = line.strip()[1:]
                ids.append(id)
                seq = f.readline().strip()
                seql = len(seq)
                pssm = readPSSM(pssmfile)
                pssm = pssm.astype(float)
                PhyChem = convertSampleToPhysicsVector_pca(seq)
                pssm = np.concatenate((PhyChem, pssm), axis=1)
                if seql <= 1000:
                    padnum = 1000 - seql
                    padmatrix = np.zeros([padnum, 25])
                    pssm = np.concatenate((pssm, padmatrix), axis=0)
                    new_pssms.append(pssm)
                    mask_seq.append(gen_mask_mat(seql, padnum))
                else:
                    pssm = np.concatenate((pssm[0:500, :], pssm[seql - 500:seql, :]), axis=0)
                    new_pssms.append(pssm)
                    mask_seq.append(gen_mask_mat(1000, 0))
            else:
                id = line.strip()[1:]
                ids.append(id)
                seq = f.readline().strip()
                seql = len(seq)
                pssm = convertSampleToBlosum62(seq)
                pssm = pssm.astype(float)
                PhyChem = convertSampleToPhysicsVector_pca(seq)
                pssm = np.concatenate((PhyChem, pssm), axis=1)
                if seql <= 1000:
                    padnum = 1000 - seql
                    padmatrix = np.zeros([padnum, 25])
                    pssm = np.concatenate((pssm, padmatrix), axis=0)
                    new_pssms.append(pssm)
                    mask_seq.append(gen_mask_mat(seql, padnum))
                else:
                    pssm = np.concatenate((pssm[0:500, :], pssm[seql - 500:seql, :]), axis=0)
                    new_pssms.append(pssm)
                    mask_seq.append(gen_mask_mat(1000, 0))
            line = f.readline()
        x = np.array(new_pssms)
        mask = np.array(mask_seq)
        return [x, mask,ids]


def mask_func(x):
    return x[0] * x[1]

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

map_lv2={1.0:"Cytoplasmic vesicle",1.1:"Cytoplasm, cytoskeleton",1.2:"Cytoplasm, myofibril",1.3:"Cytoplasm, cytosol",1.4:"Cytoplasm, perinuclear region",
         1.5:"Cytoplasm, cell cortex",1.6:"Cytoplasmic granule",1.7:"Cytoplasm, P-body",
         0.0:"Nucleus, nucleolus",0.1:"Nucleus, nucleoplasm",0.2:"Nucleus membrane",0.3:"Nucleus matrix",0.4:"Nucleus speckle",0.5:"Nucleus, PML body",
         0.6:"Nucleus, Cajal body",0.7:"Chromosome",
         3.0:"Mitochondrion inner membrane",3.1:"Mitochondrion intermembrane space",3.2:"Mitochondrion outer membrane",3.3:"Mitochondrion matrix",
         3.4:"Mitochondrion membrane",
         5.0:"Endoplasmic reticulum lumen",5.1:"Endoplasmic reticulum membrane",5.2:"Endoplasmic reticulum-Golgi intermediate compartment",
         5.3:"Microsome",5.4:"Sarcoplasmic reticulum",
         2.0:"Secreted, exosome",2.1:"Secreted, extracellular space",
         7.0:"Golgi apparatus, trans-Golgi network",7.1:"Golgi apparatus, cis-Golgi network",7.2:"Golgi apparatus membrane",7.3:"Golgi apparatus, Golgi stack membrane",
         4.0:"Membrane, clathrin-coated pit",4.1:"Membrane, coated pit",4.2:"Membrane raft",4.3:"Membrane, caveola",4.4:"Cell membrane",4.5:"Cell surface",
         8.0:"Lysosome membrane",
         9.0:"Peroxisome membrane",
         6.0:"Plastid, amyloplast",6.1:"Plastid, chloroplast membrane",6.2:"Plastid, chloroplast stroma",6.3:"Plastid, chloroplast thylakoid lumen",
         6.4:"Plastid, chloroplast thylakoid membrane"
         }

name=["Nucleus","Cytoplasm","Secreted","Mitochondrion","Membrane","Endoplasmic","Plastid","Golgi_apparatus","Lysosome","Peroxisome"]


def main():
    parser=argparse.ArgumentParser(description='MULocDeep: interpretable protein localization classifier at sub-cellular and sub-organellar levels')
    parser.add_argument('-input',  dest='inputfile', type=str, help='protein sequences in fasta format.', required=True)
    parser.add_argument('-output',  dest='outputdir', type=str, help='the name of the output directory.', required=True)
    parser.add_argument('-existPSSM', dest='existPSSM', type=str,
                        help='the name of the existing PSSM directory if there is one.', required=False, default="")
    parser.add_argument('--att', dest='drawATT', action='store_true',
                        help='Draw attention weight plot for each protein.', required=False)
    parser.add_argument('--no-att', dest='drawATT', action='store_false',
                        help='No attention weight plot.', required=False)
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    att_draw=args.drawATT
    inputfile=args.inputfile
    outputdir=args.outputdir
    if not outputdir[len(outputdir) - 1] == "/":
        outputdir = outputdir + "/"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    existPSSM=args.existPSSM
    if existPSSM!="":
       if not existPSSM[len(existPSSM) - 1] == "/":
         existPSSM = existPSSM + "/"

    if ((existPSSM=="")or(not os.path.exists(existPSSM))):
        ts = calendar.timegm(time.gmtime())
        pssmdir=outputdir+str(ts)+"_pssm/"
        if not os.path.exists(pssmdir):
          os.makedirs(pssmdir)
        process_input_train(inputfile,pssmdir)  # generate pssm
        [test_x, test_mask, test_ids] = endpad(inputfile, pssmdir)
    else:
        [test_x, test_mask, test_ids] = endpad(inputfile, existPSSM)

    pred_big = np.zeros((test_x.shape[0],10))
    att_matrix_N = np.zeros((8, test_x.shape[0], 1000))
    cross_pred_small = np.zeros((test_x.shape[0], 8, 10, 8))

    for foldnum in range(8):
        model_big, model_small = singlemodel(test_x)
        #model_big.load_weights('./cpu_models/fold' + str(foldnum) + '_big_lv1_acc-weights.hdf5')
        #cross_pred[:, foldnum] = model_big.predict([test_x, test_mask.reshape(-1, 1000, 1)])
        model_small.load_weights('./cpu_models/fold' + str(foldnum) + '_big_lv1_acc-weights.hdf5')
        cross_pred_small[:, foldnum]= model_small.predict([test_x, test_mask.reshape(-1, 1000, 1)])[0]
        model_att = Model(inputs=model_big.inputs, outputs=model_big.layers[-11].output[1])
        att_pred = model_att.predict([test_x, test_mask.reshape(-1, 1000, 1)])
        att_matrix_N[foldnum, :] = att_pred.sum(axis=1) / 41

    att_N = att_matrix_N.sum(axis=0) / 8

    pred_small = cross_pred_small.sum(axis=1) / 8
    pred_small_c = pred_small.copy()
    pred_big_c=pred_small_c.max(axis=-1)

    pred_small[pred_small >= 0.5] = 1.0
    pred_small[pred_small < 0.5] = 0.0

    for i in range(pred_small.shape[0]):
        index = pred_small_c[i].max(axis=-1).argmax()
        pred_big[i][index]=1.0
        if pred_small[i].sum() == 0:
            index = pred_small_c[i].max(axis=-1).argmax()
            index2 = pred_small_c[i][index].argmax()
            pred_small[i][index, index2] = 1.0

    #sub-cellular results
    f1 = open(outputdir+"sub_cellular_prediction.txt", "w")
    ind = 0
    for i in test_ids:
        f1.write(">" + i + ":\t")
        ans = ""
        for j in range(10):
            f1.write(name[j] + ":" + str(pred_big_c[ind, j]) + "\t")
            if pred_big[ind, j] == 1.0:
                if j == 0:
                    ans = ans + "Nucleus|"
                elif j == 1:
                    ans = ans + "Cytoplasm|"
                elif j == 2:
                    ans = ans + "Secreted|"
                elif j == 3:
                    ans = ans + "Mitochondrion|"
                elif j == 4:
                    ans = ans + "Membrane|"
                elif j == 5:
                    ans = ans + "Endoplasmic|"
                elif j == 6:
                    ans = ans + "Plastid|"
                elif j == 7:
                    ans = ans + "Golgi_apparatus|"
                elif j == 8:
                    ans = ans + "Lysosome|"
                elif j == 9:
                    ans = ans + "Peroxisome|"
        f1.write("prediction:" + ans + "\n")
        ind = ind + 1
    f1.close()

    # sub-organellar results
    f1 = open(outputdir+"sub_organellar_prediction.txt", "w")
    ind = 0
    for i in test_ids:
        f1.write(">" + i + ":\t")
        ans = ""
        for j in range(10):
            for z in range(8):
                key = float(str(j) + "." + str(z))
                if key in map_lv2:
                    f1.write(map_lv2[key] + ":" + str(pred_small_c[ind, j, z]) + "\t")
                    if pred_small[ind, j, z] == 1.0:
                        ans = ans + map_lv2[key] + "|"
        f1.write("prediction:" + ans + "\n")
        ind = ind + 1
    f1.close()

    # output attention weights
    f1 = open(outputdir+"attention_weights.txt", "w")
    ind=0
    for i in test_ids:
        end = int(test_mask[ind].sum())
        f1.write(">" + i + "\n")
        for p in att_N[ind][0:end]:
            f1.write(str(p) + " ")
        f1.write("\n")
        ind=ind+1
    f1.close()

    if att_draw:
        i = 0
        for p in test_ids:
          j = i * 2 + 1
          ind = int(test_mask[i].sum())
          f = open(inputfile)
          list_seq = f.readlines()
          seq = list(list_seq[j].strip())[:ind]
          plt.xticks(np.linspace(1, ind, ind), seq)
          plt.plot(np.linspace(1, ind, ind), att_N[i][:ind])
          plt.savefig(outputdir+p+'.png')
          i=i+1


if __name__ == "__main__":
    main()
