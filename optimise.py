import numpy as np
import keras
from keras import layers
from keras import optimizers
from keras.layers import *
from keras.models import Model
from hier_attention_mask import Attention
from keras import backend as K
#from Bio.Blast.Applications import NcbipsiblastCommandline
#from Bio import SeqIO
from keras.metrics import categorical_accuracy, binary_crossentropy
import matplotlib.pyplot as plt
import os
import calendar
import time
import argparse
import sys
from utils import *
import gc
from GPyOpt.methods import BayesianOptimization

def endpad(seqfile, labelfile, pssmdir="", npzfile = ""):
    if not os.path.exists(npzfile):
        new_pssms = []
        labels = []
        mask_seq = []
        ids=[]
        f = open(seqfile, "r")
        f2 = open(labelfile, "r")
        line = f.readline()
        while line != '':
            pssmfile = pssmdir + line[1:].strip() + "_pssm.txt"
            if line[0] == '>':
                label = f2.readline().strip()
            id = line.strip()[1:]
            ids.append(id)
            labels.append(label)
            seq = f.readline().strip()
            seql = len(seq)
            if os.path.exists(pssmfile):
                print("found " + pssmfile + "\n")
                pssm = readPSSM(pssmfile)
            else:
                print("using Blosum62\n")
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
        y = [convertlabels_to_categorical(i) for i in labels]
        y = np.array(y)
        mask = np.array(mask_seq)
        np.savez(npzfile, x=x, y=y, mask=mask, ids=ids)
        return [x, y, mask,ids]
    else:
        mask = np.load(npzfile)['mask']
        x = np.load(npzfile)['x']
        y = np.load(npzfile)['y']
        ids=np.load(npzfile)['ids']
        return [x, y, mask,ids]


bds = [{'name': 'hidden_dim', 'type': 'continuous', 'domain': (32, 490)},
       {'name': 'da', 'type': 'continuous', 'domain': (32, 430)},
       {'name': 'r', 'type': 'continuous', 'domain': (16, 64)},
       {'name': 'W_regularizer', 'type': 'continuous', 'domain': (0.00001, 0.001)},
       {'name': 'Att_regularizer_weight', 'type': 'continuous', 'domain': (0.00001, 0.001)},
       {'name': 'drop_per', 'type': 'continuous', 'domain': (0.1, 0.75)},
       {'name': 'drop_hid', 'type': 'continuous', 'domain': (0.1, 0.75)}]

train_data=[]
val_data=[]
for i in range(8):
    train_data.append(endpad( "./data/deeploc_40nr_8folds/deeploc_40nr_train_fold"+str(i)+"_seq", "./data/deeploc_40nr_8folds/deeploc_40nr_train_fold"+str(i)+"_label","./data/deeploc_train_pssm/","./data/deeploc_40nr_8folds/deeploc_40nr_train_S_1000_fold"+str(i)+".npz"))
    val_data.append(endpad("./data/deeploc_40nr_8folds/deeploc_40nr_val_fold"+str(i)+"_seq", "./data/deeploc_40nr_8folds/deeploc_40nr_val_fold"+str(i)+"_label","./data/deeploc_train_pssm/","./data/deeploc_40nr_8folds/deeploc_40nr_val_S_1000_fold"+str(i)+".npz"))

for foldnum in range(8):
    runtimes = 0
    train_x=train_data[foldnum][0]
    train_y=train_data[foldnum][1]
    train_mask=train_data[foldnum][2]
    val_x=val_data[foldnum][0]
    val_y=val_data[foldnum][1]
    val_mask=val_data[foldnum][2]
    def score(parameters, foldnum=foldnum):
       try:
        print("optimizing for foldnum:" + str(foldnum))
        output = open("./deeplocdata_phyChpssm_batchnorm_40nr/costum_record_fold" + str(foldnum) + ".txt", 'a')
        parameters = parameters[0]
        dim_lstm = int(parameters[0])
        da = int(parameters[1])
        r = int(parameters[2])
        W_regularizer = float(parameters[3])
        Att_regularizer_weight = float(parameters[4])
        drop_per = float(parameters[5])
        drop_hid = float(parameters[6])
        batch_size = 128
        lr = 0.0005
        global runtimes
        print(parameters)
        output.write("At iteration:" + str(runtimes) + "\n")
        output.write("\t".join([str(x) for x in parameters]) + "\n")

        input = Input(shape=(train_x.shape[1:]), name="Input")  # input's shape=[?,seq_len,encoding_dim]
        input_mask = Input(shape=([train_x.shape[1], 1]), dtype='float32')  # (batch_size,max_len,1)
        l_indrop = layers.Dropout(drop_per)(input)
        mask_input = []
        mask_input.append(l_indrop)
        mask_input.append(input_mask)
        mask_layer1 = Lambda(mask_func)(mask_input)
        x1 = layers.Bidirectional(
            CuDNNLSTM(dim_lstm, kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
                      return_sequences=True), merge_mode='sum')(mask_layer1)  # [?,seq_len,dim_lstm]
        x1bn = layers.BatchNormalization()(x1)
        x1d = layers.Dropout(drop_hid)(x1bn)
        mask_input = []
        mask_input.append(x1d)
        mask_input.append(input_mask)
        mask_layer2 = Lambda(mask_func)(mask_input)
        x2 = layers.Bidirectional(
            CuDNNLSTM(dim_lstm, kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
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
                        W1_constraint=None, W2_constraint=None, return_attention=False,
                        attention_regularizer_weight=Att_regularizer_weight)(
            layers.concatenate([mask_layer3, input_mask]))  # att=[?,r,dim_lstm]
        attbn = layers.BatchNormalization()(att)
        att_drop = layers.Dropout(drop_hid)(attbn)
        # lev1 = CapsuleLayer(num_capsule=10, dim_capsule=19, routings=3, name='lev1')(
        #     att_drop)  # digitcaps'shape=[?,num_capsule,dim_capsule2]
        # lev1_output = Length(name='capsnet')(lev1)  # shape=[?,num_capsule]
        flat = layers.Flatten()(att_drop)
        flat_drop = layers.Dropout(drop_hid)(flat)
        lev1_output = layers.Dense(units=10, kernel_initializer='orthogonal', activation=None)(flat_drop)
        lev1_output_bn = layers.BatchNormalization()(lev1_output)
        lev1_output_act = layers.Activation('softmax')(lev1_output_bn)
        model = Model(inputs=[input, input_mask], outputs=lev1_output_act)
        adam = optimizers.Adam(lr=lr)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        fitHistory_batch = model.fit([train_x, train_mask.reshape(-1, 1000, 1)], getTrue4out1(train_y),
                                     batch_size=batch_size, epochs=40,
                                     validation_data=([val_x, val_mask.reshape(-1, 1000, 1)], getTrue4out1(val_y)),verbose=1)
        if np.isnan(fitHistory_batch.history['val_loss']).sum() > 0:
                    score = 0
        else:
                    score = np.array(fitHistory_batch.history['val_acc'])
        gc.collect()
        score = np.max(score)
        K.clear_session()
        output.write("score=" + str(score) + "\n")
        output.close()
        runtimes += 1
       except:
        return 0
       else:
        return score
    optimizer = BayesianOptimization(f=score,
                                     domain=bds,
                                     model_type='GP',
                                     acquisition_type='EI',
                                     acquisition_jitter=0.05,
                                     exact_feval=True,
                                     maximize=True)
    optimizer.run_optimization(max_iter=150, report_file="fold_"+str(foldnum)+"_opti_report.txt")
    # optimizer.plot_convergence()
    best_x = optimizer.x_opt
    best_y = optimizer.fx_opt
    output = open("./deeplocdata_phyChpssm_batchnorm_40nr/costum_record_" + str(foldnum) + ".txt", 'a')
    output.write("best_x:" + ",".join([str(x) for x in best_x]) + "\n")
    output.write("best_y:" + str(best_y) + "\n")
    output.close()
    optimizer.plot_convergence(filename="./deeplocdata_phyChpssm_batchnorm_40nr/"+str(foldnum))
    # output2 = open("nocnncapsule_optimizedparameters_fold.txt", 'a')
    # output2.write("Fold:" + str(foldnum) + "\n")
    # output2.write(",".join([str(x) for x in best_x]) + "\n")
    # output2.close()
    del score
    del optimizer