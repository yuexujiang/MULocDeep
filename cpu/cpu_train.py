import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.Session(config=config)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import math
from itertools import product
import argparse
import sys
#sys.path.append('./review_round4/code/')                   ##modify
from cpu_utils import *
import calendar
import time


def endpad(seqfile, labelfile, pssmdir="", npzfile = ""):                    #modify
    if not os.path.exists(npzfile):                    #modify
        new_pssms = []
        labels = []
        mask_seq = []
        ids=[]
        f = open(seqfile, "r")
        f2 = open(labelfile, "r")
        line = f.readline()
        while line != '':
            pssmfile = pssmdir + line[1:].strip() + "_pssm.txt"
            print("doing " + pssmfile + "\n")
            if line[0] == '>':
                label = f2.readline().strip()
            id = line.strip()[1:]
            ids.append(id)
            labels.append(label)
            seq = f.readline().strip()
            seql = len(seq)
            if os.path.exists(pssmfile):
                pssm = readPSSM(pssmfile)
            else:
                print("using Blosum62\n")
                pssm = convertSampleToBlosum62(seq)
            pssm = pssm.astype(float)
            PhyChem = convertSampleToPhysicsVector_pca(seq)
            pssm = np.concatenate((PhyChem, pssm), axis=1)
            print(id)
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







def train_MULocDeep(lv1_dir,lv2_dir,pssm_dir,output_dir,foldnum):
    # get small data
    [train_x, train_y, train_mask, train_ids] = endpad(
        lv2_dir+"lv2_train_fold" + str(foldnum) + "_seq",
        lv2_dir+"lv2_train_fold" + str(foldnum) + "_lab",
        pssm_dir,
        "./review_2ndRound_6/data/npzfiles/lv2_train_fold"+str(foldnum)+"_seq.npz")
    [val_x, val_y, val_mask,val_ids] = endpad(
        lv2_dir+"lv2_val_fold" + str(foldnum) + "_seq",
        lv2_dir+"lv2_val_fold" + str(foldnum) + "_lab",
        pssm_dir,
        "./review_2ndRound_6/data/npzfiles/lv2_val_fold"+str(foldnum)+"_seq.npz")

    # get big data, using deeploc
    # [train_x_big, train_y_big, train_mask_big,train_ids_big] = endpad(
    #     "./data/deeploc_40nr_8folds/deeploc_40nr_train_fold" + str(foldnum) + "_seq",
    #     "./data/deeploc_40nr_8folds/deeploc_40nr_train_fold" + str(foldnum) + "_label",
    #     "./data/deeploc_train_pssm/",
    #     "./review_2ndRound_2/data/npzfiles/deeploc_80matrixY_train_fold" + str(foldnum) + ".npz",)
    #
    # [val_x_big, val_y_big, val_mask_big,val_ids_big] = endpad(
    #     "./data/deeploc_40nr_8folds/deeploc_40nr_val_fold" + str(foldnum) + "_seq",
    #     "./data/deeploc_40nr_8folds/deeploc_40nr_val_fold" + str(foldnum) + "_label",
    #     "./data/deeploc_train_pssm/",
    #     "./review_2ndRound_2/data/npzfiles/deeploc_80matrixY_val_fold" + str(foldnum) + ".npz")

    # get big data, using uniloc
    [train_x_big, train_y_big, train_mask_big, train_ids_big] = endpad(
        lv1_dir + "lv1_train_fold" + str(foldnum) + "_seq",
        lv1_dir + "lv1_train_fold" + str(foldnum) + "_lab",
        pssm_dir,
        "./review_2ndRound_6/data/npzfiles/lv1_train_fold" + str(foldnum) + "_seq.npz")

    [val_x_big, val_y_big, val_mask_big, val_ids_big] = endpad(
        lv1_dir + "lv1_val_fold" + str(foldnum) + "_seq",
        lv1_dir + "lv1_val_fold" + str(foldnum) + "_lab",
        pssm_dir,
        "./review_2ndRound_6/data/npzfiles/lv1_val_fold" + str(foldnum) + "_seq.npz")

    batch_size = 128
    print("doing " + str(foldnum) + "th fold")
    model_big, model_small = singlemodel(train_x)

    filepath_acc_big_lv1 = output_dir+"fold" + str(
        foldnum) + "_big_lv1_acc-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

    filepath_acc_small_lv2 = output_dir+"fold" + str(
        foldnum) + "_small_lv2_acc-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

    filepath_loss_big_lv1 = output_dir+"fold" + str(
        foldnum) + "_big_lv1_loss-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

    filepath_loss_small_lv2 = output_dir+"fold" + str(
        foldnum) + "_small_lv2_loss-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"


    checkpoint_acc_big_lev1 = ModelCheckpoint(filepath_acc_big_lv1, monitor='val_acc', save_best_only=True,
                                          mode='max',
                                          save_weights_only=True, verbose=1)

    checkpoint_acc_small_lev2 = ModelCheckpoint(filepath_acc_small_lv2, monitor='val_lev2_acc', save_best_only=True,
                                          mode='max',
                                          save_weights_only=True, verbose=1)
    
    checkpoint_loss_big_lev1 = ModelCheckpoint(filepath_loss_big_lv1, monitor='val_loss', save_best_only=True,
                                          mode='min',
                                          save_weights_only=True, verbose=1)
    
    checkpoint_loss_small_lev2 = ModelCheckpoint(filepath_loss_small_lv2, monitor='val_lev2_loss', save_best_only=True,
                                          mode='min',
                                          save_weights_only=True, verbose=1)
    
    
    for i in range(80):
        # train small model
        print("epoch "+str(i)+"\n")
        fitHistory_batch_small = model_small.fit([train_x, train_mask.reshape(-1, 1000, 1)],
                                                 [train_y,getTrue4out1(train_y)],
                                                 batch_size=batch_size, epochs=1,
                                                 validation_data=(
                                                 [val_x, val_mask.reshape(-1, 1000, 1)], [val_y,getTrue4out1(val_y)]),
                                                 callbacks=[checkpoint_acc_small_lev2,checkpoint_loss_small_lev2],verbose=1)
        
        # train big model
        fitHistory_batch_big = model_big.fit([train_x_big, train_mask_big.reshape(-1, 1000, 1)],
                                             [getTrue4out1(train_y_big)],
                                             batch_size=batch_size, epochs=1,
                                             validation_data=(
                                             [val_x_big, val_mask_big.reshape(-1, 1000, 1)], [getTrue4out1(val_y_big)]),
                                             callbacks=[checkpoint_acc_big_lev1,checkpoint_loss_big_lev1], verbose=1)



def train_var(input_var,pssm_dir,output_dir,foldnum):
    # get small data
    [train_x,train_y,train_mask]=endpad(input_var+"deeploc_40nr_train_fold"+str(foldnum)+"_seq",
                                        input_var+"deeploc_40nr_train_fold"+str(foldnum)+"_label",
                                        pssm_dir)
    [val_x,val_y,val_mask]=endpad(input_var+"deeploc_40nr_val_fold"+str(foldnum)+"_seq",
                                  input_var+"deeploc_40nr_val_fold"+str(foldnum)+"_label",
                                  pssm_dir)
    batch_size = 128
    print("doing " + str(foldnum) + "th fold")
    model = var_model(train_x)

    filepath_acc = output_dir+"fold" + str(
        foldnum) + "acc-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint_acc = ModelCheckpoint(filepath_acc, monitor='val_accuracy', save_best_only=True, mode='max',
                                 save_weights_only=True, verbose=1)
    fitHistory_batch = model.fit([train_x,train_mask.reshape(-1,1000,1)],getTrue4out1(train_y),
                                 batch_size=batch_size, epochs=60,
                                 validation_data=([val_x,val_mask.reshape(-1,1000,1)], getTrue4out1(val_y)),
                                 callbacks=[checkpoint_acc],verbose=1)




def main():
    parser=argparse.ArgumentParser(description='MULocDeep: interpretable protein localization classifier at sub-cellular and sub-organellar levels')
    parser.add_argument('--lv1_input_dir', dest='lv1_dir', type=str, help='sub-cellular training data, contains 8 folds protein sequences and labels', required=False)
    parser.add_argument('--lv2_input_dir', dest='lv2_dir', type=str,
                        help='sub-cellular training data, contains 8 folds protein sequences and labels', required=False)
    parser.add_argument('--input_dir', dest='var_dir', type=str,
                        help='data for traing the variant model, contains 8 folds protein sequences and labels', required=False)
    parser.add_argument('--MULocDeep_model', dest='modeltype', action='store_true',
                        help='Add this to train the MULocDeep model, otherwise train a variant model', required=False)
    parser.add_argument('--model_output', dest='outputdir', type=str, help='the name of the directory where the trained model stores', required=True)
    parser.add_argument('-existPSSM', dest='existPSSM', type=str,
                        help='the name of the existing PSSM directory if there is one.', required=False, default="")
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    model_type=args.modeltype
    input_lv1=args.lv1_dir
    input_lv2 = args.lv2_dir
    input_var=args.var_dir
    outputdir=args.outputdir
    existPSSM = args.existPSSM

    if model_type==True:
        if not input_lv1[len(input_lv1) - 1] == "/":
            input_lv1 = input_lv1 + "/"
        if not input_lv2[len(input_lv2) - 1] == "/":
            input_lv2 = input_lv2 + "/"
        if not outputdir[len(outputdir) - 1] == "/":
            outputdir = outputdir + "/"
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        if existPSSM != "":
            if not existPSSM[len(existPSSM) - 1] == "/":
                existPSSM = existPSSM + "/"
        if ((existPSSM == "") or (not os.path.exists(existPSSM))):
            ts = calendar.timegm(time.gmtime())
            pssmdir = outputdir + str(ts) + "_pssm/"
            if not os.path.exists(pssmdir):
                os.makedirs(pssmdir)
            process_input_train(input_lv1 + "processed_deeploc_train_S_seq", pssmdir)
            process_input_train(input_lv2 + "uniprot_10clas_lv2only_v2_seq.txt", pssmdir)
            for foldnum in range(8):
                train_MULocDeep(input_lv1, input_lv2, pssmdir, outputdir, foldnum)
        else:
            for foldnum in range(8):
                train_MULocDeep(input_lv1, input_lv2, existPSSM, outputdir, foldnum)
    elif model_type==False:
        if not input_var[len(input_var) - 1] == "/":
            input_var = input_var + "/"
        if not outputdir[len(outputdir) - 1] == "/":
            outputdir = outputdir + "/"
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        if existPSSM != "":
            if not existPSSM[len(existPSSM) - 1] == "/":
                existPSSM = existPSSM + "/"
        if ((existPSSM == "") or (not os.path.exists(existPSSM))):
            ts = calendar.timegm(time.gmtime())
            pssmdir = outputdir + str(ts) + "_pssm/"
            if not os.path.exists(pssmdir):
                os.makedirs(pssmdir)
            process_input_train(input_var + "processed_deeploc_train_S_seq", pssmdir)
            for foldnum in range(8):
                train_var(input_var, pssmdir, outputdir, foldnum)
        else:
            for foldnum in range(8):
                train_var(input_var, existPSSM, outputdir, foldnum)



if __name__ == "__main__":
    main()