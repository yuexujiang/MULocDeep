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





def process_eachseq(seq,pssmfile,mask_seq,new_pssms):
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
    


def endpad(seqfile, pssmdir):
        new_pssms = []
        mask_seq = []
        ids=[]
        f = open(seqfile, "r")
        index=0
        for line in f:
            pssmfile = pssmdir + str(index) + "_pssm.txt"
            if ">" in line:
                if index!=0:
                   ids.append(id)
                   process_eachseq(seq,pssmfile,mask_seq,new_pssms)
                
                seq=''
                id = line.strip()[1:]
            else:
               seq+=line.strip()
            
            index+=1
        
        ids.append(id)
        process_eachseq(seq,pssmfile,mask_seq,new_pssms)
        x = np.array(new_pssms)
        mask = np.array(mask_seq)
        return [x, mask,ids]





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
        os.makedirs(outputdir)
    existPSSM=args.existPSSM
    if existPSSM!="":
       if not existPSSM[len(existPSSM) - 1] == "/":
         existPSSM = existPSSM + "/"

    if ((existPSSM=="")or(not os.path.exists(existPSSM))):
        ts = calendar.timegm(time.gmtime())
        pssmdir=outputdir+str(ts)+"_pssm/"
        if not os.path.exists(pssmdir):
          os.makedirs(pssmdir)
        process_input_user(inputfile,pssmdir)  # generate pssm
        [test_x, test_mask, test_ids] = endpad(inputfile, pssmdir)
    else:
        [test_x, test_mask, test_ids] = endpad(inputfile, existPSSM)

    pred_big = np.zeros((test_x.shape[0],10))
    att_matrix_N = np.zeros((8, test_x.shape[0], 1000))
    cross_pred_small = np.zeros((test_x.shape[0], 8, 10, 8))

    for foldnum in range(8):
        model_big, model_small = singlemodel(test_x)
        model_small.load_weights('./gpu_model_40nr/fold' + str(foldnum) + '_big_lv1_acc-weights.hdf5')
        
        cross_pred_small[:, foldnum]= model_small.predict([test_x, test_mask.reshape(-1, 1000, 1)])[0]
        model_att = Model(inputs=model_big.inputs, outputs=model_big.layers[-11].output[1])
        att_pred = model_att.predict([test_x, test_mask.reshape(-1, 1000, 1)])
        att_matrix_N[foldnum, :] = att_pred.sum(axis=1) / 41

    att_N = att_matrix_N.sum(axis=0) / 8

    pred_small = cross_pred_small.sum(axis=1) / 8 #[?,10,8]
    pred_small_c = pred_small.copy()
    pred_big_c=pred_small_c.max(axis=-1)   #[?, 10]
    
    cutoff = np.array([[0.5, 0.2, 0.5, 0.3, 0.5, 0.4, 0.4, 0.5],
                      [0.3, 0.5, 0.3, 0.1, 0.5, 0.3, 0.5, 0.2],
                      [1, 0.5, 1, 1, 1, 1, 1, 1],
                      [0.5, 0.1, 0.5, 0.4, 0.4, 1, 1, 1],
                      [0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 1, 1],
                      [0.4, 0.5, 0.5, 0.2, 0.1, 1, 1, 1],
                      [0.2, 0.3, 0.1, 0.3, 0.3, 1, 1, 1],
                      [0.2, 0.3, 0.1, 0.1, 1, 1, 1, 1],
                      [0.1, 1, 1, 1, 1, 1, 1, 1],
                      [0.3, 1, 1, 1, 1, 1, 1, 1]])
    
    pred_small[pred_small >= cutoff]=1.0
    pred_small[pred_small < cutoff] =0.0
    for i in range(pred_small.shape[0]):
        index=((pred_small_c[i]>=cutoff).sum(axis=-1))>0
        pred_big[i][index] =1.0
        if pred_small[i].sum() == 0:
            index = pred_small_c[i].max(axis=-1).argmax()
            index2 = pred_small_c[i][index].argmax()
            pred_small[i][index, index2] = 1.0
            pred_big[i][index] = 1.0
                
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
    f = open(inputfile)
    list_seq = []
    seqindex=0
    for line in f:
        if ">" in line:
            if seqindex!=0:
               list_seq.append(seq)
            
            seq=''
        else:
            seq+=line.strip()
        
        seqindex+=1
    
    list_seq.append(seq)
    for i in test_ids:
        end = int(test_mask[ind].sum())
        j = ind 
        seq = list(list_seq[j].strip())
        f1.write(">" + i + "\n")
        if len(seq) <= 1000:
          for p in att_N[ind][0:end]:
            f1.write(str(p) + " ")
        else:
          dif = len(seq) - end
          w = np.concatenate((att_N[ind][0:500], np.zeros(dif), att_N[ind][500:]), axis=0)
          for p in w:
            f1.write(str(p) + " ")
        f1.write("\n")
        ind=ind+1
    f1.close()
    
    if att_draw:
        i = 0
        for p in test_ids:
          ind = int(test_mask[i].sum())
          seq = list(list_seq[i].strip())
          plt.clf()
          plt.cla()
          plt.xticks(np.linspace(1, len(seq), len(seq)), seq)
          if len(seq)<=1000:
            plt.plot(np.linspace(1, ind, ind), att_N[i][:ind])
          else:
            dif=len(seq)-ind
            w=np.concatenate((att_N[i][0:500],np.zeros(dif),att_N[i][500:]),axis=0)
            plt.plot(np.linspace(1, len(seq), len(seq)),w)
          plt.savefig(outputdir+p+'.png')
          i=i+1


if __name__ == "__main__":
    main()
