# MULocDeep
MULocDeep is a deep learning model for protein localization prediction at both sub-cellular level and sub-organellar level. It also has the ability to interpret localization mechanim at a amino acid resolution. Users can go to our webserver at xxx.xxx. This repository is for running MuLocDeep locally.
## Installation

  - Installation has been tested in Windows, Linux and Mac OS X with Python 3.7.4. 
  - Keras version: 2.3.0
  - For predicting, GPU is not required. For training a new model, the Tensorflow-gpu version we tested is: 1.13.1
  - Users need to install the NCBI Blast+ for the PSSM. The download link is https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/. The version we tested is 2.9.0. The database is already in the 'db' folder.

## Running on GPU or CPU

>If you want to use GPU, you also need to install [CUDA]( https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn); refer to their websites for instructions. 
CPU is only suitable for prediction not training. 
## Usage:
### Predict protein localization
Predicting protein localization using the pretrained model is fairly simple. There are several parameters that need to be specified by users. They are explained as below:
  - --input: 
To train a model, users should have .fasta file for protein sequences and domain boundary annotations on corresponding protein sequences.
Please refer to the sample_data_seq.txt and sample_data_label.txt as examples. Then use the "dataprocess.pl" to transformat the data. (run "perl dataprocess.pl -h" to see the helps) After that, the processed data can be used as input for "train.py" to train a model (run "python train.py -h" to see the helps). Note: the requirement of packages that imported in the code need to be met. 
#### Examples (using our provided example data): 
 
```sh
perl dataprocess.pl -input_seq sample_data_seq.txt -input_label sample_data_label.txt -output_seq processed_seq.txt -output_label processed_label.txt

python train.py -seqfil processed_seq.txt -labelfile processed_label.txt -model-prefix custom_model.h5
```

custom_model.h5 is the model generated, users can use this file to predict and can also use our pre-trained model that mentioned in our paper. The pre-trained model was saved in file "foldmodel_bilstmwrapper_4sum200_80_40nr_sliwin.h5".

## 2. Predict

To predict domain boundary for protein sequences, firstly, users need to transformat the .fasta sequence using "dataprocess.pl" (run "perl dataprocess.pl -h" to see the helps) and using "predict.py" to predict for protein sequences (run "python predict.py -h" to see the helps). Either users' own model or the model we provided can be used for prediction.
#### Examples (using our provided example data):
For GPU users, predict.py will use our pre-trained GPU model "foldmodel_bilstmwrapper_4sum200_80_40nr_sliwin.h5":
```sh
perl dataprocess.pl -input_seq sample_data_seq.txt -output_seq processed_seq.txt

python predict.py -input processed_seq.txt -output predict_result.txt
```
For CPU users, be sure to add -model-prefix cpu_model.h5 when call predict.py:
```sh
perl dataprocess.pl -input_seq sample_data_seq.txt -output_seq processed_seq.txt

python predict.py -input processed_seq.txt -output predict_result.txt -model-prefix cpu_model.h5
```
Or users can use the custom model trained (as shown in 1) by their own data to predict.
 ```sh
python predict.py -input processed_seq.txt -output predict_result.txt -model-prefix custom_model.h5
```

========================================================================================================================================
#### If you find the codes or our method is useful, please cite our paper "DeepDom: Predicting protein domain boundary from sequence alone using stacked bidirectional LSTM".
