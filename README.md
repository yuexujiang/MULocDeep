# MULocDeep
MULocDeep is a deep learning model for protein localization prediction at both sub-cellular level and sub-organellar level. It also has the ability to interpret localization mechanism at a amino acid resolution. Users can go to our webserver at xxx.xxx for localiztion prediction and visualization. This repository is for running MuLocDeep locally.
## Installation

  - Installation has been tested in Windows, Linux and Mac OS X with Python 3.7.4. 
  - Keras version: 2.3.0
  - For predicting, GPU is not required. For training a new model, the Tensorflow-gpu version we tested is: 1.13.1
  - Users need to install the NCBI Blast+ for the PSSM. The download link is https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/. The version we tested is 2.9.0. The database can be downloaed at https://drive.google.com/drive/folders/19gbmtZAz1kyR76YS-cvXSJAzWJdSJniq?usp=sharing. Put the downloaded 'db' folder in the same folder as other files in this project.

## Running on GPU or CPU

If you want to use GPU, you also need to install [CUDA]( https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn); refer to their websites for instructions.
CPU is only suitable for prediction not training. 
## Usage:
### Predict protein localization
Predicting protein localization using the pretrained model is fairly simple. There are several parameters that need to be specified by users. They are explained as below:
  - --input filename: The sequences of the proteins that users want to predict. Should be in fasta format.
  - --output dirname: The name of the folder where the output would be saved.
  - --existPSSM dirname: This is optional. If the pssm of the protein sequences are already calculated, users can specify the path to that folder. This will save a lot of time, since calculating pssm is time consuming. Otherwise, the prediction program will automaticlly start to generate the pssm for the prediction.
  - --att: Add this if users want to see the attention visualization. It is for interpreting the localization mechanism. Amino acids with high attention weights are considered related to the sorting signal.
  - --no-att: Add this, no attention visualization figure.

#### Example (using our provided example data): 
 
```sh
python predict.py -input ./wiki_seq.txt -output ./test --att
```

## Contacts
If you ever have any question or problem using our tool, please contact us.
  - Email: yjm85@mail.missouri.edu
