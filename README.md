# tripletMIL

## Environment: 
* Linux (Tested on Ubuntu 20.04)
* Python==3.9.12
* torch==1.11.0+cu113, tensorboard==2.9.0, numpy==1.21.6, scikit-learn==1.0.2


## Prepare the dataset: 
The dataset should be organised in a way that the feature vectors of patches from the same patient are stored in a pickle file with the filename {patient_id}.pkl. 

Alternatively, you can modify the data_utils.py file to read from your own dataset.

## Steps: 
1. Fill the arguments.py according to your setting.
2. Run tripletMIL_training.py