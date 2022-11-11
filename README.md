# MeD-Net
The codes are for the work "An artificial intelligence system for melioidosis pneumonia diagnosis and prognosis using computed tomography"

## Pre-requirements
The codebase is tested on the following setting.
* Python>=3.7
* PyTorch>=1.6.0
* torchvision>=0.7
* einops>=0.4.1
* scikit-learn>=1.0.2
* shap>=0.40.0
* xgboost>=1.6.0

## Train and Test
For easier use this project, the codes are organized in two sub-directories.
* **`lung_lesion_segmentation`**
Contains code for segmentation of lung and lesion.
### LungSeg_train.py 
Train the lung segmentation model.
### LesSeg_train.py 
Train the lesion segmentation model.
### Seg_test.py
Test output results of segmentation.

* **`melio_prediction`**
Contains code for pneumonia diagnosis and prognosis
### train_normal_other_2.py 
Train the pneumonia diagnosis model for 2 classes (normal vs pneumonia).
### train_meli_covid_cap_3.py 
Train the pneumonia diagnosis model for 3 classes (melioidosis vs COVID-19 vs CAP).
### train_prog.py
train the melioidosis prognosis model
